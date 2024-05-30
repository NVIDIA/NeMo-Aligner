# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.nlp.gpt.megatron_gpt_regression_reward_model import MegatronGPTRegressionRewardModel
from nemo_aligner.utils.train_utils import set_sync_funcs


class MegatronGPTCategoricalRewardModel(MegatronGPTRegressionRewardModel):
    """
    Megatron GPT Regression Reward Model Training. 
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # there is a problem with apex ignoring the mask on the older models
                # so we will always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("inputs", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "lengths", "loss_mask"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            forward_args = {
                "input_ids": batch["inputs"],
                "lengths": batch["lengths"],
                "position_ids": batch["position_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": None,
            }

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor):
                # Loss per micro batch (ub).
                loss_for_ub, acc_chosen = self.loss_func(output_tensor, batch["labels"])
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    num_valid_tokens_in_ub = batch["loss_mask"].sum()
                    if loss_for_ub.isnan():
                        assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )

                    return (
                        loss_for_ub,
                        {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu,},
                    )
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    reduced_acc = average_losses_across_data_parallel_group([acc_chosen])

                    return (
                        loss_for_ub,
                        {"avg": reduced_loss, "acc": reduced_acc,},
                    )

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    # def loss_func(self, output_tensor, label_tensor):
    #     """
    #     output_tensor: [B, num_attributes * num_category]
    #     label_tensor: [B, num_attributes]
    #     """
    #     mask_val = int(self.cfg.get("regression", {}).get("loss_mask_val", -100))
    #     mask = label_tensor != mask_val
    #     num_valid_attributes = mask.float().sum()
    #     assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"

    #     # Reshape output_tensor to [B * num_attributes, num_category]
    #     output_tensor = output_tensor.view(
    #         output_tensor.size(0) * label_tensor.size(1), self.cfg.get("categorical", {}).num_category
    #     )
    #     # Flatten label_tensor to [B * num_attributes]
    #     label_tensor = label_tensor.view(-1).long()
    #     criterion = torch.nn.CrossEntropyLoss(ignore_index=mask_val)
    #     # Calculate the loss
    #     loss = criterion(output_tensor, label_tensor)

    #     mask = mask.view(-1)
    #     _, predictions = torch.max(output_tensor, 1)
    #     filtered_predictions = predictions[mask]
    #     filtered_labels = label_tensor[mask]
    #     correct_predictions = (filtered_predictions == filtered_labels).float()
    #     accuracy = correct_predictions.sum() / correct_predictions.numel()

    #     return loss, accuracy

    def loss_func(self, output_tensor, label_tensor):
        """
        output_tensor: [B, num_attributes * num_category]
        label_tensor: [B, num_attributes]
        """
        mask_val = int(self.cfg.get("regression", {}).get("loss_mask_val", -100))
        mask = label_tensor != mask_val
        num_valid_attributes = mask.float().sum()
        assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"

        # Convert label_tensor to binary encoding
        num_category = self.cfg.get("categorical", {}).num_category
        binary_label_tensor = torch.zeros(
            label_tensor.size(0), label_tensor.size(1), num_category, device=label_tensor.device
        )
        for i in range(num_category):
            binary_label_tensor[:, :, i] = (label_tensor >= i).float()

        # Reshape output_tensor to [B * num_attributes, num_category]
        output_tensor = output_tensor.view(output_tensor.size(0) * label_tensor.size(1), num_category)
        # Flatten binary_label_tensor to [B * num_attributes, num_category]
        binary_label_tensor = binary_label_tensor.view(-1, num_category)

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # Calculate the loss
        loss = criterion(output_tensor, binary_label_tensor)
        loss = loss * mask.view(-1, 1).float()  # Apply mask
        loss = loss.sum() / num_valid_attributes  # Average over valid attributes

        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(output_tensor) >= 0.5
            filtered_predictions = predictions[mask.view(-1)]
            filtered_labels = binary_label_tensor[mask.view(-1)]
            correct_predictions = (filtered_predictions == filtered_labels).float().mean(dim=1)
            accuracy = correct_predictions.sum() / correct_predictions.numel()

        return loss, accuracy

    # def loss_func(self, output_tensor, label_tensor):
    #     """
    #     output_tensor: [B, num_attributes * num_category]
    #     label_tensor: [B, num_attributes]
    #     """
    #     mask_val = int(self.cfg.get("regression", {}).get("loss_mask_val", -100))
    #     mask = label_tensor != mask_val
    #     num_valid_attributes = mask.float().sum()
    #     assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"

    #     num_category = self.cfg.get("categorical", {}).num_category

    #     # Define the new label encoding for arbitrary num_category
    #     # 0 -> [1,0,0,0,0]
    #     # 1 -> [1,1,0,0,0,]
    #     # ...
    #     def label_encoding(label, num_category):
    #         encoded_label = [0] * num_category
    #         for i in range(label + 1):
    #             encoded_label[i] = 1
    #         return encoded_label

    #     # Reshape output_tensor to [B * num_attributes, num_category]
    #     output_tensor = output_tensor.view(output_tensor.size(0) * label_tensor.size(1), num_category)

    #     # Convert label_tensor to the new encoding
    #     label_tensor_encoded = torch.zeros_like(output_tensor)
    #     for i, label in enumerate(label_tensor.view(-1).long()):
    #         label_tensor_encoded[i] = torch.tensor(label_encoding(label.item(), num_category))

    #     # Calculate the loss
    #     criterion = torch.nn.CrossEntropyLoss()
    #     loss = criterion(output_tensor, label_tensor_encoded)

    #     mask = mask.view(-1)
    #     _, predictions = torch.max(output_tensor, 1)
    #     filtered_predictions = predictions[mask]
    #     filtered_labels = label_tensor_encoded[mask]
    #     correct_predictions = (filtered_predictions == filtered_labels.argmax(1)).float()
    #     accuracy = correct_predictions.sum() / correct_predictions.numel()

    #     return loss, accuracy

    def get_loss_and_metrics(self, batch, forward_only):
        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
            acc_tensors_list = [loss_reduced["acc"] for loss_reduced in losses_reduced_per_micro_batch]

            if len(acc_tensors_list) == 1:
                acc_tensor = acc_tensors_list[0]
            elif len(acc_tensors_list) > 1:
                acc_tensor = torch.concat(acc_tensors_list)
            acc_mean = acc_tensor.mean()

        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            acc_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(acc_mean, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "acc": acc_mean,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

        return loss_mean.item(), metrics
