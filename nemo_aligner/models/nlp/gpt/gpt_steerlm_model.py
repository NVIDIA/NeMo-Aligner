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
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.train_utils import set_sync_funcs


def get_batch(data_iterator):
    """Generate a batch."""

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    batch = {
        "tokens": data["tokens"],
        "labels": data["labels"],
        "loss_mask": data["loss_mask"],
        "attention_mask": data["attention_mask"],
        "position_ids": data["position_ids"],
    }
    if "weight" in data:
        batch.update({"weight": data["weight"]})
    if "avg_num_valid_tokens_in_ub" in data:
        batch.update({"avg_num_valid_tokens_in_ub": data["avg_num_valid_tokens_in_ub"]})
    return batch


class GPTSteerLMModel(GPTSFTModel):
    def loss_func(self, loss_mask, output_tensor):
        # loss is the nll loss for each token in the batch
        # losses [batch_size, sequence_length]
        # calculate sum of log(p) for all tokens in a one batch item
        # this will be later used to calculate the baseline weights
        loss = torch.sum(output_tensor * loss_mask, dim=-1)
        return loss

    def weight_loss_func(self, loss_mask, avg_num_valid_tokens_in_ub, output_tensor, weight):
        # loss is the nll loss for each token in the batch
        losses = output_tensor
        loss_mask = loss_mask.float()
        # apply the per batch weight
        losses = losses * weight[:, None]
        # losses [batch_size, sequence_length]
        # loss per batch item, log(p) for all tokens in one batch item
        loss = torch.sum(losses * loss_mask, dim=-1)
        # average lop(p) by the number of valid tokens in the microbatch and scale it by the (number of microbatches / batch)
        loss = loss / avg_num_valid_tokens_in_ub
        # sum log(p) in microbatch
        loss = loss.sum()
        return loss

    def get_forward_output_only_for_weight_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            with torch.no_grad():
                batch = next(dataloader_iter)
                batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}

                forward_args = {
                    "input_ids": batch["tokens"],
                    "position_ids": batch["position_ids"],
                    "attention_mask": None if self.get_attention_mask_from_fusion else batch["attention_mask"],
                    "labels": batch["labels"],
                }

                output_tensor = model(**forward_args)

                def id_func(output_tensor):
                    nll = self.loss_func(batch["loss_mask"], output_tensor)
                    return output_tensor, {"loss": nll}

            return output_tensor, id_func

        return fwd_output_only_func

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            # Get data batch
            batch = get_batch(dataloader_iter)
            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids"))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "loss_mask", "weight", "avg_num_valid_tokens_in_ub"))
            if self.get_attention_mask_from_fusion and "attention_mask" in required_keys:
                required_keys.remove("attention_mask")
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                "input_ids": batch["tokens"],
                "position_ids": batch["position_ids"],
                "attention_mask": None if self.get_attention_mask_from_fusion else batch["attention_mask"],
                "labels": batch["labels"],
            }

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.weight_loss_func(
                    batch["loss_mask"], batch["avg_num_valid_tokens_in_ub"], output_tensor, batch["weight"]
                )
                reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                return loss_for_ub, {"avg": reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def compute_baselilne_weights(self, batch):
        # first compute the baseline weights
        set_sync_funcs(self, True)
        # calculate the baseline weights for each group of generated reponses
        # set the batch size for log likelihood calculation
        response_size = batch["num_responses"][0].item()
        gbs, seq_length = batch["tokens"].shape
        mbs = int(self.cfg.steerlm2.forward_micro_batch_size)
        num_of_microbatches = divide(gbs, mbs)
        group_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        data_iter = get_iterator_k_split(group_batch, num_of_microbatches)

        fwd_bwd_function = get_forward_backward_func()

        # first get the forward nll loss for each of the microbatches
        fwd_loss_fn = self.get_forward_output_only_for_weight_func()
        losses_forward = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_of_microbatches,
            forward_only=True,
            micro_batch_size=mbs,
            seq_length=seq_length,
        )
        distance = None
        if parallel_state.is_pipeline_last_stage():
            updated_weight = torch.concatenate([i["loss"] for i in losses_forward])
            base_weight = (
                -updated_weight.reshape(-1, response_size) - batch["log(Q(y|a,x))"].reshape(-1, response_size).cuda()
            )
            base_weight_p = torch.nn.functional.softmax(base_weight, dim=-1)
            ws_weight = batch["ws"].reshape(-1, response_size).cuda()
            weight = ws_weight - base_weight_p
            # group size is the number of responses generated
            loss_mbs = int(self.cfg.steerlm2.micro_batch_size)
            num_of_microbatches = gbs // loss_mbs
            # split updated weights into groups of size response_size
            num_of_batches = len(updated_weight) // response_size
            avg_num_valid_tokens_in_ub = (
                batch["loss_mask"]
                .reshape(-1, response_size, seq_length)
                .sum(axis=-1)
                .float()
                .mean(axis=-1, keepdim=True)
                .repeat([1, response_size])
            )
            avg_num_valid_tokens_in_ub = avg_num_valid_tokens_in_ub / (num_of_microbatches / num_of_batches)

            distance = ((ws_weight + 1e-8) * torch.log((ws_weight + 1e-8) / (base_weight_p + 1e-8))).sum(axis=1).mean()
            distance = average_losses_across_data_parallel_group([distance])

            batch["base_weight"] = base_weight_p.reshape(-1)
            batch["weight"] = weight.reshape(-1)
            batch["avg_num_valid_tokens_in_ub"] = avg_num_valid_tokens_in_ub.reshape(-1).cuda()
        return distance, seq_length, gbs

    def get_loss_and_metrics(self, batch, forward_only):
        """Take a data_iter which is an interator over the microbatches
            and return loss as well as metrics
        """
        distance, seq_length, gbs = self.compute_baselilne_weights(batch)

        fwd_bwd_function = get_forward_backward_func()

        set_sync_funcs(self, forward_only)
        mbs = int(self.cfg.steerlm2.micro_batch_size)
        number_of_microbatches = divide(gbs, mbs)

        # modify the iterator_k_split to iterate over all the data
        data_iter = get_iterator_k_split(batch, number_of_microbatches)

        fwd_loss_fn = self.get_forward_output_and_loss_func(forward_only)

        losses_reduced = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=number_of_microbatches,
            forward_only=forward_only,
            micro_batch_size=mbs,
            seq_length=seq_length,
        )

        # only the last stages of the pipeline return losses
        if parallel_state.is_pipeline_last_stage():
            loss_mean = torch.concat([loss["avg"] for loss in losses_reduced]).mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()
            distance = torch.tensor(0.0).cuda()
        # Logging
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(distance, get_last_rank())
        loss_value = loss_mean.detach().item()
        metrics = {"loss": loss_value, "distance": distance.item()}
        return loss_value, metrics
