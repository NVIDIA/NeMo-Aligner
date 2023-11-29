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


import warnings
from typing import List, Union

import torch
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_num_microbatches
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.optim.distributed_adam import _str_to_dtype
from nemo.utils import AppState, logging
from nemo_aligner.models.alignable_interface import Inferrable, SupervisedInterface
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel
from nemo_aligner.utils.distributed import broadcast_2d_tensor, gather_tensor
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)


class MegatronGPTRewardModel(MegatronGPTModel, SupervisedInterface, Inferrable):
    """
    Megatron GPT Reward Model Training.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        if self.cfg.pipeline_model_parallel_size > 1 and not self.cfg.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )
        self.automatic_optimization = False

        reward_standardization = cfg.get("reward_standardization")

        self.enable_standardization = reward_standardization.enable if reward_standardization is not None else False

        if self.enable_standardization:
            self.rew_mean = cfg.reward_standardization.mean
            self.rew_std = cfg.reward_standardization.std

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""

        force_head_dtype = self.cfg.get("force_head_dtype", torch.float32)
        head_dtype = None if force_head_dtype is None else _str_to_dtype(force_head_dtype)
        if self.cfg.get("megatron_amp_O2", False) and (head_dtype is None or torch.finfo(head_dtype).bits < 32):
            logging.warning(
                "When `megatron_amp_O2` is enabled, it is recommended to set `force_head_dtype=32` "
                "to improve the convergence and accuracy of the model"
            )

        if self.cfg.get("share_embeddings_and_output_weights", False):
            logging.warning(
                "`share_embeddings_and_output_weights` is not supported with the reward model since we don't use the "
                "normal output layer. Overriding it to False"
            )

        model = GPTRewardModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=self.cfg.get("override_vocab_size", self.padded_vocab_size),
            max_sequence_length=self.cfg.get("encoder_seq_length", 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            rotary_percent=self.cfg.get("rotary_percentage", 1.0),
            seq_len_interpolation_factor=self.cfg.get("seq_len_interpolation_factor", None),
            output_sequence=self.cfg.get("output_sequence", False),
            use_avg_pool=self.cfg.get("use_avg_pool", False),
            head_dtype=head_dtype,
            num_attributes=self.cfg.get("regression", {}).get("num_attributes", 1),
            attribute_weights=self.cfg.get("regression", {}).get("attribute_weights", None),
            merge_attributes=self.cfg.get("regression", {}).get("merge_attributes", False),
        )
        return model

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
                    required_keys.update(("chosen", "rejected", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("chosen_length", "rejected_length", "loss_mask"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # only do the torch.cat if it's available
            lengths, tokens = None, None
            position_ids = (
                torch.cat((batch["position_ids"], batch["position_ids"]), dim=0)
                if batch["position_ids"] is not None
                else None
            )

            if batch["chosen_length"] is not None and batch["rejected_length"] is not None:
                # Combine chosen and rejected lengths and then tokens.
                lengths = torch.cat((batch["chosen_length"], batch["rejected_length"]), dim=0)

            if batch["chosen"] is not None and batch["rejected"] is not None:
                tokens = torch.cat((batch["chosen"], batch["rejected"]), dim=0)

            forward_args = {
                "input_ids": tokens,
                "lengths": lengths,
                "position_ids": position_ids,
                "attention_mask": batch["attention_mask"],
                "labels": None,
            }

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            @torch.no_grad()
            def gather_and_split_rewards(rewards_out):
                rewards_out = rewards_out.detach()

                dp_group = parallel_state.get_data_parallel_group()
                output_list = [torch.zeros_like(rewards_out) for _ in range(dp_group.size())]

                # gather it to compute the std later on
                torch.distributed.all_gather(output_list, output_tensor, group=dp_group)

                # output_list is a list of tensors with len == number of DP workers
                # we need to split each of these tensors and concat them back into a single tensor for chosen and rejected rewards
                split_iter = map(self.split_output_tensor, output_list)

                # go from [(out_chosen_dp0, out_rejected_dp0), (out_chosen_dp1, out_rejected_dp1)] ->
                # [out_chosen_dp0, out_chosen_dp1], [out_rejected_dp0, out_rejected_dp1]
                out_chosen, out_rejected = map(torch.cat, zip(*split_iter))

                return out_chosen.flatten(), out_rejected.flatten()

            def loss_func(output_tensor):
                # Loss per micro batch (ub).
                loss_for_ub, acc_chosen = self.loss_func(output_tensor)
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
                    out_chosen, out_rejected = gather_and_split_rewards(output_tensor)

                    return (
                        loss_for_ub,
                        {
                            "loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu,
                            "out_chosen": out_chosen,
                            "out_rejected": out_rejected,
                        },
                    )
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    reduced_acc = average_losses_across_data_parallel_group([acc_chosen])

                    out_chosen, out_rejected = gather_and_split_rewards(output_tensor)

                    return (
                        loss_for_ub,
                        {
                            "avg": reduced_loss,
                            "acc": reduced_acc,
                            "out_chosen": out_chosen,
                            "out_rejected": out_rejected,
                        },
                    )

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor):
        out_chosen, out_rejected = torch.split(output_tensor.float(), output_tensor.shape[0] // 2, dim=0)
        return out_chosen, out_rejected

    def loss_func(self, output_tensor):
        out_chosen, out_rejected = self.split_output_tensor(output_tensor)
        comp = out_chosen > out_rejected
        acc_chosen = torch.sum(comp) / comp.shape[0]
        loss = -torch.nn.functional.logsigmoid(out_chosen - out_rejected).mean()
        return loss, acc_chosen

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
            micro_batch_size=self.cfg.micro_batch_size
            * 2,  # each minibatch has 2 comparisons so tensor shape will be mbs * 2
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            rewards_chosen = torch.cat([item["out_chosen"] for item in losses_reduced_per_micro_batch])
            rewards_rejected = torch.cat([item["out_rejected"] for item in losses_reduced_per_micro_batch])

            rewards_all = torch.cat((rewards_chosen, rewards_rejected))
            rewards_chosen_mean = rewards_chosen.mean()
            rewards_rejected_mean = rewards_rejected.mean()
            rewards_all_mean = rewards_all.mean()
            rewards_all_std = rewards_all.std()

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

            rewards_chosen_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_rejected_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(acc_mean, get_last_rank())

        torch.distributed.broadcast(rewards_chosen_mean, get_last_rank())
        torch.distributed.broadcast(rewards_rejected_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "acc": acc_mean,
            "rewards_chosen_mean": rewards_chosen_mean,
            "rewards_rejected_mean": rewards_rejected_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

        return loss_mean.item(), metrics

    def on_load_checkpoint(self, checkpoint) -> None:
        """NOTE: Have to set strict to False because we have a rm head
        """
        # mcore uses distributed checkpointing
        if "state_dict" in checkpoint and checkpoint["state_dict"]:
            for index, module in enumerate(self.get_gpt_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
                else:
                    checkpoint_state_dict = checkpoint["state_dict"]
                # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                checkpoint_state_dict = {
                    key.replace("model.", ""): checkpoint_state_dict.pop(key)
                    for key in list(checkpoint_state_dict.keys())
                }
                module.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
            # see NLPModel.on_load_checkpoint
            checkpoint["state_dict"] = {}

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)

    def infer(
        self,
        inputs: Union[List[str], List[List[int]]] = None,
        sequence_length: List[int] = None,
        add_EOS: bool = False,
    ):
        tokenizer = self.tokenizer
        max_seq_length = self.cfg.encoder_seq_length

        exceeded = None
        context_tokens_tensor = None
        context_length_tensor = None
        if torch.distributed.get_rank() == 0:
            if sequence_length is not None:
                exceeded = [False for _ in range(len(inputs))]
                context_tokens_tensor = torch.tensor(inputs).cuda()
                context_length_tensor = torch.tensor(sequence_length).cuda()
            else:
                context_tokens_tensor, context_length_tensor, exceeded = tokenize_batch(
                    tokenizer, inputs, max_seq_length, False, add_EOS=add_EOS
                )
            if context_length_tensor.dim() == 1:
                context_length_tensor = context_length_tensor.unsqueeze(-1)

        context_length_tensor = broadcast_2d_tensor(context_length_tensor, src=0, group=None, dtype=torch.int64)
        if context_length_tensor.dim() == 2:
            context_length_tensor = context_length_tensor.squeeze(-1)

        context_tokens_tensor = broadcast_2d_tensor(context_tokens_tensor, src=0, group=None, dtype=torch.int64)

        # Select subset of data needed for this rank.
        dp_size = parallel_state.get_data_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()
        context_length_tensor = context_length_tensor.chunk(dp_size)[dp_rank]
        context_tokens_tensor = context_tokens_tensor.chunk(dp_size)[dp_rank]

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            context_tokens_tensor,
            tokenizer.eos_id,
            self.cfg.get("reset_position_ids", False),
            self.cfg.get("reset_attention_mask", False),
            self.cfg.get("eod_mask_loss", False),
        )
        micro_batch_size = context_tokens_tensor.shape[0]
        sequence_length = context_tokens_tensor.shape[1]

        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )
        attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
        rewards = self.forward_step(
            [context_tokens_tensor, context_length_tensor, position_ids, attention_mask_repeat],
            micro_batch_size,
            sequence_length,
        )

        if parallel_state.is_pipeline_last_stage():
            rewards = rewards[0]["reward"]

            # Standardize values to subtract a bias.
            if self.enable_standardization:
                rewards = (rewards - self.rew_mean) / self.rew_std

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            rewards = broadcast_2d_tensor(
                rewards,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                parallel_state.get_pipeline_model_parallel_group(),
            )

        rewards_list = gather_tensor(
            rewards, dst=parallel_state.get_data_parallel_src_rank(), group=parallel_state.get_data_parallel_group()
        )

        return rewards_list, exceeded

    def forward_step(self, batch, micro_batch_size, sequence_length):
        set_sync_funcs(self, forward_only=True)

        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=self.get_forward_output_only_func(),
            data_iterator=iter([batch]),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
        )
        return output_tensor

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            (tokens, length, position_ids, attention_mask,) = next(batch)
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            attention_mask = attention_mask.cuda()
            output_tensor = model(tokens, length, position_ids, attention_mask)

            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def id_func(output_tensor):
                return output_tensor, {"reward": output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def prepare_for_inference(self):
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)

    def finish_inference(self):
        # training will onload the adam states, no need to onload it here
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()
        set_train(self)
