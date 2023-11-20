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

from contextlib import nullcontext
from enum import Enum

import torch
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import Float16Module
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState
from nemo_aligner.models.alignable_interface import CriticModelInterface
from nemo_aligner.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel
from nemo_aligner.utils.train_utils import set_sync_funcs
from nemo_aligner.utils.utils import masked_mean, offload_distributed_adam, swap_dict


class StateDictState(Enum):
    """Enum to determine which model state is loaded
    """

    CRITIC = 0
    REWARD = 1


class MegatronGPTCriticModel(MegatronGPTRewardModel, CriticModelInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # filled by the examples script
        self.rm_state_dict = None

        # for the critic states on cpu
        self.cpu_state_dict = None

        # for distributed adam offload
        self.distributed_adam_offload_manager = None

        self.loaded_state_dict = StateDictState.CRITIC
        self.to_offload_adam_states = self.cfg.get("offload_adam_states")
        self.clip_val = self.cfg.get("loss_clip_val")

    def prepare_for_inference(self):
        super().prepare_for_inference()
        self.offload_adam_states()

    def prepare_for_training(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.global_batch_size,
            micro_batch_size=self.cfg.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()
        self._load_critic()

    def get_loss_and_metrics(self, batch, forward_only):
        sequence_length = batch["tokens"].size(-1)
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=sequence_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        metrics = {
            "loss": loss_mean.item(),
        }

        return loss_mean.item(), metrics

    def get_forward_output_and_loss_func(self):
        # validation step is not used
        def fwd_output_and_loss_func(data_iterator, model):
            batch = next(data_iterator)
            tokens = batch["tokens"]
            returns = batch["returns"]
            prev_values = batch["prev_values"]
            mask = batch["mask"]

            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                tokens, self.tokenizer.eos_id, False, True, False,
            )

            attention_mask = attention_mask[0:1].cuda(non_blocking=True)

            # when using PP, set the unused variables to None just to be safe
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if parallel_state.is_pipeline_first_stage():
                    returns, prev_values, mask = None, None, None

                    tokens, position_ids = map(lambda x: x.cuda(non_blocking=True), (tokens, position_ids))

                elif parallel_state.is_pipeline_last_stage():
                    tokens, position_ids = None, None

                    # loss needs these 3 values
                    prev_values, mask, returns = map(lambda x: x.cuda(non_blocking=True), (prev_values, mask, returns))

                else:
                    # intermediates don't need anything
                    tokens, position_ids, returns, prev_values, mask = [None] * 5

            output = model(
                input_ids=tokens, lengths=None, position_ids=position_ids, attention_mask=attention_mask, labels=None,
            )

            if not parallel_state.is_pipeline_last_stage():
                output = output.to(dtype=self.autocast_dtype)

            def loss_func(curr_values):
                curr_values = curr_values[:, :-1].contiguous()

                # Standardize values to subtract a bias.
                if self.enable_standardization:
                    curr_values = (curr_values - self.rew_mean) / self.rew_std

                # Critic loss. Uses clipped version of the loss.
                clip_val = self.clip_val

                if clip_val > 0.0:
                    values_clipped = prev_values + (curr_values - prev_values).clamp(-clip_val, clip_val)
                    v_loss1 = (values_clipped - returns) ** 2
                else:
                    v_loss1 = torch.tensor(0.0).cuda()
                v_loss2 = (curr_values - returns) ** 2

                # Critic loss
                loss = 0.5 * masked_mean(torch.max(v_loss1, v_loss2), mask)

                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {"avg": reduced_loss}

            return output, loss_func

        return fwd_output_and_loss_func

    def offload_adam_states(self):
        if self.distributed_adam_offload_manager is None:

            self.distributed_adam_offload_manager = (
                offload_distributed_adam(self._optimizer.state_dict(state_dict_format=1, gather_on_root=False))
                if self.to_offload_adam_states and self.with_distributed_adam
                else nullcontext()
            )

            # offload onto cpu
            self.distributed_adam_offload_manager.__enter__()

    def onload_adam_states(self):
        if self.distributed_adam_offload_manager is not None:
            # load back onto GPU
            self.distributed_adam_offload_manager.__exit__(None, None, None)

        self.distributed_adam_offload_manager = None

    def infer_rm_critic(self, *args, **kwargs):
        call_order = (self._infer_rm, self._infer_critic)

        original_state = self.loaded_state_dict

        if original_state == StateDictState.CRITIC:
            # if we have the critic model already do that first
            # so we don't need to load again
            call_order = reversed(call_order)

        outputs = []
        for fn in call_order:
            output, exceeded = fn(*args, **kwargs)
            outputs.append(output)

        if original_state == StateDictState.CRITIC:
            # reverse the output back
            outputs = reversed(outputs)

        # always return rewards, value, exceeded
        return (*outputs, exceeded)

    def set_output_sequence_flag(self, value_to_set):
        if isinstance(self.model, Float16Module):
            base_module = self.model.module
        else:
            base_module = self.model

        if hasattr(base_module, "rm_head"):
            base_module.rm_head.output_sequence = value_to_set

    def _load_critic(self):
        if self.loaded_state_dict == StateDictState.REWARD:
            # no need to put the RM back to cpu, we already have it
            swap_dict(self, self.cpu_state_dict, offload_onto_cpu=False, megatron_amp_O2=self.megatron_amp_O2)

            self.set_output_sequence_flag(True)

            self.loaded_state_dict = StateDictState.CRITIC

    def _load_rm(self):
        if self.loaded_state_dict == StateDictState.CRITIC:
            self.cpu_state_dict = swap_dict(self, self.rm_state_dict, megatron_amp_O2=self.megatron_amp_O2)

            self.set_output_sequence_flag(False)
            self.loaded_state_dict = StateDictState.REWARD

    def _infer_critic(self, *args, **kwargs):
        self._load_critic()
        return self.infer(*args, **kwargs)

    def _infer_rm(self, *args, **kwargs):
        self._load_rm()
        return self.infer(*args, **kwargs)
