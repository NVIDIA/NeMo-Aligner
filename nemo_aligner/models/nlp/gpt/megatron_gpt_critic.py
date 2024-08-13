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
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
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
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp
from nemo_aligner.utils.train_utils import set_sync_funcs
from nemo_aligner.utils.utils import copy_model_states_to_cpu, masked_mean, offload_distributed_adam


class StateDictState(Enum):
    """Enum to determine which model state is loaded
    """

    CRITIC = 0
    REWARD = 1


class MegatronGPTCriticModel(MegatronGPTRewardModel, CriticModelInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        self.clip_val = self.cfg.get("loss_clip_val")

    def prepare_for_inference(self):
        super().prepare_for_inference()

    def prepare_for_training(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.global_batch_size,
            micro_batch_size=self.cfg.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def get_loss_and_metrics(self, batch, forward_only):
        sequence_length = batch["tokens"].size(-1)
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(data_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=sequence_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )
        pred_values = None
        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.as_tensor(loss_tensors_list, device=torch.cuda.current_device())
            loss_mean = loss_tensor.mean()
            pred_values = (
                torch.stack(
                    [
                        torch.as_tensor(loss_reduced["values"], device=torch.cuda.current_device())
                        for loss_reduced in losses_reduced_per_micro_batch
                    ]
                )
                .mean(0)
                .view(1, -1)
            )
            mask_amount_0 = (
                torch.as_tensor(
                    [loss_reduced["mask_amount_0"] for loss_reduced in losses_reduced_per_micro_batch],
                    device=torch.cuda.current_device(),
                )
                .sum()
                .float()
            )
        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            mask_amount_0 = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(mask_amount_0, get_last_rank())

        pred_values = broadcast_2d_tensor_within_pp(pred_values).view(-1)
        dictionary = {f"pred_value_{i}": item for i, item in enumerate(pred_values.tolist())}

        metrics = {
            "loss": loss_mean.item(),
            "mask_amount_0": mask_amount_0.item(),
            **dictionary,
        }

        return loss_mean.item(), metrics

    def get_forward_output_and_loss_func(self):
        # validation step is not used
        def fwd_output_and_loss_func(data_iterator, model):
            batch = next(data_iterator)

            required_keys = set()

            if parallel_state.is_pipeline_first_stage():
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("scores", "mask"))

            batch = {
                key: val.cuda(non_blocking=True) if key in required_keys and isinstance(val, torch.Tensor) else None
                for key, val in batch.items()
            }

            output = model(
                input_ids=batch["tokens"],
                lengths=None,
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
                labels=None,
            )

            if not parallel_state.is_pipeline_last_stage():
                output = output.to(dtype=self.autocast_dtype)

            def loss_func(curr_values):
                scores = batch["scores"]
                sequence_mask = batch["mask"]

                # mask out the prefices we don't want and scores we don't want
                mask = (scores != -100) & sequence_mask.bool().unsqueeze(-1)

                loss = torch.nn.functional.huber_loss(curr_values, scores, reduction="none", delta=self.clip_val)

                mask_denom = mask.sum((1, 2))
                if mask_denom.sum() > 0:
                    loss = (loss * mask).sum((1, 2)) / mask_denom
                    loss = loss[mask_denom].mean()
                else:
                    loss = loss.sum() * 0

                with torch.no_grad():
                    pred_values = (curr_values * mask).sum((0, 1)) / mask.sum((0, 1))
                    mask_amount_0 = (mask_denom == 0).sum()

                reduced_loss, *values = average_losses_across_data_parallel_group([loss, *pred_values])
                torch.distributed.all_reduce(mask_amount_0, group=parallel_state.get_data_parallel_group())

                return loss, {"avg": reduced_loss, "values": values, "mask_amount_0": mask_amount_0}

            return output, loss_func

        return fwd_output_and_loss_func
