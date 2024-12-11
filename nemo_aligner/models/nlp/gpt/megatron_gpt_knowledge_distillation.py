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

from typing import List, Optional, Tuple, Union

import hydra
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import get_micro_batch_size, get_num_microbatches
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils.distributed import _TopKLogitsCrossEntropy
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_sync_funcs,
)


class GPTKnowledgeDistillationModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        self.target_logits_scale = self.cfg.knowledge_distillation.get("target_logits_scale", 1.0)
        self.logits_scale = self.cfg.knowledge_distillation.get("logits_scale", 1.0)

        self.kd_loss = self.cfg.knowledge_distillation.get("kd_loss", "fwd_kl")
        self.kd_loss_weight = self.cfg.knowledge_distillation.get("kd_loss_weight", 1)
        self.sft_loss_weight = self.cfg.knowledge_distillation.get("sft_loss_weight", 0)
        self.cross_tokenizer = self.cfg.knowledge_distillation.get("cross_tokenizer", False)
        assert (
            self.kd_loss_weight != 0 or self.sft_loss_weight != 0
        ), "sft loss weight and knowledge distillation loss weight cannot both be 0"

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # there is a problem with apex ignoring the mask on the older models
                # so we will always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "loss_mask", "topk_logits", "topk_token_ids"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # this is necessary if MBS > 1 with the new GBS padding logic, as you may get batch dim > 1 in some configs
            # these two lines ensure your position_ids and attn_mask are always B=1
            # position_ids = batch["position_ids"][0:1]
            attention_mask = batch["attention_mask"][0:1]

            tokens = batch["tokens"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]
            if loss_mask is not None:
                loss_mask = loss_mask.clamp(min=0, max=1)
            target_topk_logits = batch["topk_logits"]
            target_topk_token_ids = batch["topk_token_ids"]
            # Model forward pass
            forward_args = {
                "input_ids": tokens,
                "position_ids": batch["position_ids"],
                "attention_mask": attention_mask,
                "labels": None,
                "loss_mask": None,
            }

            # TODO: we can remove this someday when we no longer support legacy models
            if not self.mcore_gpt:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop("loss_mask")
            else:
                forward_args.pop("loss_mask")

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def loss_func(predicted_logits):

                loss, kd_loss, sft_loss = _TopKLogitsCrossEntropy.apply(
                    self.logits_scale * predicted_logits,
                    self.target_logits_scale * target_topk_logits,
                    target_topk_token_ids,
                    labels,
                    self.kd_loss_weight,
                    self.sft_loss_weight,
                    self.kd_loss == "bwd_kl",
                    self.cross_tokenizer,
                )

                ## reduce losses
                loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask).clamp(min=1.0)
                kd_loss = torch.sum(kd_loss * loss_mask) / torch.sum(loss_mask).clamp(min=1.0)
                sft_loss = torch.sum(sft_loss * loss_mask) / torch.sum(loss_mask).clamp(min=1.0)

                reduced_loss, reduced_kd_loss, reduced_sft_loss = average_losses_across_data_parallel_group(
                    [loss, kd_loss, sft_loss]
                )

                return (
                    loss,
                    {"avg": reduced_loss, "avg_sft_loss": reduced_sft_loss, "avg_kd_loss": reduced_kd_loss,},
                )

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only):
        """Take a data_iter which is an interator over the microbatches
            and return loss as well as metrics
        """
        _, seq_length = batch["tokens"].shape
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()
        fwd_loss_fn = self.get_forward_output_and_loss_func(forward_only)

        losses_reduced = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            micro_batch_size=get_micro_batch_size(),
            seq_length=seq_length,
        )

        torch.cuda.synchronize()

        # only the last stages of the pipeline return losses
        if parallel_state.is_pipeline_last_stage():
            # average loss across micro batches
            loss_mean = torch.as_tensor(
                [loss_reduced["avg"] for loss_reduced in losses_reduced], device=torch.cuda.current_device(),
            ).mean()
            sft_loss_mean = torch.as_tensor(
                [loss_reduced["avg_sft_loss"] for loss_reduced in losses_reduced], device=torch.cuda.current_device(),
            ).mean()
            kd_loss_mean = torch.as_tensor(
                [loss_reduced["avg_kd_loss"] for loss_reduced in losses_reduced], device=torch.cuda.current_device(),
            ).mean()
        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            sft_loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            kd_loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
        # Logging
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(sft_loss_mean, get_last_rank())
        torch.distributed.broadcast(kd_loss_mean, get_last_rank())
        loss_value = loss_mean.item()
        metrics = {
            "loss": loss_value,
            "sft_loss": sft_loss_mean.item(),
            "kd_loss": kd_loss_mean.item(),
        }
        return loss_value, metrics

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)
