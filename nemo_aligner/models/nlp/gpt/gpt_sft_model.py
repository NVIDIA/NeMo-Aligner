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
from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_sync_funcs,
)
from nemo_aligner.utils.utils import configure_batch_sizes


class GPTSFTModel(MegatronGPTModel, SupervisedInterface):
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
            loss_mean = torch.concat([loss_reduced["avg"] for loss_reduced in losses_reduced]).mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()
        # Logging
        torch.distributed.broadcast(loss_mean, get_last_rank())
        loss_value = loss_mean.detach().item()
        metrics = {"loss": loss_value}
        return loss_value, metrics

    def prepare_for_training_step(self):
        """things to call to preprare for training
        """
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        """things to call to finish training for example grad reductions
        """
        grad_reductions(self)

    def prepare_for_validation_step(self):
        """things to call to prepare for validation
        """
        prepare_for_validation_step(self)
        gbs = int(self.cfg.data.validation_ds.global_batch_size)
        mbs = int(self.cfg.data.validation_ds.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

    def finish_validation_step(self):
        """things to call to prepare for validation
        """
        finish_validation_step(self)
        # restore the batch sizes for training
        gbs = int(self.cfg.data.train_ds.global_batch_size)
        mbs = int(self.cfg.data.train_ds.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)
