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
from megatron.core.num_microbatches_calculator import get_micro_batch_size, get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from lightning.pytorch.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils import parallel_state
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
from nemo_aligner.utils.utils import configure_batch_sizes


class GPTSFTModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        inference_params = dict(cfg.get("inference", {}))
        # note that this will fail if import path is not available when the model is restored
        # this is by design as it might not be possible to use model correctly without a matching
        # inference strategy
        if "strategy" in inference_params:
            if inference_params["strategy"] is not None:
                inference_params["strategy"] = hydra.utils.instantiate(inference_params["strategy"], model=self)
        self.set_inference_params(**inference_params)

    def set_inference_params(self, length_params=None, sampling_params=None, strategy=None):
        # TODO (igitman): the name self._inference_params is very similar to self.inference_params
        #    that's used by the base model for another purpose. There is also self._inference_config
        #    that has a similar role to the parameters below but is less convenient.
        #    While there is a danger for accidental name collision and this adds confusion, it's ok for now
        #    as we are planning to remove dependence on the MegatronGPTModel after which we can remove this note

        # registering inference parameters or default values
        self._inference_params = {
            "length_params": length_params or get_default_length_params(),
            "sampling_params": sampling_params or get_default_sampling_params(),
            "strategy": strategy,
        }

    def get_inference_params(self):
        return self._inference_params

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

    def generate(
        self,
        inputs: Union[List[str], Tuple[torch.Tensor, torch.Tensor]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        *,
        strategy: Optional[TextGenerationStrategy] = None,
    ) -> OutputType:
        """
        Same as base model generate, except the following:

        1. Apply padding to max length.
        2. Add a "predictions" key to the output, which is the model output without the prompt.

        These two additional steps above are only performed for actual generation from the model:
        if `generate()` is called with `compute_logprob=True` then the base model method is used.
        """
        if sampling_params is not None and sampling_params.get("compute_logprob", False):
            return super().generate(
                inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy
            )

        if not isinstance(inputs, (list, tuple)):
            raise NotImplementedError(f"Expected type(inputs)=(list or tuple) but got {type(inputs)=}")

        if isinstance(inputs[0], str):
            # add_EOS=False since it is absent from nemo.collections.nlp.modules.common.text_generation_utils.megatron_gpt_generate
            prompt_tokens, prompt_lengths = tokenize_batch(
                sentences=inputs,
                tokenizer=self.tokenizer,
                max_len=self.cfg.encoder_seq_length,
                add_BOS=sampling_params["add_BOS"],
                add_EOS=False,
            )
        else:
            prompt_tokens, prompt_lengths = inputs

        max_prompt_length = prompt_lengths.max().item()
        max_response_length = length_params["max_length"]
        max_length = max_prompt_length + max_response_length
        # # nemo requires us to pad the response length up before we do anything
        prompt_tokens = torch.nn.functional.pad(prompt_tokens, (0, max_length), value=self.tokenizer.eos_id)
        output = super().generate(
            inputs=(prompt_tokens, prompt_lengths),
            length_params=length_params,
            sampling_params=sampling_params,
            strategy=strategy,
        )
        if output is not None:  # may be `None` for intermediate PP ranks when PP>2
            # adding predictions key which contains only model predictions without the prompt
            output["predictions"] = [
                self.tokenizer.ids_to_text(tokens[length.item() :][:max_response_length])
                for tokens, length in zip(output["token_ids"], prompt_lengths)
            ]
        return output

    @torch.no_grad()
    def infer(self, inference_batch, length_params=None, sampling_params=None, strategy=None):
        prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        default_params = self.get_inference_params()

        self.prepare_for_inference()
        outputs = self.generate(
            (prompt_tokens, prompt_lengths),
            length_params=length_params or default_params["length_params"],
            sampling_params=sampling_params or default_params["sampling_params"],
            strategy=strategy or default_params["strategy"],
        )
        self.finish_inference()

        return outputs

    def prepare_for_inference(self):
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)

    def finish_inference(self):
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()
        set_train(self)
