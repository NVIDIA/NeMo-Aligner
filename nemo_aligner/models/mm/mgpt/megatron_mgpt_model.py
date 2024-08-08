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
from megatron.core import parallel_state
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel, MCoreNevaModel
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import get_specs
from nemo.utils import logging
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo_aligner.utils.text_generation_utils import MGPTModelTextGenerationStrategy

class MultimodalGPTModel(MegatronNevaModel):
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

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        media_start_id = self.tokenizer.token_to_id(self.cfg.mm_cfg.get("im_start_token", "<extra_id_4>"))
        media_end_id = self.tokenizer.token_to_id(self.cfg.mm_cfg.get("im_end_token", "<extra_id_5>"))

        if self.mcore_gpt:
            if not parallel_state.is_initialized():

                def dummy():
                    return

                if self.trainer.strategy.launcher is not None:
                    self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
                self.trainer.strategy.setup_environment()

            model = MCoreNevaModel(
                mm_cfg=self.cfg.mm_cfg,
                media_start_id=media_start_id,
                media_end_id=media_end_id,
                mcore_gpt=self.mcore_gpt,
                config=self.transformer_config,
                transformer_layer_spec=get_specs(self.spec_name),
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                max_sequence_length=self.cfg.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
            )
        else:
            raise NotImplementedError("Only MCoreGPT models are supported! Please set mcore_gpt=True.")
            

        logging.info(
            f"Neva model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        return model
    
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



