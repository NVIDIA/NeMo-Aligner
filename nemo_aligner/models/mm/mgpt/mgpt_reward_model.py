# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy
from typing import Callable, List, Literal, Optional, Union
from unittest.mock import patch
from omegaconf.dictconfig import DictConfig
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MCoreNevaModel
from nemo_aligner.models.nlp.gpt.gpt_reward_model import RewardModelHead
"""Megatron Core based Multimodal Reward Model"""

class MultimodalGPTRewardModel(MCoreNevaModel):
    """MCoreGPT-based reward model."""

    return_rm_head_in_state_dict = True

    def __init__(
        self,
        mm_cfg: DictConfig,
        media_start_id: int,
        media_end_id: int,
        media_token_id: int,
        mcore_gpt: bool,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        output_sequence: bool = False,
        use_avg_pool: bool = False,
        head_dtype: torch.dtype = None,
        num_attributes: int = 1,
        attribute_weights: Optional[List[Union[float, int]]] = None,
        merge_attributes: bool = False,
    ):
        super().__init__(
            mm_cfg=mm_cfg,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
            media_token_id=media_token_id,
            mcore_gpt=mcore_gpt,
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )

        # if last stage or no PP
        if post_process:
            self.rm_head = RewardModelHead(
                self.config.hidden_size,
                num_attributes,
                config=config,
                init_method=self.config.init_method,
                output_sequence=output_sequence,
                use_avg_pool=use_avg_pool,
                dtype=self.dtype if head_dtype is None else head_dtype,
                merge_attributes=merge_attributes,
                attributes_weights=attribute_weights,
            )

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        media: Tensor = None,
        inference_params=None,
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            #print("Setting media", media)
            self.embedding.word_embeddings.set_media(media)
        # TODO(geshen): hack to get the hidden states
        # and for mcore to not call the output layer
        with patch.object(self, "post_process", False):
            hidden_states = MCoreGPTModel.forward(
                self,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_params=inference_params,
            )

        if self.post_process:
            return self.rm_head(hidden_states, lengths)

        return hidden_states

    def sharded_state_dict(self, prefix=""):
        # need to turn post process off to not load the output layer
        # from the parent
        sharded_state_dict = super().sharded_state_dict(prefix=prefix)

        if not self.return_rm_head_in_state_dict:
            sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "rm_head" not in k}

        return sharded_state_dict
