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

from typing import List, Literal, Optional, Union
from unittest.mock import patch

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint
from torch import Tensor

from nemo_aligner.models.nlp.gpt.gpt_reward_model import RewardModelHead

"""Megatron Core based Reward Model"""


def init_method_constant(constant):
    """Init method to zero"""

    def init_(tensor):
        return torch.nn.init.constant_(tensor, constant)

    return init_


class ValueHead(TransformerBlock):
    """Value head is a transformer block.
       Override the build_layers method so we can change the layer number offset to
       start from the last layer of the decoder.
       This is needed because the kv-cache dict in the inference parameters uses 
       layer number as key. The rename can make sure there is no key clash and we 
       can reuse the same inference parameters for both the decoder and the value head.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        layer_number_offset: int = 0,
    ):
        self.layer_number_offset = layer_number_offset
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number,)

        # offset is implicit in TransformerLayer
        # add self.layer_number_offset to prevent key clash in inference parameters
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1 + self.layer_number_offset)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = TENorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon,
            )

    def layer_sharded_state_dict(self, layer, prefix=""):
        """need to override this method to change the layer number offset which is based on the layer number.
        Otherwise the sharded_pp_offset will be wrong.
        The right behavior of shard_pp_offset should be (0, layer_number, num_layers), where layer number is the 
        layer number in the value head, and num_layers are the total number of layers in the value head.
        """
        offset = 0
        num_layers = layer.config.num_layers

        global_layer_offset = layer.layer_number - 1  # layer.layer_number starts at 1
        state_dict_prefix = f"{prefix}{global_layer_offset - offset}."  # module list index in TransformerBlock
        offset = layer.layer_number - self.layer_number_offset - 1
        sharded_pp_offset = [(0, offset, num_layers)]  # PP sharding offset for ShardedTensors

        attn_state_dict = layer.self_attention.sharded_state_dict(
            prefix=f"{state_dict_prefix}self_attention.",
            sharded_key_prefix=f"{prefix}self_attention.",
            sharded_offsets=sharded_pp_offset,
        )

        mlp_state_dict = layer.mlp.sharded_state_dict(
            prefix=f"{state_dict_prefix}mlp.", sharded_key_prefix=f"{prefix}mlp.", sharded_offsets=sharded_pp_offset,
        )

        sharded_state_dict = {**mlp_state_dict, **attn_state_dict}

        return sharded_state_dict

    def sharded_state_dict(self, prefix: str = ""):

        sharded_state_dict = {}

        layer_prefix = f"{prefix}layers."
        for layer in self.layers:
            sharded_state_dict.update(self.layer_sharded_state_dict(layer, layer_prefix))

        if self.post_process and self.post_layer_norm:
            state_dict = self.state_dict(keep_vars=True)

            tensor = state_dict["final_layernorm.weight"]
            layer_name = f"{prefix}final_layernorm.weight"
            sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)

            # RMSNorm doesn't have bias.
            if "final_layernorm.bias" in state_dict.keys():
                tensor = state_dict["final_layernorm.bias"]
                layer_name = f"{prefix}final_layernorm.bias"
                sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)

        return sharded_state_dict


class GPTHybridModel(GPTModel):
    """MCoreGPT-based multiple heads GPT + Critic model."""

    return_value_head_in_state_dict = True

    def __init__(
        self,
        config: TransformerConfig,
        head_config: TransformerConfig,
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
        seq_len_interpolation_factor: Optional[float] = None,
        output_sequence: bool = False,
        use_avg_pool: bool = False,
        head_dtype: torch.dtype = None,
        num_attributes: int = 1,
        attribute_weights: Optional[List[Union[float, int]]] = None,
        merge_attributes: bool = False,
    ):
        super().__init__(
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
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )
        self.head_config = head_config

        # if last stage or no PP
        if post_process:
            self.value_head = ValueHead(
                config=head_config,
                spec=transformer_layer_spec,
                pre_process=True,
                post_process=True,
                layer_number_offset=len(self.decoder.submodules.layer_specs),
            )
            assert output_sequence is True, "output_sequence must be True for hybrid model"
            assert num_attributes == 1, "num_attributes must be 1 for hybrid model"

            self.rm_head = RewardModelHead(
                self.config.hidden_size,
                num_attributes,
                config=config,
                init_method=init_method_constant(0.0),
                output_sequence=output_sequence,
                use_avg_pool=use_avg_pool,
                dtype=self.dtype if head_dtype is None else head_dtype,
                merge_attributes=merge_attributes,
                attributes_weights=attribute_weights,
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ):
        # TODO(geshen): hack to get the hidden states
        # and for mcore to not call the output layer
        with patch.object(self, "post_process", False):
            hidden_states_raw = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_params=inference_params,
            )

        value = None
        if self.post_process:
            # logits and loss
            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()
            # added all the post process stuff here
            logits, _ = self.output_layer(hidden_states_raw, weight=output_weight)

            # Rotary positional embeddings (embedding is None for PP intermediate devices)
            rotary_pos_emb = None
            if self.position_embedding_type == "rope":
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.value_head, decoder_input, self.head_config
                )
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

            hidden_states_raw = self.value_head(
                hidden_states_raw,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
            )
            value = self.rm_head(hidden_states_raw, None)
            if labels is None:
                output = logits.transpose(0, 1).contiguous()
                return output, value
            output = self.compute_language_model_loss(labels, logits)
        else:
            output = hidden_states_raw
        return output, value

    def sharded_state_dict(self, prefix=""):
        # need to turn post process off to not load the output layer
        # from the parent
        sharded_state_dict = super().sharded_state_dict(prefix=prefix)

        if self.post_process and self.return_value_head_in_state_dict:
            value_head_prefix = f"{prefix}value_head."
            value_head_state_dict = self.value_head.sharded_state_dict(prefix=value_head_prefix)
            sharded_state_dict.update(value_head_state_dict)

            rm_head_prefix = f"{prefix}rm_head."
            rm_head_state_dict = self.rm_head.state_dict(prefix=rm_head_prefix, keep_vars=True)

            # weights are sharded row wise
            weight_key = f"{rm_head_prefix}weight"

            sharded_state_dict[weight_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor=rm_head_state_dict[weight_key],
                key=weight_key,
                replica_id=parallel_state.get_data_parallel_rank(),
                allow_shape_mismatch=False,
                tp_axis=1,
            )

            # biases are not sharded
            bias_key = f"{rm_head_prefix}bias"
            sharded_state_dict[bias_key] = make_sharded_tensor_for_checkpoint(rm_head_state_dict[bias_key], bias_key)

        return sharded_state_dict
