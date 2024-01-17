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

from abc import ABC, abstractclassmethod
from copy import deepcopy
from typing import Callable, List, Literal, Optional, Union
from unittest.mock import patch

import torch
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint
from torch import Tensor
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules

from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


"""Megatron Core based Reward Model"""


class RewardModelHead(RowParallelLinear):
    """
    Reward model head to convert from output_size to scalar reward.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        # RM args
        output_sequence: bool = False,
        use_avg_pool: bool = False,
        dtype: torch.dtype = torch.float32,
        merge_attributes: bool = False,
        attributes_weights: Optional[List[Union[float, int]]] = None,
    ):
        config = deepcopy(config)
        config.params_dtype = dtype

        assert output_size > 0, "Output size of reward model head should be greater than zero"

        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
        )

        self.output_sequence = output_sequence
        self.use_avg_pool = use_avg_pool
        self.dtype = dtype
        self.merge_attributes = merge_attributes

        if attributes_weights is None:
            self.attributes_weights = torch.full((self.output_size,), 1.0 / self.output_size)
        else:
            self.attributes_weights = torch.tensor(attributes_weights, dtype=torch.float)

        assert self.attributes_weights.size(0) == self.output_size

    def _compute_attributes(self, hidden_states, lengths):
        """
        for critic, return a tensor with shape [B x S x self.output_size]
        for reward, return a tensor with shape [B x self.output_size]
        """

        # we sometimes want to run our RM head in FP32, this allows it
        autocast_context = torch.autocast(device_type=hidden_states.device.type, dtype=self.dtype)

        # hidden_size is S x B x D
        if self.output_sequence:
            with autocast_context:
                output = super().forward(hidden_states.to(self.weight.dtype))[0]  # [S x B x self.output_size]

            # Making it contiguous is required at least by `torch.distributed.gather()`.
            return output.permute(1, 0, 2).contiguous()  # [B x S x self.output_size]

        if self.use_avg_pool:
            # lengths is shape B and arange is shape S, broadcast it to S x B
            # S x 1 op with B -> mask for S x B
            mask = torch.arange(hidden_states.size(0), device=lengths.device).unsqueeze(-1) < lengths

            # S x B x D * S x B x 1
            last_state = (hidden_states * mask.unsqueeze(-1)).sum(0)

            # divide by mean post hoc
            # sum implicitly casts back to fp32, but the autocast will handle it below if needed
            last_state = last_state / lengths.unsqueeze(-1)
        else:
            last_state = hidden_states[lengths - 1, torch.arange(lengths.shape[0], device=hidden_states.device), :]

        # B x D -> 1 x B x D b/c RowParallel wants S x B x D
        last_state = last_state.unsqueeze(0)

        # squeeze out the S term on dim 0, we always add bias
        with autocast_context:
            output = super().forward(last_state.to(self.weight.dtype))[0].squeeze(0)

        return output

    def forward(self, hidden_states, lengths):
        attributes = self._compute_attributes(
            hidden_states, lengths
        )  # [B x S x self.output_size] or [B x self.output_size]

        self.attributes_weights = self.attributes_weights.to(attributes.device)
        if self.output_sequence:
            # a sequence of multi-attribute rewards, used for critic model, returning tensor with shape [B x S]
            assert attributes.dim() == 3, "for critic, attributes should have shape [B x S x self.output_size]"
            return attributes @ self.attributes_weights

        else:
            assert attributes.dim() == 2, "for reward, attributes should have shape [B x self.output_size]"
            if not self.merge_attributes:
                # do not merge attributes during regression rm training
                return attributes
            else:
                # during ppo, returning tensor with shape [B x 1]
                return (attributes @ self.attributes_weights).unsqueeze(-1)


class ValueHead(TransformerBlock):

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
            config = config,
            spec = spec,
            post_layer_norm = post_layer_norm,
            pre_process = pre_process,
            post_process = post_process,
        )
 

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number,)

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1 + self.layer_number_offset)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # # TODO: add back standalone_embedding_stage
        # if self.num_layers == 0:
        #     # When a standalone embedding stage is used (e.g.,
        #     # args.standalone_embedding_stage == True), virtual pipeline ranks
        #     # on pipeline rank 0 will have zero transformer layers assigned to
        #     # them. This results in the model's input and output tensors to be
        #     # the same, which will cause failure for certain output tensor
        #     # optimizations (e.g., pipeline output deallocation). To remedy
        #     # this, we assign a 'no-op' layer on these ranks, which will
        #     # disconnect the input tensor from the output tensor.
        #     self.num_layers = 1
        #     self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        # else:
        #     self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def layer_sharded_state_dict(self, layer, prefix=''):
        offset = 0
        num_layers = layer.config.num_layers

        global_layer_offset = layer.layer_number - 1  # layer.layer_number starts at 1
        state_dict_prefix = (
            f'{prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock
        )
        offset = layer.layer_number - self.layer_number_offset - 1 
        sharded_pp_offset = [
            (0, offset, num_layers)
        ]  # PP sharding offset for ShardedTensors

        attn_state_dict = layer.self_attention.sharded_state_dict(
            prefix=f'{state_dict_prefix}self_attention.',
            sharded_key_prefix=f'{prefix}self_attention.',
            sharded_offsets=sharded_pp_offset,
        )

        mlp_state_dict = layer.mlp.sharded_state_dict(
            prefix=f'{state_dict_prefix}mlp.',
            sharded_key_prefix=f'{prefix}mlp.',
            sharded_offsets=sharded_pp_offset,
        )

        sharded_state_dict = {**mlp_state_dict, **attn_state_dict}

        return sharded_state_dict


    def sharded_state_dict(self, prefix: str = ''):

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            sharded_state_dict.update(self.layer_sharded_state_dict(layer, layer_prefix))

        if self.post_process and self.post_layer_norm:
            state_dict = self.state_dict(keep_vars=True)

            tensor = state_dict['final_layernorm.weight']
            layer_name = f'{prefix}final_layernorm.weight'
            sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)

            # RMSNorm doesn't have bias.
            if 'final_layernorm.bias' in state_dict.keys():
                tensor = state_dict['final_layernorm.bias']
                layer_name = f'{prefix}final_layernorm.bias'
                sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(
                    tensor, layer_name
                )

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

        # if last stage or no PP
        if post_process:
            self.value_head = ValueHead(
            config=head_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            layer_number_offset=len(self.decoder.submodules.layer_specs))

            # self.rm_head = RewardModelHead(
            #     self.config.hidden_size,
            #     num_attributes,
            #     config=config,
            #     init_method=self.config.init_method,
            #     output_sequence=output_sequence,
            #     use_avg_pool=use_avg_pool,
            #     dtype=self.dtype if head_dtype is None else head_dtype,
            #     merge_attributes=merge_attributes,
            #     attributes_weights=attribute_weights,
            # )

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ):
        # TODO(geshen): hack to get the hidden states
        # and for mcore to not call the output layer
        with patch.object(self, "post_process", False):
            hidden_states = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_params=inference_params,
            )

        if self.post_process:
            return self.value_head(hidden_states, attention_mask=attention_mask, )

        return hidden_states

    def sharded_state_dict(self, prefix=""):
        # need to turn post process off to not load the output layer
        # from the parent
        sharded_state_dict = super().sharded_state_dict(prefix=prefix)

        if self.post_process and self.return_value_head_in_state_dict:
            value_head_prefix = f"{prefix}value_head."
            value_head_state_dict = self.value_head.sharded_state_dict(prefix=value_head_prefix)
            sharded_state_dict.update(value_head_state_dict)

        return sharded_state_dict
