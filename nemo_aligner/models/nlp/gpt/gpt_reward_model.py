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


class GPTRewardModel(GPTModel):
    """MCoreGPT-based reward model."""

    return_rm_head_in_state_dict = True

    def __init__(
        self,
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
            return self.rm_head(hidden_states, lengths)

        return hidden_states

    def sharded_state_dict(self, prefix=""):
        # need to turn post process off to not load the output layer
        # from the parent
        sharded_state_dict = super().sharded_state_dict(prefix=prefix)

        if self.post_process and self.return_rm_head_in_state_dict:
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
