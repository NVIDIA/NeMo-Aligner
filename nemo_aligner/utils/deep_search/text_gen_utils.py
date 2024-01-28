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

"""Utilities for generating text."""

import os
import pickle
import re
from collections.abc import Iterable
from functools import partial
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.utils import AppState
from nemo_aligner.utils.deep_search.communication_util import receive_generate_info, send_generate_info, get_model_parallel_src_rank
import torch.distributed as dist
from nemo_aligner.utils.distributed import broadcast_2d_tensor, gather_tensor

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = [
    "megatron_gpt_generate",
    "megatron_neva_generate",
    "generate",
]


def dp_search(
    model,
    inputs=None,
    action=None,
    context_ids=None,
    session_info=None,
    tokens_to_generate=1,  # max search depth
    top_k=0,
    end_strings=["<|endoftext|>"],
    **strategy_args,
) -> OutputType:
    """
    similar to the search function, but the assumption is that dp_search is called 
    by rank0 inside each data parallel group. no need of gather and broadcast. 
    """

    if "strategy" in strategy_args:
        inference_strategy = strategy_args["strategy"]
    else:
        raise ValueError("strategy is not specified")
    # init the objects for inference
    init = inputs is not None
    if inputs is None:
        # not the first node
        assert action is not None
        # action is a tensor of shape [batch_size, 1], type int32
        # it is the selected actions during tree search from node at depth, type int32
        # depth is a tensor of shape [batch_size, 1]
        # from the action and depth values, we can retrieve the node
        action = torch.cuda.IntTensor(action)
        batch_size = action.shape[0]
        context_tokens_tensor = torch.cuda.LongTensor(batch_size, 1)
        context_length_tensor = torch.cuda.LongTensor(batch_size)
    else:
        # first time to run inference,
        # need to initialize the root node, kv-cache
        assert action is None
        if isinstance(inputs, tuple): # tuple of (context_tokens_tensor, context_length_tensor)
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                inputs, 0, False
            )
        batch_size = context_tokens_tensor.size(0)
        action = torch.cuda.IntTensor(batch_size, 1)
        action[:] = 0

    cache_hit_indicator = []
    if not init:
        # check if the data hits the cache
        cache_infer_results = []
        for a, context_id in zip(action, context_ids):
            infer = inference_strategy.search_db.get_infer_cache(session_info, context_id, a.item())
            if infer is not None:
                cache_hit_indicator.append(True)
                cache_infer_results.append(infer)
            else:
                cache_hit_indicator.append(False)
        cache_hit_indicator = torch.cuda.BoolTensor(cache_hit_indicator)
        context_tokens_tensor = context_tokens_tensor[~cache_hit_indicator]
        context_length_tensor = context_length_tensor[~cache_hit_indicator]
        action = action[~cache_hit_indicator]
        context_ids = [context_ids[i] for i in range(len(context_ids)) if not cache_hit_indicator[i]]

    if len(context_tokens_tensor) > 0:
        true_context_length = None
        if init:
            inference_strategy.init(context_tokens_tensor, tokens_to_generate, session_info)
        else:
            context_tokens_tensor, context_length_tensor, true_context_length = inference_strategy.compute_inference_params(session_info, context_ids, action)

        output_actions, output_policys, output_values = sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            context_ids=context_ids,
            session_info=session_info,
            top_k=top_k,
            init=init,
            true_context_length=true_context_length,
        )

    if not init:
        if sum(cache_hit_indicator) > 0:
            actions = []
            policys = []
            values = []
            count = 0
            for cache_hit in cache_hit_indicator:
                if cache_hit:
                    infer = cache_infer_results.pop(0)
                    actions.append(torch.cuda.IntTensor(infer['action']))
                    policys.append(torch.cuda.FloatTensor(infer['policy']))
                    values.append(torch.cuda.FloatTensor(infer['value']))
                else:
                    # output_actions are tensors of shape [batch_size, top_k]
                    actions.append(output_actions[count])
                    policys.append(output_policys[count])
                    values.append(output_values[count])
                    count += 1
            output_actions = torch.vstack(actions)
            output_policys = torch.vstack(policys)
            output_values = torch.hstack(values)

    output = {}
    output['action'] = output_actions
    output['policy'] = output_policys
    output['value'] = output_values
    output = inference_strategy.post_generation_process(output)
    return output


def search(
    model,
    inputs=None,
    action=None,
    context_ids=None,
    session_info=None,
    tokens_to_generate=1,  # max search depth
    top_k=0,
    end_strings=["<|endoftext|>"],
    **strategy_args,
) -> OutputType:
    """
    """
    if "strategy" in strategy_args:
        inference_strategy = strategy_args["strategy"]
    else:
        raise ValueError("strategy is not specified")
    # init the objects for inference
    init = inputs is not None
    if torch.distributed.get_rank() == 0:
        if inputs is None:
            # not the first node
            assert action is not None
            # action is a tensor of shape [batch_size, 1], type int32
            # it is the selected actions during tree search from node at depth, type int32
            # depth is a tensor of shape [batch_size, 1]
            # from the action and depth values, we can retrieve the node
            action = torch.cuda.IntTensor(action)
            batch_size = action.shape[0]
            context_tokens_tensor = torch.cuda.LongTensor(batch_size, 1)
            context_length_tensor = torch.cuda.LongTensor(batch_size)

        else:
            # first time to run inference,
            # need to initialize the root node, kv-cache
            assert action is None
            if isinstance(inputs, tuple): # tuple of (context_tokens_tensor, context_length_tensor)
                context_tokens_tensor, context_length_tensor = inputs
            else:
                context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                    inputs, 0, False
                )
            batch_size = context_tokens_tensor.size(0)
            action = torch.cuda.IntTensor(batch_size, 1)
            action[:] = 0

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            action,
            tokens_to_generate,
            top_k,
            end_strings,
            context_ids,
            session_info,
        )
    else:
        (
            context_tokens_tensor,
            context_length_tensor,
            action,
            tokens_to_generate,
            top_k,
            end_strings,
            context_ids,
            session_info,
        ) = receive_generate_info()

    # distributed batch to data parallel groups
    # Select subset of data needed for this rank.
    dp_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    context_length_tensor = context_length_tensor.chunk(dp_size)[dp_rank]
    context_tokens_tensor = context_tokens_tensor.chunk(dp_size)[dp_rank]
    action = action.chunk(dp_size)[dp_rank]
    # chunk the context_ids, which is a list of strings
    batch_size = len(context_ids)
    assert batch_size % dp_size == 0
    chuck_size = batch_size // dp_size
    context_ids = context_ids[dp_rank * chuck_size : (dp_rank + 1) * chuck_size]

    cache_hit_indicator = []
    if not init:
        # check if the data hits the cache
        cache_infer_results = []
        for a, context_id in zip(action, context_ids):
            infer = inference_strategy.search_db.get_infer_cache(session_info, context_id, a.item())
            if infer is not None:
                print('cache hit', len(context_id), a.item())
                cache_hit_indicator.append(True)
                cache_infer_results.append(infer)
            else:
                cache_hit_indicator.append(False)
        cache_hit_indicator = torch.cuda.BoolTensor(cache_hit_indicator)
        context_tokens_tensor = context_tokens_tensor[~cache_hit_indicator]
        context_length_tensor = context_length_tensor[~cache_hit_indicator]
        action = action[~cache_hit_indicator]
        context_ids = [context_ids[i] for i in range(len(context_ids)) if not cache_hit_indicator[i]]

    if len(context_tokens_tensor) > 0:
        true_context_length = None
        if init:
            inference_strategy.init(context_tokens_tensor, tokens_to_generate, session_info)
        else:
            context_tokens_tensor, context_length_tensor, true_context_length = inference_strategy.compute_inference_params(session_info, context_ids, action)

        output_actions, output_policys, output_values = sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            context_ids=context_ids,
            session_info=session_info,
            top_k=top_k,
            init=init,
            true_context_length=true_context_length,
        )
    
    if not init:
        if sum(cache_hit_indicator) > 0:
            actions = []
            policys = []
            values = []
            count = 0
            for cache_hit in cache_hit_indicator:
                if cache_hit:
                    infer = cache_infer_results.pop(0)
                    actions.append(torch.cuda.IntTensor(infer['action']))
                    policys.append(torch.cuda.FloatTensor(infer['policy']))
                    values.append(torch.cuda.FloatTensor(infer['value']))
                else:
                    # output_actions are tensors of shape [batch_size, top_k]
                    actions.append(output_actions[count])
                    policys.append(output_policys[count])
                    values.append(output_values[count])
                    count += 1
            output_actions = torch.vstack(actions)
            output_policys = torch.vstack(policys)
            output_values = torch.hstack(values)
        # combine cache and non-cache results

    result_output_actions_list = gather_tensor(
        output_actions, dst=parallel_state.get_data_parallel_src_rank(), group=parallel_state.get_data_parallel_group(), dtype=output_actions.dtype
    )

    result_output_policys_list = gather_tensor(
        output_policys, dst=parallel_state.get_data_parallel_src_rank(), group=parallel_state.get_data_parallel_group(), dtype=output_policys.dtype
    )

    result_output_values_list = gather_tensor(
        output_values, dst=parallel_state.get_data_parallel_src_rank(), group=parallel_state.get_data_parallel_group(), dtype=output_values.dtype
    )

    output = {}
    if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
        # concate the output_actions and output_policys
        output["action"] = torch.cat(result_output_actions_list, dim=0)
        output["policy"] = torch.cat(result_output_policys_list, dim=0)
        output["value"] = torch.cat(result_output_values_list, dim=0)
        # gather the output from data parallel workers
        output = inference_strategy.post_generation_process(output)
    else:
        pass
    return output


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    context_ids,
    session_info,
    top_k,
    init,
    true_context_length=None,):
    # Importing here to avoid circular import errors

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert (
        model.cfg.get("sequence_parallel", False) == False
    ), "sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"
    assert (
        model.cfg.get("activations_checkpoint_granularity", None) is None
    ), "activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"
    assert (
        model.cfg.get("activations_checkpoint_method", None) is None
    ), "activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        # get min context length
        context_length = context_lengths.min().item()

        counter = 0

        batch_size = context_tokens.size(0)

        tokens = context_tokens

        maxlen = 1 + context_lengths.max().item()

        maxlen = inference_strategy.clip_max_len(maxlen)


        output_actions = torch.cuda.IntTensor(batch_size, top_k)
        output_policy = torch.cuda.FloatTensor(batch_size, top_k)
        output_values = torch.cuda.FloatTensor(batch_size)

        while context_length < maxlen:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, micro_batch_size, context_length, init, session_info, counter, true_context_length
            )

            batch_update_indicator = context_lengths == context_length

            if (not init) and (not batch_update_indicator.any()):
                # if there is nothing to update, skip the computation
                # only works for depths > 0 nodes
                context_length += 1
                counter += 1
                continue

            output = inference_strategy.forward_step(batch, tensor_shape, session_info)
            if parallel_state.is_pipeline_last_stage():
                # get last rank
                logits = output[0]["logits"][:, -1].contiguous() # output[0]["logits"] shape[batch_size, length, partial vocab_size]
                logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
                assert logits is not None
                logits = logits.view(batch_size, -1)

                # make sure it won't sample outside the vocab_size range
                logits[:, tokenizer.vocab_size :] = -float("Inf")

                updated_logits, actions = torch.topk(logits, top_k)
                probs = F.softmax(updated_logits, dim=-1)

                output_actions[batch_update_indicator] = actions[batch_update_indicator].type(torch.int32)
                output_policy[batch_update_indicator] = probs[batch_update_indicator]
                if 'value' in output[0]:
                    value = output[0]['value'][:, -1]
                    output_values[batch_update_indicator] = value[batch_update_indicator]
 
            context_length += 1
            counter += 1
        # sync from last pipeline stage to src rank, so that it can be returned
        if parallel_state.is_pipeline_last_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_pipeline_model_parallel_group()
            torch.distributed.broadcast(output_actions, src, group)
            torch.distributed.broadcast(output_policy, src, group)
            torch.distributed.broadcast(output_values, src, group)
        else:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_pipeline_model_parallel_group()
            torch.distributed.broadcast(output_actions, src, group)
            torch.distributed.broadcast(output_policy, src, group)
            torch.distributed.broadcast(output_values, src, group)
        # after inference, save the kv cache to the search db
        # inference_strategy.save_kv_cache(session_id)
        if init:
            parent_nodes = [None] * batch_size
            actions_taken = torch.cuda.IntTensor([-1] * batch_size)
            actions_policy = torch.cuda.FloatTensor([0.0] * batch_size)

            # construct and save the root node
            inference_strategy.save_kv_cache(session_info, context_ids, batch_size, context_lengths, parent_nodes, actions_taken, actions_policy, output_values, context_tokens)

            # construct and save the first level nodes
            for i in range(batch_size):
                context_id = context_ids[i]
                parent_nodes[i] = inference_strategy.get_node(session_info, context_id, -1)
                assert parent_nodes[i].parent is None

            for j in range(top_k):
                # actions taken is a tensor of shape [batch_size, top_k]
                # actions policy is a tensor of shape [batch_size, top_k]
                actions_taken = output_actions[:, j]
                actions_policy = output_policy[:, j]
                children_context_ids = context_ids
                inference_strategy.save_kv_cache(session_info, children_context_ids, batch_size, context_lengths, parent_nodes, actions_taken, actions_policy, None, None)
        else:
            # construct and save the next level nodes
            parent_nodes = [None] * batch_size

            for i in range(batch_size):
                context_id = context_ids[i]
                context = context_lengths[i].item()
                action = context_tokens[i, context].item()
                parent_nodes[i] = inference_strategy.get_node(session_info, context_id, action)
                parent_context_length = true_context_length[i].item() - 1
                # need to correct the parent_nodes's k-v cache
                inference_strategy.update_kv_cache(session_info, parent_nodes[i], parent_context_length, i, output_values[i].item())

            for j in range(top_k):
                actions_taken = output_actions[:, j]
                actions_policy = output_policy[:, j]
                children_context_ids = [ context_id + (parent_node.action, ) for context_id, parent_node in zip(context_ids, parent_nodes)]
                inference_strategy.save_kv_cache(session_info, children_context_ids, batch_size, true_context_length, parent_nodes, actions_taken, actions_policy, None, None)
        return output_actions, output_policy, output_values
