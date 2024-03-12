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

import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.transformer.text_generation import OutputType
from nemo.utils import AppState
from nemo_aligner.utils.deep_search.communication_util import receive_generate_info, send_generate_info
from nemo_aligner.utils.distributed import gather_tensor

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
        if isinstance(inputs, tuple):  # tuple of (context_tokens_tensor, context_length_tensor)
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(inputs, 0, False)
        batch_size = context_tokens_tensor.size(0)
        action = torch.cuda.IntTensor(batch_size, 1)
        action[:] = 0

    # cache_hit_indicator = []
    # if not init:
    #     # check if the data hits the cache
    #     cache_infer_results = []
    #     for a, context_id in zip(action, context_ids):
    #         infer = inference_strategy.search_db.get_infer_cache(session_info, context_id, a.item())
    #         if infer is not None:
    #             cache_hit_indicator.append(True)
    #             cache_infer_results.append(infer)
    #         else:
    #             cache_hit_indicator.append(False)
    #     cache_hit_indicator = torch.cuda.BoolTensor(cache_hit_indicator)
    #     context_tokens_tensor = context_tokens_tensor[~cache_hit_indicator]
    #     context_length_tensor = context_length_tensor[~cache_hit_indicator]
    #     action = action[~cache_hit_indicator]
    #     context_ids = [context_ids[i] for i in range(len(context_ids)) if not cache_hit_indicator[i]]

    if len(context_tokens_tensor) > 0:
        true_context_length = None
        if init:
            inference_strategy.init(context_tokens_tensor, tokens_to_generate, session_info)
        else:
            tokenizer = model.tokenizer
            (
                context_tokens_tensor,
                context_length_tensor,
                true_context_length,
            ) = inference_strategy.compute_inference_params(session_info, context_ids, action, tokenizer.pad_id)

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

    # if not init:
    #     if sum(cache_hit_indicator) > 0:
    #         actions = []
    #         policys = []
    #         values = []
    #         count = 0
    #         for cache_hit in cache_hit_indicator:
    #             if cache_hit:
    #                 infer = cache_infer_results.pop(0)
    #                 actions.append(torch.cuda.IntTensor(infer["action"]))
    #                 policys.append(torch.cuda.FloatTensor(infer["policy"]))
    #                 values.append(torch.cuda.FloatTensor(infer["value"]))
    #             else:
    #                 # output_actions are tensors of shape [batch_size, top_k]
    #                 actions.append(output_actions[count])
    #                 policys.append(output_policys[count])
    #                 values.append(output_values[count])
    #                 count += 1
    #         output_actions = torch.vstack(actions)
    #         output_policys = torch.vstack(policys)
    #         output_values = torch.hstack(values)

    output = {}
    output["action"] = output_actions
    output["policy"] = output_policys
    output["value"] = output_values
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
    **strategy_args,
) -> OutputType:
    """
    """
    if "strategy" in strategy_args:
        inference_strategy = strategy_args["strategy"]
    else:
        raise ValueError("strategy is not specified")
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
            if isinstance(inputs, tuple):  # tuple of (context_tokens_tensor, context_length_tensor)
                context_tokens_tensor, context_length_tensor = inputs
            else:
                context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(inputs, 0, False)
            batch_size = context_tokens_tensor.size(0)
            action = torch.cuda.IntTensor(batch_size, 1)
            action[:] = 0

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            action,
            tokens_to_generate,
            top_k,
            context_ids,
            session_info,
            inputs,
        )
    else:
        (
            context_tokens_tensor,
            context_length_tensor,
            action,
            tokens_to_generate,
            top_k,
            context_ids,
            session_info,
            inputs,
        ) = receive_generate_info()
    # init the objects for inference
    init = inputs is not None

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

    # cache_hit_indicator = []
    # if not init:
    #     # check if the data hits the cache
    #     cache_infer_results = []
    #     for a, context_id in zip(action, context_ids):
    #         infer = inference_strategy.search_db.get_infer_cache(session_info, context_id, a.item())
    #         if infer is not None:
    #             # print("cache hit", len(context_id), a.item())
    #             cache_hit_indicator.append(True)
    #             cache_infer_results.append(infer)
    #         else:
    #             cache_hit_indicator.append(False)
    #     cache_hit_indicator = torch.cuda.BoolTensor(cache_hit_indicator)
    #     context_tokens_tensor = context_tokens_tensor[~cache_hit_indicator]
    #     context_length_tensor = context_length_tensor[~cache_hit_indicator]
    #     action = action[~cache_hit_indicator]
    #     context_ids = [context_ids[i] for i in range(len(context_ids)) if not cache_hit_indicator[i]]

    if len(context_tokens_tensor) > 0:
        true_context_length = None
        if init:
            inference_strategy.init(context_tokens_tensor, tokens_to_generate, session_info)
        else:
            tokenizer = model.tokenizer
            (
                context_tokens_tensor,
                context_length_tensor,
                true_context_length,
            ) = inference_strategy.compute_inference_params(session_info, context_ids, action, tokenizer.pad_id)

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

    # if not init:
    #     if sum(cache_hit_indicator) > 0:
    #         actions = []
    #         policys = []
    #         values = []
    #         count = 0
    #         for cache_hit in cache_hit_indicator:
    #             if cache_hit:
    #                 infer = cache_infer_results.pop(0)
    #                 actions.append(torch.cuda.IntTensor(infer["action"]))
    #                 policys.append(torch.cuda.FloatTensor(infer["policy"]))
    #                 values.append(torch.cuda.FloatTensor(infer["value"]))
    #             else:
    #                 # output_actions are tensors of shape [batch_size, top_k]
    #                 actions.append(output_actions[count])
    #                 policys.append(output_policys[count])
    #                 values.append(output_values[count])
    #                 count += 1
    #         output_actions = torch.vstack(actions)
    #         output_policys = torch.vstack(policys)
    #         output_values = torch.hstack(values)
    #     # combine cache and non-cache results

    result_output_actions_list = gather_tensor(
        output_actions,
        dst=parallel_state.get_data_parallel_src_rank(),
        group=parallel_state.get_data_parallel_group(),
        dtype=output_actions.dtype,
    )

    result_output_policys_list = gather_tensor(
        output_policys,
        dst=parallel_state.get_data_parallel_src_rank(),
        group=parallel_state.get_data_parallel_group(),
        dtype=output_policys.dtype,
    )

    result_output_values_list = gather_tensor(
        output_values,
        dst=parallel_state.get_data_parallel_src_rank(),
        group=parallel_state.get_data_parallel_group(),
        dtype=output_values.dtype,
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
    true_context_length=None,
):
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
        batch_size, token_len = context_tokens.shape

        tokens = context_tokens

        token_to_generate = 1
        if true_context_length is not None:
            new_context_lengths = true_context_length - true_context_length.min()
            token_to_generate = (token_len - new_context_lengths).min().item()

        maxlen = token_to_generate + context_lengths.max().item()

        maxlen = inference_strategy.clip_max_len(maxlen)

        output_actions = torch.cuda.IntTensor(batch_size, top_k)
        output_policy = torch.cuda.FloatTensor(batch_size, top_k)
        output_values = torch.cuda.FloatTensor(batch_size)

        update_pos = []
        for batch_id in range(batch_size):
            batch_token = tokens[batch_id]
            for last_pos in range(token_len - 1, -1, -1):
                if batch_token[last_pos] == tokenizer.pad_id:
                    pass
                else:
                    break
            update_pos.append(last_pos)
        update_pos = torch.cuda.IntTensor(update_pos)

        batch, tensor_shape = inference_strategy.prepare_batch(
            tokens, micro_batch_size, init, session_info, true_context_length
        )
        output = inference_strategy.forward_step(batch, tensor_shape, session_info)

        # batch_update_indicator = context_lengths == context_length

        if parallel_state.is_pipeline_last_stage():
            logits = output[0]["logits"]
            logits = logits[torch.arange(batch_size), update_pos].contiguous()
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float("Inf")

            updated_logits, actions = torch.topk(logits, top_k)
            probs = F.softmax(updated_logits, dim=-1)

            output_actions = actions.type(torch.int32)
            output_policy = probs
            if "value" in output[0]:
                value = output[0]["value"]
                output_values = value[torch.arange(batch_size), update_pos].contiguous()
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
            # construct and save the root node
            # root node kv cache is all we need
            inference_strategy.save_kv_cache(
                session_info,
                context_ids,
                batch_size,
                context_lengths,
                parent_nodes,
                actions_taken,
                output_policy,
                output_values,
                output_actions,
                context_tokens,
                update_pos,
                context_lengths,
            )
        else:
            # construct and save the next level nodes
            parent_nodes = []
            for i in range(batch_size):
                context_id = context_ids[i]
                parent_node = inference_strategy.get_node(session_info, context_id)
                parent_nodes.append(parent_node)
            action_taken = torch.gather(context_tokens, 1, context_lengths.unsqueeze(-1).type(torch.int64))

            inference_strategy.save_kv_cache(
                session_info,
                context_ids,
                batch_size,
                true_context_length - 1,
                parent_nodes,
                action_taken,
                output_policy,
                output_values,
                output_actions,
                context_tokens,
                update_pos,
                context_lengths,
            )
        return output_actions, output_policy, output_values
