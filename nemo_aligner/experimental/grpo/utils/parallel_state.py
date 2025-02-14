# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Wrapper around mcore parallel state to handle cases of resharding"""

from contextlib import contextmanager

import torch

from megatron.core import parallel_state as mcore_parallel_state

from nemo_aligner.utils.distributed import gather_and_sort_lists_of_lists
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_model_parallel_src_rank as nemo_get_model_parallel_src_rank,
)

_INFERENCE_RESHARD = False

_GROUP_TO_RANKS_CACHE = {}
_RESHARDED_DP_GROUP = None
_NODE_GROUP = None


def enable_inference_reshard_calls():
    global _INFERENCE_RESHARD
    _INFERENCE_RESHARD = True


def disable_inference_reshard_calls():
    global _INFERENCE_RESHARD
    _INFERENCE_RESHARD = False


def is_inference_reshard():
    return _INFERENCE_RESHARD


"""
The following functions check if you are in an inference resharding context
and return the 'current' sharding. 
"""
def get_model_parallel_src_rank():
    src_rank = (
        mcore_parallel_state.get_tensor_model_parallel_src_rank()
        if is_inference_reshard()
        else nemo_get_model_parallel_src_rank()
    )

    return src_rank


def get_model_parallel_group():
    group = (
        mcore_parallel_state.get_tensor_model_parallel_group()
        if is_inference_reshard()
        else mcore_parallel_state.get_model_parallel_group()
    )
    return group


def get_data_parallel_world_size():
    data_parallel_size = mcore_parallel_state.get_data_parallel_world_size()

    return (
        data_parallel_size * mcore_parallel_state.get_pipeline_model_parallel_world_size()
        if is_inference_reshard()
        else data_parallel_size
    )


def get_data_parallel_rank():
    data_parallel_rank = mcore_parallel_state.get_data_parallel_rank()

    if is_inference_reshard():
        data_parallel_rank = data_parallel_rank + (
            mcore_parallel_state.get_data_parallel_world_size()
            * mcore_parallel_state.get_pipeline_model_parallel_rank()
        )

    return data_parallel_rank

def get_data_parallel_group():
    if is_inference_reshard():
        global _RESHARDED_DP_GROUP
        if _RESHARDED_DP_GROUP is None:
            # gather all dp and pp ranks into one list
            pp_ranks = torch.tensor(get_all_rank_ids_in_group(mcore_parallel_state.get_pipeline_model_parallel_group()), dtype=torch.int, device=torch.cuda.current_device())
            # distributed all gather
            ppdp = [torch.empty_like(pp_ranks) for _ in range(mcore_parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather(ppdp, pp_ranks, group=mcore_parallel_state.get_data_parallel_group())
            # flatten the list of tensors
            ppdp = torch.cat(ppdp).tolist()
            print(f"my ppdp: {ppdp}", flush=True)

            # gather over all tp ranks
            all_ppdp = gather_and_sort_lists_of_lists([ppdp], mcore_parallel_state.get_tensor_model_parallel_group())
            for rank_group in all_ppdp:
                print(f"DP RESHARD Rank {torch.distributed.get_rank()} Building group with ranks: {rank_group}",flush=True)
                group = torch.distributed.new_group(list(rank_group))
                if int(torch.distributed.get_rank()) in list(rank_group):
                    _RESHARDED_DP_GROUP = group
                    print(f"DP RESHARD Rank {torch.distributed.get_rank()} local Gloo group built successfully",flush=True)

        return _RESHARDED_DP_GROUP
    else:
        return mcore_parallel_state.get_data_parallel_group()


def get_tensor_model_parallel_world_size():
    return mcore_parallel_state.get_tensor_model_parallel_world_size()

def get_pipeline_model_parallel_world_size():
    return 1 if is_inference_reshard() else mcore_parallel_state.get_pipeline_model_parallel_world_size()

def get_pipeline_model_parallel_group():
    group = (
        mcore_parallel_state.get_pipeline_model_parallel_group()
        if is_inference_reshard()
        else mcore_parallel_state.get_pipeline_model_parallel_group()
    )
    return group

def is_model_parallel_src_rank():
    return torch.distributed.get_rank() == get_model_parallel_src_rank()

def get_node_group():
    global _NODE_GROUP
    if _NODE_GROUP is None:
        import socket
        # Get the hostname of the current machine.
        local_hostname = socket.gethostname()
        world_size = torch.distributed.get_world_size()
        # Gather hostnames from all ranks.
        hostnames = [None] * world_size
        torch.distributed.all_gather_object(hostnames, local_hostname)
        # Build a mapping from hostname to the ordered list of ranks (preserving original order).
        node_order = []
        node_to_ranks = {}
        for rank, host in enumerate(hostnames):
            if host not in node_to_ranks:
                node_to_ranks[host] = []
                node_order.append(host)
            node_to_ranks[host].append(rank)
        # Construct process groups for every node following the established order.
        node_groups = {}
        for host in node_order:
            node_groups[host] = torch.distributed.new_group(ranks=node_to_ranks[host])
        # Set _NODE_GROUP as the group corresponding to the current node.
        _NODE_GROUP = node_groups[local_hostname]
        torch.cuda.synchronize()
    return _NODE_GROUP

"""
These functions will ignore your current 'resharded' context and return 
parallism sharding for the training context.
"""
def get_training_pipeline_model_parallel_rank():
    return mcore_parallel_state.get_pipeline_model_parallel_rank()

def get_training_pipeline_model_parallel_world_size():
    return mcore_parallel_state.get_pipeline_model_parallel_world_size()

def get_training_pipeline_model_parallel_group():
    return mcore_parallel_state.get_pipeline_model_parallel_group()

def get_training_data_parallel_rank():
    return mcore_parallel_state.get_data_parallel_rank()

def get_training_data_parallel_group():
    return mcore_parallel_state.get_data_parallel_group()

def get_training_data_parallel_world_size():
    return mcore_parallel_state.get_data_parallel_world_size()

def get_training_tensor_model_parallel_group():
    return mcore_parallel_state.get_tensor_model_parallel_group()

def get_training_tensor_model_parallel_src_rank():
    return mcore_parallel_state.get_tensor_model_parallel_src_rank()

def get_all_rank_ids_in_group(group):
    if group in _GROUP_TO_RANKS_CACHE:
        return _GROUP_TO_RANKS_CACHE[group]

    curr_global_rank = int(torch.distributed.get_rank())
    group_size = torch.distributed.get_world_size(group=group)
    global_rank_tensor = torch.tensor([curr_global_rank], dtype=torch.int, device=torch.cuda.current_device())
    global_ranks = [torch.empty(1, dtype=torch.int, device=torch.cuda.current_device()) for _ in range(group_size)]
    torch.distributed.all_gather(global_ranks, global_rank_tensor, group=group)
    _GROUP_TO_RANKS_CACHE[group] = [int(global_ranks[i].item()) for i in range(group_size)]
    return _GROUP_TO_RANKS_CACHE[group]
    

@contextmanager
def inference_reshard_region():
    """mutates global state so distributed call are aware of inference backend resharding
        from PP to TP only
    """
    try:
        enable_inference_reshard_calls()
        yield
    finally:
        disable_inference_reshard_calls()


def __getattr__(name):
    if is_inference_reshard():
        raise NotImplementedError(
            f"reshard is currently enabled, but called a parallel state function {name} that aligner doesn't implement with resharding."
        )

    return getattr(mcore_parallel_state, name)
