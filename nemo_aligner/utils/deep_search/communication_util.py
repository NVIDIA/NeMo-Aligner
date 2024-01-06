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

import pickle

import numpy as np
import torch


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

def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    world_size = torch.distributed.get_world_size()
    all_ranks = np.arange(world_size)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    # [pipeline dim, data parallel, tensor dim]
    all_ranks = all_ranks.reshape(pp_size, -1, tp_size)
    dp_rank = parallel_state.get_data_parallel_rank()
    return all_ranks[:, dp_rank, :].min()


def send_generate_info(
    context_tokens_tensor,
    context_length_tensor,
    action,
    depth,
    tokens_to_generate,
    top_k,
    end_strings,
    session_id,
):
    """
    Needs to be synced up with receive_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = get_model_parallel_src_rank()
    # Send the sizes of the tensors
    input_info = [
        context_tokens_tensor.size(0),  # batch_size
        context_tokens_tensor.size(1),  # seq_len
        tokens_to_generate,
        top_k,
    ]
    input_info_tensor = torch.cuda.IntTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)

    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    # send action and depth
    torch.distributed.broadcast(action, src, model_parallel_group)
    torch.distributed.broadcast(depth, src, model_parallel_group)

    # send end strings
    string_tensor = torch.as_tensor(
        np.frombuffer(pickle.dumps(end_strings), dtype=np.int8), device=torch.cuda.current_device()
    )
    size = torch.as_tensor([string_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
    torch.distributed.broadcast(size, src, model_parallel_group)
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)

    # send session id
    string_tensor = torch.as_tensor(
        np.frombuffer(pickle.dumps(session_id), dtype=np.int8), device=torch.cuda.current_device()
    )
    size = torch.as_tensor([string_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
    torch.distributed.broadcast(size, src, model_parallel_group)
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)

def receive_generate_info():
    """
    Needs to be synced up with send_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = get_model_parallel_src_rank()
    input_info_tensor = torch.empty(4, dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
    batch_size = int(input_info_tensor[0].item())
    seq_len = int(input_info_tensor[1].item())
    tokens_to_generate = int(input_info_tensor[2].item())
    top_k = int(input_info_tensor[3].item())

    context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())
    # receive context, length tensors variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    # receive action and depth
    action = torch.empty(batch_size, 1, dtype=torch.int32, device=torch.cuda.current_device())
    depth = torch.empty(batch_size, 1, dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.broadcast(action, src, model_parallel_group)
    torch.distributed.broadcast(depth, src, model_parallel_group)


    # receive end strings
    # get the size of the string
    array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.broadcast(array_size, src, model_parallel_group)

    # get the string
    string_tensor = torch.empty(array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)
    bytes = string_tensor.cpu().numpy().tobytes()
    end_strings = pickle.loads(bytes)

    # receive session id
    # get the size of the string
    array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.broadcast(array_size, src, model_parallel_group)

    string_tensor = torch.empty(array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)
    bytes = string_tensor.cpu().numpy().tobytes()
    session_id = pickle.loads(bytes)

    return (
        context_tokens_tensor,
        context_length_tensor,
        action,
        depth,
        tokens_to_generate,
        top_k,
        end_strings,
        session_id,
    )
