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
from nemo_aligner.utils.distributed import broadcast_2d_tensor, broadcast_python_obj


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
    tokens_to_generate,
    top_k,
    end_strings,
    context_ids,
    session_info,
):
    """
    Needs to be synced up with receive_generate_info
    """
    src = 0

    broadcast_2d_tensor(context_tokens_tensor, src, None, dtype=torch.int64)
    broadcast_2d_tensor(context_length_tensor.reshape(-1, 1), src, None, dtype=torch.int64)

    broadcast_2d_tensor(action, src, None, dtype=torch.int32)

    # Send the sizes of the tensors
    input_info = [
        tokens_to_generate,
        top_k,
    ]
    input_info_tensor = torch.cuda.IntTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, src, None)

    broadcast_python_obj(end_strings, src, None)
    broadcast_python_obj(context_ids, src, None)
    broadcast_python_obj(session_info, src, None)


def receive_generate_info():
    """
    Needs to be synced up with send_generate_info
    """
    src = 0

    context_tokens_tensor = broadcast_2d_tensor(None, src, None, dtype=torch.int64)
    context_length_tensor = broadcast_2d_tensor(None, src, None, dtype=torch.int64)
    context_length_tensor.squeeze_(1)

    # receive action and depth
    action = broadcast_2d_tensor(None, src, None, dtype=torch.int32)

    input_info_tensor = torch.empty(2, dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, src, None)
    tokens_to_generate = int(input_info_tensor[0].item())
    top_k = int(input_info_tensor[1].item())

    end_strings = broadcast_python_obj(None, src, None)
    context_ids = broadcast_python_obj(None, src, None)
    session_info = broadcast_python_obj(None, src, None)

    return (
        context_tokens_tensor,
        context_length_tensor,
        action,
        tokens_to_generate,
        top_k,
        end_strings,
        context_ids,
        session_info,
    )
