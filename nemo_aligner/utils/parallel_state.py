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

"""Wrapper around mcore parallel state to handle cases of resharding"""

from contextlib import contextmanager

from megatron.core import parallel_state as mcore_parallel_state

from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_model_parallel_src_rank as nemo_get_model_parallel_src_rank,
)

_TRT_LLM_RESHARD = False


def enable_trt_llm_reshard_calls():
    global _TRT_LLM_RESHARD
    _TRT_LLM_RESHARD = True


def disable_trt_llm_reshard_calls():
    global _TRT_LLM_RESHARD
    _TRT_LLM_RESHARD = False


def is_trt_llm_reshard():
    return _TRT_LLM_RESHARD


def get_model_parallel_src_rank():
    src_rank = (
        mcore_parallel_state.get_tensor_model_parallel_src_rank()
        if is_trt_llm_reshard()
        else nemo_get_model_parallel_src_rank()
    )

    return src_rank


def get_model_parallel_group():
    group = (
        mcore_parallel_state.get_tensor_model_parallel_group()
        if is_trt_llm_reshard()
        else mcore_parallel_state.get_model_parallel_group()
    )
    return group


def get_data_parallel_world_size():
    data_parallel_size = mcore_parallel_state.get_data_parallel_world_size()

    return (
        data_parallel_size * mcore_parallel_state.get_pipeline_model_parallel_world_size()
        if is_trt_llm_reshard()
        else data_parallel_size
    )


def get_data_parallel_rank():
    data_parallel_rank = mcore_parallel_state.get_data_parallel_rank()

    if is_trt_llm_reshard():
        data_parallel_rank = data_parallel_rank + (
            mcore_parallel_state.get_data_parallel_world_size()
            * mcore_parallel_state.get_pipeline_model_parallel_rank()
        )

    return data_parallel_rank


def get_pipeline_model_parallel_world_size():
    return 1 if is_trt_llm_reshard() else mcore_parallel_state.get_pipeline_model_parallel_world_size()


@contextmanager
def trt_llm_reshard_region():
    """mutates global state so distributed call are aware of TRT-LLM resharding
        from PP to TP only
    """
    try:
        enable_trt_llm_reshard_calls()
        yield
    finally:
        disable_trt_llm_reshard_calls()


def __getattr__(name):
    if is_trt_llm_reshard():
        raise NotImplementedError(
            f"reshard is currently enabled, but called a parallel state function {name} that aligner doesn't implement with resharding."
        )

    return getattr(mcore_parallel_state, name)
