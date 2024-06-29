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

from megatron.core import parallel_state as mcore_parallel_state

from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_model_parallel_src_rank as nemo_get_model_parallel_src_rank,
)


def get_model_parallel_src_rank():
    src_rank = nemo_get_model_parallel_src_rank()

    return src_rank


def __getattr__(name):
    return getattr(mcore_parallel_state, name)
