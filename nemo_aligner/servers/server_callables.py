# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import threading
from typing import Dict

import numpy as np
import torch
from megatron.core import parallel_state
from pytriton.decorators import batch
from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config import Tensor

from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.server_utils import decode_bytes_ndarray, lock_method, pad_input


def run_rm_or_critic_inference(infer_fn, inputs):
    """run the infer function for either the critic or the rm
    """
    sentences = inputs.pop("sentences", None)
    if sentences is not None:
        sentences = decode_bytes_ndarray(sentences)
    tokens = inputs.pop("tokens", None)

    sequence_lengths = inputs.pop("sequence_lengths", None)
    add_EOS = inputs.pop("add_EOS", None)

    assert sentences is not None or tokens is not None, "Both sentences and tokens cannot be None."

    dp_size = parallel_state.get_data_parallel_world_size()

    # Ensure that the batch size is a multiple of the data parallel size. Otherwise, pad it.
    sentences, extra_sentences = pad_input(sentences, dp_size)
    tokens, extra_tokens = pad_input(tokens, dp_size)
    sequence_lengths, extra_sequence_lengths = pad_input(sequence_lengths, dp_size)

    if add_EOS is not None:
        add_EOS = add_EOS[0]

    inputs = sentences if sentences is not None else tokens
    extra = extra_sentences if sentences is not None else extra_tokens
    if sequence_lengths is not None:
        assert len(inputs) == len(sequence_lengths)
        assert extra_sequence_lengths == extra

    try:
        *list_outputs, exceeded = infer_fn(inputs=inputs, sequence_length=sequence_lengths, add_EOS=add_EOS)

        processed_outputs = []

        for output in list_outputs:
            output = torch.cat(output, dim=0)
            # unpad
            output = output[: output.size(0) - extra]

            processed_outputs.append(output.cpu().numpy())

        exceeded = exceeded[: len(exceeded) - extra]

    except RuntimeError as e:
        raise PyTritonUnrecoverableError(f"Fatal error occurred - no further inferences possible. {e}") from e

    return (*processed_outputs, np.array(exceeded, dtype=np.int32).reshape(-1, 1))


class RewardModelCallable:
    def __init__(self, *, model_name: str, infer_fn: callable, lock: threading.Lock):
        self.model_name = model_name
        self.lock = lock
        self.infer_fn = infer_fn
        self.inputs = (
            Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="tokens", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="sequence_lengths", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="add_EOS", shape=(1,), dtype=np.bool_, optional=True),
        )
        self.outputs = (
            Tensor(name="rewards", shape=(1,), dtype=np.float32),
            Tensor(name="exceeded", shape=(1,), dtype=np.int32),
        )

    @batch
    @lock_method("self.lock")
    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        choice = ServerSignal.FORWARD.cuda()
        torch.distributed.broadcast(choice, 0)

        rewards, exceeded = run_rm_or_critic_inference(self.infer_fn, inputs=inputs)

        output_dict = {
            "rewards": rewards,
            "exceeded": exceeded,
        }
        return output_dict
