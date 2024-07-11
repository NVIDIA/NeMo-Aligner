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

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from nemo_aligner.utils import parallel_state


def decode_bytes_ndarray(str_ndarray: np.ndarray) -> np.ndarray:
    str_ndarray = str_ndarray.astype("bytes")
    str_ndarray = np.char.decode(str_ndarray, encoding="utf-8")
    return str_ndarray.squeeze(axis=-1)


def lock_method(lock_name):
    """
    Decorator to use in a class to ensure only one method is executing at a time.

    For instance:

        class MyClass:

            def __init__(self):
                self.my_lock = threading.Lock()

            @lock_method("self.my_lock")
            def method1(self):
                return ...

            @lock_method("self.my_lock")
            def method2(self):
                return ...
    """
    # We enforce the usage of the "self." prefix to make it explicit where the lock comes from.
    prefix = "self."
    assert lock_name.startswith(prefix), f"`lock_name` ({lock_name}) must start with '{prefix}'"
    lock_name = lock_name[len(prefix) :]

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with getattr(self, lock_name):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def pad_input(value: Optional[np.ndarray], size: int, pad_value: int = 0):
    """pad the input to a multiple of `size` and return it as a list"""
    extra = 0
    if value is not None:
        if value.dtype == bytes:
            value = decode_bytes_ndarray(value)
        if value.shape[0] % size != 0:
            extra = size - (value.shape[0] % size)

            pad_width = [(0, extra)] + [(0, 0)] * (value.ndim - 1)
            value = np.pad(value, pad_width=pad_width, mode="constant", constant_values=pad_value)
        value = value.tolist()
    return value, extra


def calculate_inference_batch_padding_multiple(current_size, model_forward_micro_batch_size):
    """calculates the multiple to pad an inference batch up to. If the batch is smaller
        than the total inference size, then we pad up to a multiple of DP. Otherwise
        we pad to a multiple of model_forward_mbs * DP
    """
    total_inference_size = model_forward_micro_batch_size * parallel_state.get_data_parallel_world_size()

    if current_size <= total_inference_size:
        return parallel_state.get_data_parallel_world_size()

    return total_inference_size


def process_inputs(inputs, tokenize_func):
    sentences = inputs.get("sentences", None)
    if sentences is not None:
        sentences = decode_bytes_ndarray(sentences)
        tokens, sequence_lengths = tokenize_func(sentences)
        sequence_lengths = sequence_lengths.unsqueeze(-1)
    else:
        tokens = torch.as_tensor(inputs["tokens"], dtype=torch.long, device=torch.cuda.current_device())
        sequence_lengths = torch.as_tensor(
            inputs["sequence_lengths"], dtype=torch.long, device=torch.cuda.current_device()
        )

    return tokens, sequence_lengths


def pad_batch_and_strip_sequence(tokens, sequence_lengths, pad_to_multiple, strip_sequence_length_to_multiple=None):
    prestrip_sequence_length = tokens.shape[1]
    if strip_sequence_length_to_multiple is not None:
        stripped_sequence_length = (
            math.ceil(sequence_lengths.max().item() / strip_sequence_length_to_multiple)
            * strip_sequence_length_to_multiple
        )
        if stripped_sequence_length < prestrip_sequence_length:
            tokens = tokens[:, :stripped_sequence_length]

    # padding on the batch dim
    num_extra = tokens.size(0) % pad_to_multiple
    amount_to_pad = 0 if num_extra == 0 else pad_to_multiple - num_extra

    tokens = F.pad(tokens, (0, 0, 0, amount_to_pad), mode="constant", value=0)
    sequence_lengths = F.pad(sequence_lengths, (0, 0, 0, amount_to_pad), mode="constant", value=0)

    assert len(tokens) == len(
        sequence_lengths
    ), "length of tokens and sequence lengths must be the same, but got {} and {}".format(
        len(tokens), len(sequence_lengths)
    )
    return {"inputs": tokens, "sequence_length": sequence_lengths}, amount_to_pad, prestrip_sequence_length


class FutureResult(ABC):
    """Generic class for trainers to wait if an object is an instance of this class
    """

    @abstractmethod
    def result(self):
        """called by the trainer to get the result must be broadcasted to all ranks
        """
