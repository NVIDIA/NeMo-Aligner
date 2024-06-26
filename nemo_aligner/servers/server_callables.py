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
import numpy as np

from nemo_aligner.utils.server_utils import decode_bytes_ndarray, pad_input


def process_inference_request(inputs, pad_to, pad_sequence_length_to_multiple=None, tokenize_func=None):
    sentences = inputs.pop("sentences", None)
    if sentences is not None:
        sentences = decode_bytes_ndarray(sentences)
        tokens, sequence_lengths = tokenize_func(
            sentences, pad_sequence_length_to_multiple=pad_sequence_length_to_multiple
        )
    else:
        tokens = inputs.pop("tokens", None)
        sequence_lengths = inputs.pop("sequence_lengths", None)

    prepad_sequence_length = tokens.shape[1]
    tokens, extra_tokens = pad_input(tokens, pad_to)
    sequence_lengths, _ = pad_input(sequence_lengths, pad_to)

    assert len(tokens) == len(
        sequence_lengths
    ), "length of tokens and sequence lengths must be the same, but got {} and {}".format(
        len(tokens), len(sequence_lengths)
    )
    return {"inputs": tokens, "sequence_length": sequence_lengths}, extra_tokens, prepad_sequence_length
