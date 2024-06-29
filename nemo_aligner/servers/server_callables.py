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

import torch
import torch.nn.functional as F

from nemo_aligner.utils.server_utils import decode_bytes_ndarray


def process_inference_request(inputs, pad_to_multiple, tokenize_func=None, strip_sequence_length_to_multiple=None):
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

    # strip along sequence dim
    prestrip_sequence_length = tokens.shape[1]
    if strip_sequence_length_to_multiple is not None:
        stripped_sequence_length = (
            math.ceil(sequence_lengths.max().item() / strip_sequence_length_to_multiple)
            * strip_sequence_length_to_multiple
        )
        if stripped_sequence_length < prestrip_sequence_length:
            tokens = tokens[:, :stripped_sequence_length]

    # padding on the batch dim
    _, amount_to_pad = divmod(tokens.size(0), pad_to_multiple)
    tokens = F.pad(tokens, (0, 0, 0, amount_to_pad), mode="constant", value=0)
    sequence_lengths = F.pad(sequence_lengths, (0, 0, 0, amount_to_pad), mode="constant", value=0)

    assert len(tokens) == len(
        sequence_lengths
    ), "length of tokens and sequence lengths must be the same, but got {} and {}".format(
        len(tokens), len(sequence_lengths)
    )
    return {"inputs": tokens, "sequence_length": sequence_lengths}, amount_to_pad, prestrip_sequence_length
