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
from pytriton.decorators import batch
from pytriton.model_config import Tensor

from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor, rebalance_nd_tensor
from nemo_aligner.utils.server_utils import decode_bytes_ndarray, lock_method, pad_input


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


class RewardModelCallable:
    def __init__(
        self,
        *,
        model_name: str,
        infer_fn: callable,
        tokenizer: None,
        forward_micro_batch_size: None,
        lock: threading.Lock,
        tokenize_func=None,
    ):
        self.model_name = model_name
        self.lock = lock
        self.infer_fn = infer_fn
        self.inputs = (
            Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="tokens", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="sequence_lengths", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="add_EOS", shape=(1,), dtype=np.bool_, optional=True),
        )
        self.outputs = (Tensor(name="rewards", shape=(1,), dtype=np.float32),)
        self.tokenizer = tokenizer
        self.forward_micro_batch_size = forward_micro_batch_size
        self.tokenize_func = tokenize_func

    @batch
    @lock_method("self.lock")
    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        choice = ServerSignal.FORWARD.cuda()
        torch.distributed.broadcast(choice, 0)

        inputs, extra, _ = process_inference_request(
            inputs,
            pad_to=self.forward_micro_batch_size * parallel_state.get_data_parallel_world_size(),
            pad_sequence_length_to_multiple=None,
            tokenizer=self.tokenizer,
            tokenize_func=self.tokenize_func,
        )

        rewards = self.run_inference(inputs=inputs, extra=extra)
        rewards = rewards[: rewards.shape[0] - extra]

        output_dict = {
            "rewards": rewards,
        }

        return output_dict

    @torch.no_grad()
    def run_inference(self, inputs=None, extra=None):
        """only rank 0 has valid data
        """
        tokens, lengths = None, None
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        is_rank_0 = torch.distributed.get_rank() == 0

        if is_rank_0:
            tokens = torch.as_tensor(inputs["inputs"], dtype=torch.long, device=torch.cuda.current_device())
            lengths = torch.as_tensor(inputs["sequence_length"], dtype=torch.long, device=torch.cuda.current_device())

        tokens = broadcast_2d_tensor(tokens, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank]
        lengths = broadcast_2d_tensor(lengths, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank].squeeze(-1)

        rewards = self.infer_fn(inputs=(tokens, lengths))
        rewards = rebalance_nd_tensor(rewards, group=parallel_state.get_data_parallel_group()).squeeze().cpu().numpy()

        return rewards
