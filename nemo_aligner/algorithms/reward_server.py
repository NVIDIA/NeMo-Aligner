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

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import run_distributed_inference
from nemo_aligner.utils.server_utils import (
    calculate_inference_batch_padding_multiple,
    lock_method,
    pad_batch_and_strip_sequence,
    process_inputs,
)

ENDPOINT_BIND_ADDRESS = "0.0.0.0"
MAX_BATCH = 9999999


@dataclass
class RewardModelServer:
    """Class that implements Serving the Reward Model"""

    infer_fn: Callable
    tokenize_func: Callable
    model_name: str
    port: int
    inference_micro_batch_size: Union[int, List]
    model_forward_micro_batch_size: int
    strip_sequence_length_to_multiple: Optional[int]
    max_queue_delay_microseconds: float = 2000

    def __post_init__(self):
        self.lock = threading.Lock()
        self.inputs = (
            Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="tokens", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="sequence_lengths", shape=(-1,), dtype=np.int64, optional=True),
        )
        self.outputs = (Tensor(name="rewards", shape=(1,), dtype=np.float32),)
        if isinstance(self.inference_micro_batch_size, int):
            self.inference_micro_batch_size = [self.inference_micro_batch_size]

        self.preferred_batch_size = [
            item * parallel_state.get_data_parallel_world_size() for item in self.inference_micro_batch_size
        ]

    @batch
    @lock_method("self.lock")
    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        choice = ServerSignal.FORWARD.cuda()
        torch.distributed.broadcast(choice, 0)

        tokens, sequence_lengths = process_inputs(inputs, self.tokenize_func)
        pad_batch_to_multiple = calculate_inference_batch_padding_multiple(
            tokens.shape[0], self.model_forward_micro_batch_size
        )
        inputs, extra, _ = pad_batch_and_strip_sequence(
            tokens,
            sequence_lengths,
            pad_to_multiple=pad_batch_to_multiple,
            strip_sequence_length_to_multiple=self.strip_sequence_length_to_multiple,
        )
        rewards = run_distributed_inference(inputs=inputs, infer_fn=self.infer_fn)

        rewards = rewards[: rewards.shape[0] - extra]

        output_dict = {
            "rewards": rewards,
        }

        return output_dict

    def run_server(self):
        if torch.distributed.get_rank() == 0:
            triton_config = TritonConfig(
                allow_http=True,
                allow_grpc=False,
                allow_metrics=False,
                http_address=ENDPOINT_BIND_ADDRESS,
                http_port=self.port,
            )

            dynamic_batcher = DynamicBatcher(
                max_queue_delay_microseconds=self.max_queue_delay_microseconds,
                preferred_batch_size=self.preferred_batch_size,
            )

            # we cut the batch into pieces so we don't need to have a max batch size
            infer_model_config = ModelConfig(batching=True, max_batch_size=MAX_BATCH, batcher=dynamic_batcher)

            with Triton(config=triton_config) as triton:
                triton.bind(
                    model_name=self.model_name,
                    infer_func=self.infer,
                    inputs=self.inputs,
                    outputs=self.outputs,
                    config=infer_model_config,
                )
                triton.serve()

        else:
            self.run_subscriber_loop()

    def run_subscriber_loop(self):
        while True:
            command = ServerSignal.INVALID.cuda()
            torch.distributed.broadcast(command, 0)
            op = command.item()

            if op == ServerSignal.FORWARD:
                run_distributed_inference(infer_fn=self.infer_fn)
            else:
                raise RuntimeError(f"Invalid operation: {op}")
