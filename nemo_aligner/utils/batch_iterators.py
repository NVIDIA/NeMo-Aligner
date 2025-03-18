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

import itertools
import json
import socket
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

import requests
import torch
import torch.distributed
from flask import Flask, request

from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, run_if_model_parallel_src
from nemo_aligner.utils.utils import get_global_set
from nemo_aligner.experimental.grpo.utils import parallel_state


def _send_request(host, port, endpoint="get_idx", batch_size=1):
    output = run_if_model_parallel_src(
        requests.put,
        url=f"http://{host}:{port}/{endpoint}",
        data=json.dumps({"batch_size": batch_size}),
        headers={"Content-Type": "application/json"},
    )

    if output is not None:
        output = output.json()
        output = torch.as_tensor(output, dtype=torch.long).view(1, -1)

    output = broadcast_2d_tensor_within_mp(output, dtype=torch.long)
    return output.flatten().tolist()


@dataclass
class SharedSet:
    def __post_init__(self):
        self.data = set()
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.data.clear()

    def set_idx(self, ids):
        with self.lock:
            self.data.update(ids)

    def get_idx(self, batch_size):
        with self.lock:
            to_ret = [self.data.pop() for _ in range(batch_size) if len(self.data) > 0]

        return to_ret


@dataclass
class DefaultBatchIterator:
    """The default batch iterator used for getting samples for generation stage 
    """

    sampler_iter: Iterator[int]
    num_microbatches: int
    dataset: Mapping
    collate_fn: Callable

    def __iter__(self):
        for _, ids in zip(range(self.num_microbatches), self.sampler_iter):
            batch = self.collate_fn([self.dataset[index] for index in ids])
            yield batch


@dataclass
class GRPOBatchIterator:
    """The default batch iterator used for getting samples for generation stage 
    """

    sampler_iter: Iterator[int]
    micro_batch_size: int
    samples_per_prompt: int
    num_prompts_per_grpo_step: int
    dataset: Mapping
    collate_fn: Callable

    def __iter__(self):
        dp_size = parallel_state.get_data_parallel_world_size()
        # The prompt IDs we need to sample among all DP ranks
        ids = next(self.sampler_iter)
        # Repeat each prompt ID for samples_per_prompt times
        ids_with_repetitions = [x for item in ids for x in [item]*self.samples_per_prompt]
        # Suppose we have enough data, this is the amount of samples generated in one micro batch.
        global_num_samples_per_micro_batch = dp_size * self.micro_batch_size

        dp_rank = parallel_state.get_data_parallel_rank()
        global_num_samples = self.num_prompts_per_grpo_step * self.samples_per_prompt
        print(f"ids = {ids}")
        print(f"ids_with_repetitions = {ids_with_repetitions}")
        assert global_num_samples % dp_size == 0, f"global_num_samples = {global_num_samples}, num_prompts_per_grpo_step = {self.num_prompts_per_grpo_step}, samples_per_prompt = {self.samples_per_prompt}, dp_size = {dp_size}"
        for ids_offset in range(0, global_num_samples, global_num_samples_per_micro_batch):
            start_id = ids_offset
            end_id = min(ids_offset + global_num_samples_per_micro_batch, len(ids_with_repetitions))
            samples_to_be_distributed = ids_with_repetitions[start_id : end_id]
            print(f"samples_to_be_distributed = {samples_to_be_distributed}")
            num_samples_per_dp_rank = len(samples_to_be_distributed) // dp_size
            print(f"start_id = {start_id}, end_id = {end_id}, global_num_samples_per_micro_batch = {global_num_samples_per_micro_batch}, num_samples_per_dp_rank = {num_samples_per_dp_rank}")

            # Prompt IDs that we will sampe from this rank
            prompt_ids_for_the_rank = samples_to_be_distributed[dp_rank * num_samples_per_dp_rank : (dp_rank + 1) * num_samples_per_dp_rank]
            print(f"prompt_ids_for_the_rank = {prompt_ids_for_the_rank}")

            output = [self.dataset[index] for index in prompt_ids_for_the_rank]
            print(f"output = {output}")
            batch = self.collate_fn([self.dataset[index] for index in prompt_ids_for_the_rank])
            yield batch


@dataclass
class HTTPBatchIterator:
    shared_set: Union[SharedSet, None]
    host: str
    port: int
    sampler_iter: Iterator[int]
    num_microbatches: int
    dataset: Mapping
    collate_fn: Callable

    def __post_init__(self):
        local_ids = [ids for _, ids in zip(range(self.num_microbatches), self.sampler_iter)]
        self.desired_batch_size = len(local_ids[0]) if len(local_ids) > 0 else 1

        local_ids = set(itertools.chain.from_iterable(local_ids))
        global_ids = get_global_set(local_ids)

        if torch.distributed.get_rank() == 0:
            self.shared_set.clear()
            self.shared_set.set_idx(global_ids)

        torch.distributed.barrier()

    def __iter__(self):
        ids = _send_request(host=self.host, port=self.port, batch_size=self.desired_batch_size)

        while len(ids) > 0:
            batch = self.collate_fn([self.dataset[idx] for idx in ids])
            yield batch
            ids = _send_request(host=self.host, port=self.port, batch_size=self.desired_batch_size)


def get_batch_iterator_cls(batch_iterator_cfg):
    use_flask = batch_iterator_cfg.get("use_flask", False)
    if not use_flask:
        return GRPOBatchIterator

    port = batch_iterator_cfg.get("port", 5557)
    ip_address = [socket.gethostbyname(socket.gethostname())]
    torch.distributed.broadcast_object_list(ip_address, src=0, group=None, device=torch.cuda.current_device())
    flask_host = ip_address[0]
    shared_set = None

    # start the server on rank 0
    if torch.distributed.get_rank() == 0:
        shared_set = SharedSet()
        app = Flask(__name__)

        @app.route("/get_idx", methods=["PUT"])
        def get_http_idx():
            batch_size = request.get_json()["batch_size"]
            return shared_set.get_idx(batch_size)

        flask_thread = threading.Thread(
            target=lambda: app.run(host=flask_host, port=port, use_reloader=False), daemon=True
        )
        # TODO Starting the thread as daemon means it may not release all resources on exit => maybe implement a clean shutdown logic?
        flask_thread.start()

    return partial(HTTPBatchIterator, shared_set, flask_host, port)
