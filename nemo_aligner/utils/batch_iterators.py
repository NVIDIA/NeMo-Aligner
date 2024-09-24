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
        return DefaultBatchIterator

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
