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

from pytriton.client import FuturesModelClient
from concurrent import futures
from typing import Dict, Tuple, List, Optional, Union
import requests
import torch

from nemo.utils import logging
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp

def get_mp_future_result(future, *keys):
    """It waits for the result of the future to be ready, gets the value with the given key,
    and broadcasts it to the model parallel group. Then it returns it as output.
    """
    output = None if future is None else future.result()

    results = []

    for key in keys:

        result = None
        if output is not None:
            result = torch.tensor(output[key], device=torch.cuda.current_device())

        ten = broadcast_2d_tensor_within_mp(result)

        results.append(ten)

    if len(results) == 1:
        return results[0]

    return results

class FlaskCommunicator:
    """
    Communicator class for async requests to flask servers
    """
    def __init__(self, servers: Dict[str, Dict[str, Union[str, int]]]):
        """
        servers: A dictionary of server names to server ips + ports
                 requests will be sent to {ip}:{port}/{name}
        """
        self.executor = futures.ThreadPoolExecutor(max_workers=8 * len(servers))
        self.servers = servers

    def send_data_to_server(self, name: str, data: List[Dict]) -> Optional[FutureResult]:
        ip = self.servers[name]["ip"]
        port = self.servers[name]["port"]
        url = f"http://{ip}:{port}/{name}"

        future = self.executor.submit(lambda: requests.post(url, json=data))
        return future

    def get_result(self, future: futures.Future, *keys):
        resp = None if future is None else future.result()
        output = None
        if resp is not None:
            output = resp.json()

        results = []
        for key in keys:
            result = None
            if output is not None:
                result = torch.tensor(output[key], device=torch.cuda.current_device())

            ten = broadcast_2d_tensor_within_mp(result)
            results.append(ten)

        if len(results) == 1:
            return results[0]

        return results

class HTTPCommunicator:
    """Communicator class for the actor to send async requests to the remote servers
    """

    def __init__(self, init_timeout_s=6000):
        super().__init__()
        self.connections = {}
        self.headers = {"Content-Type": "application/json"}
        self.init_timeout_s = init_timeout_s

    @classmethod
    def create_http_communicator_from_dict(cls, servers):
        communicator = cls()
        communicator.connections = {}
        for server_name, (ip, port) in servers.items():
            communicator.add_server_by_name(server_name=server_name, ip=ip, port=port)
        communicator.print_server_dict()
        return communicator

    def add_server_by_name(self, server_name, ip="localhost", port=5555):
        url = f"http://{ip}:{port}"
        client = FuturesModelClient(url, server_name, init_timeout_s=self.init_timeout_s, inference_timeout_s=600000)
        self.connections[server_name] = (ip, port, client)

    def print_server_dict(self):
        logging.info("====== Server connections: ======")
        logging.info("")
        if len(self.connections) == 0:
            logging.info("No server connections found.")
            logging.info("")

        for server_name, (ip, port, _) in self.connections.items():
            logging.info(f"Server Name: {server_name}")
            logging.info(f"         IP: {ip}")
            logging.info(f"       Port: {port}")
            logging.info("")

        logging.info("=================================")

    def send_data_to_server(self, server_name, data, batching=True):
        *_, client = self.connections[server_name]
        output_future = client.infer_batch(**data) if batching else client.infer_sample(**data)
        return output_future

    def close(self):
        for server_name, (ip, port, client) in self.connections.items():
            logging.info(f"Cleaning up communicator: {server_name=!r} {ip=!r} {port=!r}")
            client.close()
