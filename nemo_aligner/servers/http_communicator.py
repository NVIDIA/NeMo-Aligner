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

from nemo.utils import logging


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
