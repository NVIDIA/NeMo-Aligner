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
from typing import Dict, List

import numpy as np
import torch
from megatron.core import parallel_state
from pytriton.decorators import batch, get_inference_request_batch_size
from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config import Tensor

from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.server_utils import decode_bytes_ndarray, lock_method, pad_input
import base64


def encode_context_data(context_data: List[List]):
    context_data = [np.array(t, dtype=np.int32).tostring() for t in context_data]
    str_context = [base64.b64encode(t).decode()  for t in context_data]
    str_ndarray = np.array(str_context)[..., np.newaxis]
    context_data = np.char.encode(str_ndarray, "utf-8")
    return context_data

def decode_context_data(context_data: np.ndarray):
    decoded_str = decode_bytes_ndarray(context_data)
    decode_str = [base64.b64decode(t) for t in decoded_str]
    context = [tuple(np.frombuffer(t, dtype=np.int32)) for t in decode_str]
    return context

class SearchCallable:
    def __init__(self, *, model_name: str, infer_fn: callable, lock: threading.Lock):
        self.model_name = model_name
        self.lock = lock
        self.infer_fn = infer_fn
        self.inputs = (Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
                       Tensor(name="context_ids", shape=(-1,), dtype=bytes, optional=False),
                       Tensor(name="action", shape=(-1,), dtype=np.int32, optional=True))
        self.outputs = (Tensor(name="action", shape=(-1,), dtype=np.int32),
                        Tensor(name="policy", shape=(-1,), dtype=np.float32),
                        Tensor(name="value", shape=(-1,), dtype=np.float32),
                        )

    @lock_method("self.lock")
    def infer(self, requests):
        sessions = {}
        for request in requests:
            session_info = request.parameters["session"]
            # group requests by session
            sessions.setdefault(session_info, []).append(request)
        outputs = {}
        for key in sessions:
            choice = ServerSignal.FORWARD.cuda()
            torch.distributed.broadcast(choice, 0)
            session_requests = sessions[key]
            sentences, action, context_ids = self.batch_inputs(session_requests)
            output = self.infer_fn(sentences, action, context_ids, key)
            outputs[key] = output
        outputs = self._split_result(outputs, requests)
        return outputs

    def _get_batch(self, **inputs: np.ndarray):
        sentences = inputs.pop("sentences", None)
        if sentences is not None:
            sentences = decode_bytes_ndarray(sentences)
            sentences = [i.item() for i in sentences]

        context_data = inputs.pop("context_ids", None)
        context_ids = decode_context_data(context_data)

        action = inputs.pop("action", None)

        return sentences, action, context_ids
    
    def batch_inputs(self, req_list):
        input_names = req_list[0].keys()
        for req_dict2 in req_list[1:]:
            if input_names != req_dict2.keys():
                raise ValueError("Cannot batch requests with different set of inputs keys")

        inputs = {}
        for model_input in input_names:
            concatenated_input_data = np.concatenate([req[model_input] for req in req_list])
            inputs[model_input] = concatenated_input_data
        outputs = self._get_batch(**inputs)
        return outputs
    

    def _split_result(self, outputs, req_list):

        # batch_size for each of the sessions 
        # session_batch_size = {'session1': [batch_size1, batch_size2, ...], 'session2': [batch_size1, batch_size2, ...]}
        session_batch_size = {}
        for request in req_list:
            session_info = request.parameters["session"]
            session_batch_size.setdefault(session_info, [0]).append(get_inference_request_batch_size(request))
        # get cumsum of batch_size for each session
        for key in session_batch_size:
            session_batch_size[key] = np.cumsum(session_batch_size[key]).tolist()

        out_list = []
        for request in req_list:
            session_info = request.parameters["session"]
            output = outputs[session_info]
            output_names = output.keys()
            batch_sizes = session_batch_size[session_info]
            start_idx = batch_sizes.pop(0)
            end_idx = batch_sizes[0]
            req_output_dict = {}
            for _output_ind, output_name in enumerate(output_names):
                req_output = output[output_name][start_idx:end_idx, ...]
                req_output_dict[output_name] = req_output
            out_list.append(req_output_dict)
        return out_list
