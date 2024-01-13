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
from megatron.core import parallel_state
from pytriton.decorators import batch, get_inference_request_batch_size
from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config import Tensor

from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.server_utils import decode_bytes_ndarray, lock_method, pad_input


def run_rm_or_critic_inference(infer_fn, inputs):
    """run the infer function for either the critic or the rm
    """
    sentences = inputs.pop("sentences", None)
    if sentences is not None:
        sentences = decode_bytes_ndarray(sentences)
    tokens = inputs.pop("tokens", None)

    sequence_lengths = inputs.pop("sequence_lengths", None)
    add_EOS = inputs.pop("add_EOS", None)

    assert sentences is not None or tokens is not None, "Both sentences and tokens cannot be None."

    dp_size = parallel_state.get_data_parallel_world_size()

    # Ensure that the batch size is a multiple of the data parallel size. Otherwise, pad it.
    sentences, extra_sentences = pad_input(sentences, dp_size)
    tokens, extra_tokens = pad_input(tokens, dp_size)
    sequence_lengths, extra_sequence_lengths = pad_input(sequence_lengths, dp_size)

    if add_EOS is not None:
        add_EOS = add_EOS[0]

    inputs = sentences if sentences is not None else tokens
    extra = extra_sentences if sentences is not None else extra_tokens
    if sequence_lengths is not None:
        assert len(inputs) == len(sequence_lengths)
        assert extra_sequence_lengths == extra

    try:
        *list_outputs, exceeded = infer_fn(inputs=inputs, sequence_length=sequence_lengths, add_EOS=add_EOS)

        processed_outputs = []

        for output in list_outputs:
            output = torch.cat(output, dim=0)
            # unpad
            output = output[: output.size(0) - extra]

            processed_outputs.append(output.cpu().numpy())

        exceeded = exceeded[: len(exceeded) - extra]

    except RuntimeError as e:
        raise PyTritonUnrecoverableError(f"Fatal error occurred - no further inferences possible. {e}") from e

    return (*processed_outputs, np.array(exceeded, dtype=np.int32).reshape(-1, 1))


class SearchCallable:
    def __init__(self, *, model_name: str, infer_fn: callable, lock: threading.Lock):
        self.model_name = model_name
        self.lock = lock
        self.infer_fn = infer_fn
        self.inputs = (Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
                       Tensor(name="context_ids", shape=(-1,), dtype=bytes, optional=False),
                       Tensor(name="action", shape=(-1,), dtype=np.int32, optional=True),
                       Tensor(name="depth", shape=(-1,), dtype=np.int32, optional=True),)
        self.outputs = (Tensor(name="action", shape=(-1,), dtype=np.int32),
                        Tensor(name="policy", shape=(-1,), dtype=np.float32),
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
            sentences, action, depth, context_ids = self.batch_inputs(session_requests)
            output = self.infer_fn(sentences, action, depth, context_ids)
            outputs[key] = output
        outputs = self._split_result(outputs, requests)
        return outputs

    def _get_batch(self, **inputs: np.ndarray):
        sentences = inputs.pop("sentences", None)
        if sentences is not None:
            sentences = decode_bytes_ndarray(sentences)
            sentences = [i.item() for i in sentences]

        context_data = inputs.pop("context_ids", None)
        context = decode_bytes_ndarray(context_data)
        context = [i.item() for i in context] 

        action = inputs.pop("action", None)

        depth = inputs.pop("depth", None)
        return sentences, action, depth, context
    
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
