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

from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, gather_tensor, run_if_model_parallel_src
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.models.nlp.gpt.reward_critic_clients import get_future_result, RMFutureResult

"""A remote client that acts like a real Reward Model and Critic forwards all requests from the actor
    over to the remote PyTrition server
"""

def triton_textencode(text_batch: List[str]):
    enc = np.array([[np.char.encode(i, 'utf-8')] for i in text_batch])
    enc = np.reshape(enc, (enc.shape[0], 1))

    return enc

@dataclass
class RemoteGraderClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        server_dict = {cfg.reward_model.name: (cfg.reward_model.ip, cfg.reward_model.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()

    def infer_rm(self, rollout_batch):
        # response_tokens = rollout_batch["response_tokens"].cpu()
        response_sentences = rollout_batch["response_sentences"]
        ground_truths = rollout_batch["ground_truths"]
        # og_seq_length = response_tokens.size(-1)

        send_data = {
            "pred_responses": triton_textencode(response_sentences),
            "ground_truth": triton_textencode(ground_truths),
            # "sequence_lengths": rollout_batch["response_lengths"].unsqueeze(1).cpu().numpy(),
        }

        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
        )

        return RMFutureResult(rm_future)
