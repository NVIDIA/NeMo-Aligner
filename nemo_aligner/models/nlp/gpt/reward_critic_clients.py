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

import numpy as np
import torch
from megatron.core import parallel_state
from omegaconf import DictConfig

from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, gather_tensor, run_if_model_parallel_src
from nemo_aligner.utils.server_utils import FutureResult


"""A remote client that acts like a real Reward Model and Critic forwards all requests from the actor
    over to the remote PyTrition server
"""


def get_future_result(future, *keys):
    """It waits for the result of the future to be ready, gets the value with the given key,
    and broadcasts it to the model parallel group. Then it returns it as output.
    """
    output = None if future is None else future.result()

    results = []

    for key in keys:

        result = None
        if output is not None:
            result = torch.tensor(output[key], device=torch.cuda.current_device())

        results.append(broadcast_2d_tensor_within_mp(result))

    if len(results) == 1:
        return results[0]

    return results


class RMCriticFutureResult(FutureResult):
    def __init__(self, critic_future, rm_future, combine_rm_and_critic_server, og_seq_length):
        self.critic_future = critic_future
        self.rm_future = rm_future
        self.combine_rm_and_critic_server = combine_rm_and_critic_server

        self.og_seq_length = og_seq_length

    def result(self):
        if self.combine_rm_and_critic_server:
            rewards, values = get_future_result(self.critic_future, "rewards", "values")
        else:
            rewards = get_future_result(self.rm_future, "rewards")
            values = get_future_result(self.critic_future, "values")

        values = values[:, : self.og_seq_length - 1].contiguous()

        self.critic_future = None
        self.rm_future = None

        return rewards, values


class SaveFuture(FutureResult):
    def __init__(self, pytriton_save_future):
        self.pytriton_save_future = pytriton_save_future

    def result(self):
        if self.pytriton_save_future is not None:
            self.pytriton_save_future.result()

        # need to make sure it's saved
        torch.distributed.barrier()


@dataclass
class RemoteGPTRMCriticClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        critic_ip_and_port = (cfg.critic.ip, cfg.critic.port)
        server_dict = {
            cfg.critic.name.train: critic_ip_and_port,
            cfg.critic.name.infer: critic_ip_and_port,
            cfg.critic.name.save: critic_ip_and_port,
        }

        if not cfg.combine_rm_and_critic_server:
            server_dict[cfg.reward_model.name] = (cfg.reward_model.ip, cfg.reward_model.port)

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.combine_rm_and_critic_server = self.cfg.combine_rm_and_critic_server
        self.pad_to_length = self.cfg.pad_to_length

    def infer_rm_critic(self, rollout_batch):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        if self.pad_to_length is not None:
            assert (
                og_seq_length <= self.pad_to_length
            ), f"original shape before padding {og_seq_length} is higher than {self.pad_to_length}"
            response_tokens = torch.nn.functional.pad(
                response_tokens, (0, self.pad_to_length - response_tokens.size(-1)), value=0
            )

        send_data = {
            "tokens": response_tokens.numpy(),
            "sequence_lengths": rollout_batch["response_lengths"].unsqueeze(1).cpu().numpy(),
        }

        critic_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.critic.name.infer, data=send_data
        )

        rm_future = None
        if not self.combine_rm_and_critic_server:
            rm_future = run_if_model_parallel_src(
                self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
            )

        return RMCriticFutureResult(critic_future, rm_future, self.combine_rm_and_critic_server, og_seq_length)

    def train(self, ppo_rollout_data):
        send_data = {}

        func = partial(
            gather_tensor,
            dst=parallel_state.get_data_parallel_src_rank(),
            group=parallel_state.get_data_parallel_group(),
        )

        send_data["tokens"] = func(ppo_rollout_data["response_tokens"], dtype=torch.int64)
        send_data["returns"] = func(ppo_rollout_data["returns"])
        send_data["prev_values"] = func(ppo_rollout_data["values"])
        send_data["mask"] = func(ppo_rollout_data["mask"])

        future = None
        if torch.distributed.get_rank() == 0:
            send_data = {k: torch.cat(v, dim=0).detach().cpu().numpy() for k, v in send_data.items()}
            future = self.communicator.send_data_to_server(
                server_name=self.cfg.critic.name.train, data=send_data, batching=False
            )

        return future

    def save(self):
        save_future = None
        if torch.distributed.get_rank() == 0:
            send_data = {"dummy_var": np.array([0])}
            save_future = self.communicator.send_data_to_server(
                server_name=self.cfg.critic.name.save, data=send_data, batching=False
            )

        return SaveFuture(save_future)
