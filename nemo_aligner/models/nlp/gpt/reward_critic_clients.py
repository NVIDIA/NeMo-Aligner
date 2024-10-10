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

import re
import subprocess
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig

from nemo_aligner.models.nlp.gpt.math_grader import extract_answer, math_equal
from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, gather_tensor, run_if_model_parallel_src
from nemo_aligner.utils.server_utils import FutureResult

"""A remote client that acts like a real Reward Model and Critic forwards all requests from the actor
    over to the remote PyTrition server
"""


class HelpsteerTemplate:
    def get_first_turn_template(self, text):
        return f"""<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User\n{text}"""

    def get_assistant_turn_template(self, text):
        return f"""\n<extra_id_1>Assistant\n{text}"""

    def get_user_turn_template(self, text):
        return f"""\n<extra_id_1>User\n{text}"""

    def add_ending(self, text):
        return f"""{text}\n<extra_id_2>"""


def chat_template(user_text, assistant_text, template):
    formatter = HelpsteerTemplate()

    text = ""
    for i in range(len(user_text)):
        if i == 0:
            text += formatter.get_first_turn_template(user_text[i])
        else:
            text += formatter.get_user_turn_template(user_text[i])
        text += formatter.get_assistant_turn_template(assistant_text[i])
    text = formatter.add_ending(text)
    return text


def extract_dialogue_helpsteer(text):
    user_pattern = r"<extra_id_1>User\n(.*?)\n<extra_id_1>"
    assistant_pattern = r"<extra_id_1>Assistant\n(.*?)\n(?:<extra_id_1>|<extra_id_2>)"

    user_text = re.findall(user_pattern, text, re.DOTALL)
    assistant_text = re.findall(assistant_pattern, text, re.DOTALL)

    return user_text, assistant_text


def extract_dialogue_llama(text):
    user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"
    assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>"

    user_text = re.findall(user_pattern, text, re.DOTALL)
    assistant_text = re.findall(assistant_pattern, text, re.DOTALL)

    return user_text, assistant_text


def _str_list2numpy(str_list) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


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

        ten = broadcast_2d_tensor_within_mp(result)

        results.append(ten)

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
        return rewards.flatten(), values


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
            self.communicator.send_data_to_server, server_name=self.cfg.critic.name.infer, data=send_data,
        )

        rm_future = None
        if not self.combine_rm_and_critic_server:
            rm_future = run_if_model_parallel_src(
                self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data,
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
        send_data["returns"] = func(ppo_rollout_data["returns"], dtype=torch.float32)
        send_data["prev_values"] = func(ppo_rollout_data["values"], dtype=torch.float32)
        send_data["mask"] = func(ppo_rollout_data["mask"], dtype=torch.float32)

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


class RMFutureResult(FutureResult):
    def __init__(self, rm_future):
        self.rm_future = rm_future

    def result(self):
        rewards = get_future_result(self.rm_future, "rewards")

        self.rm_future = None
        return rewards.flatten()


class MathFutureResult(FutureResult):
    def __init__(self, rm_future):
        self.rm_future = rm_future

    def result(self):
        return self.rm_future


@dataclass
class RemoteGPTRMClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        server_dict = {cfg.reward_model.name: (cfg.reward_model.ip, cfg.reward_model.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.pad_to_length = self.cfg.pad_to_length

    def infer_rm_critic(self, rollout_batch, model):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        texts = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            text = model.tokenizer.ids_to_text(
                rollout_batch["response_tokens"][i, : rollout_batch["response_lengths"][i]].tolist()
            )
            user_text, assistant_text = extract_dialogue_llama(text + "<|start_header_id|>")
            print(text + "<|start_header_id|>")
            print("--" * 80)
            print("USER TEXT", user_text)
            print("ASSISTANT_TEXT", assistant_text)
            text = chat_template(user_text=user_text, assistant_text=assistant_text, template="HS2")
            print("**" * 80)
            print(text)
            print("0O0" * 60)
            texts.append(text)

        send_data = {
            "sentences": _str_list2numpy(texts),
        }

        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data,
        )

        return RMFutureResult(rm_future)


@dataclass
class RemoteGPTMathClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

    def gsm8k_rewards(self, prompt, response, args):
        ans = args["answer"]
        pattern = r"-?\$?\d[\d,]*\.?\d*|-?\.\d+"
        matches = re.findall(pattern, response)
        # print(prompt, response, matches, ans)
        if matches:
            try:
                prediction = float(matches[-1].replace("$", "").replace(",", ""))
                return int(prediction == ans)
            except:
                return 0
        else:
            return 0

    def MATH_rewards(self, prompt, response, args):
        subprocess.run(["python", "-m", "pip", "install", "antlr4-python3-runtime==4.11.0"])
        ans = args["answer"]
        correctness = 0
        try:
            prediction = extract_answer(response)
            correctness = math_equal(prediction, ans)
            print(f"prediction: {prediction}, answer: {answer}, correctness: {correctness}")
        except:
            pass

        subprocess.run(["python", "-m", "pip", "install", "antlr4-python3-runtime==4.9.3"])
        return correctness

    def infer_rm_critic(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        rewards = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            prompt = model.tokenizer.ids_to_text(
                rollout_batch["response_tokens"][i, : rollout_batch["prompt_lengths"][i]].tolist()
            )
            response = model.tokenizer.ids_to_text(
                rollout_batch["response_tokens"][
                    i, rollout_batch["prompt_lengths"][i] : rollout_batch["response_lengths"][i]
                ].tolist()
            )
            print(response, model.cfg.reinforce.sampling_params.end_strings)
            print(args)
            for end_string in model.cfg.reinforce.sampling_params.end_strings:
                response = response.replace(end_string, "")
            rewards.append(self.MATH_rewards(prompt, response, args[i]))

        rewards = torch.tensor(rewards, device=rollout_batch["response_tokens"].device).float()

        return MathFutureResult(rewards)
