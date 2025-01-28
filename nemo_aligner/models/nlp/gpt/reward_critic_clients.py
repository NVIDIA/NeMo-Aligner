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
from collections import Counter

# List of packages to install
packages = ["absl-py", "langdetect", "nltk==3.8.1", "immutabledict"]
# Run pip install for each package
for package in packages:
    subprocess.run(["python", "-m", "pip", "install", package])

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig

from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, gather_tensor, run_if_model_parallel_src
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.utils.verifiers.code_verifier_unsafe import CodeVerifier
from nemo_aligner.utils.verifiers.instruction_following.instructions_registry import INSTRUCTION_DICT
from nemo_aligner.utils.verifiers.math_grader import extract_answer, math_equal

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


def MATH_rewards(response, args):

    ans = args["answer"]
    correctness = -10
    prediction = None
    try:
        prediction = extract_answer(response)
        correctness = int(math_equal(str(prediction), str(ans))) or int(math_equal(prediction, ans))
        correctness = -10 if correctness == 0 else 5
        print(f"prediction: {prediction}, answer: {ans}, correctness: {correctness}")
    except Exception as e:
        print(f"Error in math_rewards: {e}, task: {args}")
        correctness = -10

    return correctness, prediction, ans


def instruction_following_rewards(prompt, response, args):
    """Tests response to see if instrutions are followed."""
    try:
        task_args = args
        instruction_list = task_args["instruction_id_list"]
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            try:
                instruction_cls = INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                kwargs = (
                    task_args["instruction_kwargs"][index]
                    if task_args["instruction_kwargs"][index] is not None
                    else {}
                )
                instruction.build_description(**kwargs)
                instruction_args = instruction.get_instruction_args()
                if instruction_args and "prompt" in instruction_args:
                    instruction.build_description(prompt=prompt)

                if response.strip() and instruction.check_following(response):
                    is_following_list.append(True)
                else:
                    is_following_list.append(False)
            except Exception as e:
                print(f"Error in instruction_following_rewards: {e}, task: {args}")

        low, high = -10, 5
        correctness = sum(is_following_list) / len(is_following_list)
        score = low + (high - low) * correctness
        return score, True
    except Exception as e:
        print(f"Error in instruction_following_rewards: {e}")
        return 0, False


def game24_rewards(response, args):
    inputs = args["input"]
    try:
        prediction = extract_answer(response)
        prediction = prediction.replace("\\times", "*").replace("\\div", "/")
        numbers1 = Counter(re.findall(r"\d+", str(prediction)))
        numbers2 = Counter(re.findall(r"\d+", inputs))
        if numbers1 != numbers2:
            return -10, True
        correctness = int(eval(prediction) == 24) or int(math_equal(str(prediction), "24"))
        score = -10 if correctness == 0 else 5
        return score, True
    except Exception as e:
        print(f"Error in game24_rewards: {e}, task: {args}")
        return -10, False


def parentheses_rewards(response, args):
    input_left = args["input_left"]
    answer = args["answer"]
    try:
        prediction = extract_answer(response)
        prediction = prediction.replace("\\times", "*").replace("\\div", "/")
        if str(prediction).count("(") != 1 or str(prediction).count(")") != 1:
            return -10, True
        if str(prediction).replace("(", "").replace(")", "").replace(" ", "") != input_left.replace(" ", ""):
            return -10, True
        correctness = int(eval(prediction) == eval(answer)) or int(math_equal(str(prediction), str(answer)))
        score = -10 if correctness == 0 else 5
        return score, True
    except Exception as e:
        print(f"Error in parentheses_rewards: {e}, task: {args}")
        return -10, False


def code_rewards(response, args):
    try:
        code_str = response.replace("```python", "```").split("```")[1].strip()
        tests = args["unittests"]
        verifier = CodeVerifier()
        if args.get("verifier_type", None) == "code_verifier_assertion":
            results = verifier.verify_assertions(code_str, tests)
        else:
            results = verifier.verify(code_str, tests, args.get("fn_name", None))
        num_pass = 0
        for result in results:
            test_num = result["test_number"]
            if result.get("passed"):
                print(f"Test #{test_num}: PASS")
                num_pass += 1
            else:
                print(f"Test #{test_num}: FAIL")
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("Expected:")
                    print(result["expected_output"])
                    print("Got:")
                    print(result["actual_output"])

        pass_rate = num_pass / len(results)
        print(f"pass rate: {pass_rate}")
        low, high = -10, 5
        score = low + (high - low) * pass_rate
        return score, True
    except Exception as e:
        print(f"Error in code_rewards: {e}, task: {args}")
        return -10, True


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

        subprocess.run(["python", "-m", "pip", "install", "antlr4-python3-runtime==4.11.0"])

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


# class RMFutureResult(FutureResult):
#     def __init__(self, rm_future):
#         self.rm_future = rm_future

#     def result(self):
#         rewards = get_future_result(self.rm_future, "rewards")

#         self.rm_future = None
#         return rewards.flatten()


class RMFutureResult(FutureResult):
    def __init__(self, rm_future, verifier_rm_future=None):
        self.rm_future = rm_future
        self.verifier_rm_future = verifier_rm_future

    def result(self):
        rewards = get_future_result(self.rm_future, "rewards")

        self.rm_future = None
        out = rewards.flatten()
        print(rewards, self.verifier_rm_future)
        # If verifier_rm_future exists, combine values based on conditions
        if self.verifier_rm_future is not None:
            verifier_rewards = self.verifier_rm_future.flatten()
            out = out + verifier_rewards

        return out


@dataclass
class RemoteGPTRMClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        server_dict = {cfg.reward_model.name: (cfg.reward_model.ip, cfg.reward_model.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.pad_to_length = self.cfg.pad_to_length
        subprocess.run(["python", "-m", "pip", "install", "antlr4-python3-runtime==4.11.0"])

    def infer_rm_critic(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        verifier_rewards = []
        texts = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            text = model.tokenizer.ids_to_text(
                rollout_batch["response_tokens"][i, : rollout_batch["response_lengths"][i]].tolist()
            )
            format_correct = True

            # user_text, assistant_text = extract_dialogue_llama(text + "<|start_header_id|>")

            # text = chat_template(user_text=user_text, assistant_text=assistant_text, template="HS2")
            texts.append(text)
            print(text)

            if args[i] is not None:
                if "code" in args[i]["task"]:
                    score, success = code_rewards(assistant_text[-1], args[i])
                    print(f"check done: {success}, score: {score}, args: {args[i]}")
                elif args[i]["task"] == "instruction_following":
                    score, success = instruction_following_rewards(user_text[-1], assistant_text[-1], args[i])
                    print(f"check done: {success}, score: {score}, args: {args[i]}")
                elif args[i]["task"] == "reasoning_game24":
                    score, success = game24_rewards(assistant_text[-1], args[i])
                    print(f"check done: {success}, score: {score}, args: {args[i]}")
                elif args[i]["task"] == "reasoning_parentheses":
                    score, success = parentheses_rewards(assistant_text[-1], args[i])
                    print(f"check done: {success}, score: {score}, args: {args[i]}")
                else:
                    score, prediction, answer = MATH_rewards(assistant_text[-1], args[i])
                    print(f"prediction: {prediction}, answer: {answer}, score: {score}")

                score = score if format_correct else -100
                verifier_rewards.append(score)
                print("task: ", args[i]["task"], "score: ", score)
            else:
                score = 0 if format_correct else -100
                verifier_rewards.append(score)

        verifier_rewards = torch.tensor(verifier_rewards, device=rollout_batch["response_tokens"].device).float()

        send_data = {
            "sentences": _str_list2numpy(texts),
        }

        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data,
        )

        return RMFutureResult(rm_future, verifier_rewards)
