import os
import re
from typing import List

import pandas as pd
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import LocalSandbox
from pytriton.client import FuturesModelClient
import numpy as np

from nemo_aligner.utils.deep_search.mcts.reward_functions import _str_list2numpy, get_reward


class Feedback(object):
    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError


class DummyScore(Feedback):
    def score(self, response, data_id):
        return 0.0


class GSK8KFeedbackDataset(Feedback):
    def __init__(self, ds):
        self.ds = ds
        # local_rank = os.getenv("local_rank", "0")
        host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "localhost")
        port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "1034")
        self.sandbox = LocalSandbox(host=host, port=port)

    def score(self, response, data_id):
        """
        score the response
        """
        key = int(data_id.split("@")[0])
        assert self.ds[key]["data_id"] == key
        response = response.lower()
        answer = self.ds[key]["expected_answer"]
        # this needs to be on a seperate server for anything
        # complicated but for GSM8K this is fine
        response = extract_answer(response)
        try:
            score = float(self.sandbox.is_output_correct(response, answer))
        except Exception as e:
            print("############ Inference failed ############")
            print(answer, response)
            print(e)
            score = 0.0
        finally:
            return score


class SteerLMFeedback(Feedback):
    def __init__(self):
        # local_rank = os.getenv("local_rank", "0")
        self.host = os.getenv("REWARD_SERVER_HOST", "localhost")
        self.port = os.getenv("REWARD_SERVER_PORT", "1234")

    def score(self, response, data_id):
        """
        score the response
        """
        # remove the trailing extra_id_1
        if response.endswith("<extra_id_1>"):
            response = response[: -len("<extra_id_1>")]
        # get the expected answer, e.g. 'quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2'
        attribute_str = response.split("<extra_id_2>")[-1].split("\n")[0]
        # extract the numbers
        attributes = attribute_str.split(",")
        numbers = [int(attr.split(":")[-1]) for attr in attributes]
        # remove the <extra_id_2> line
        response = "\n".join([i for i in response.split("\n") if not i.startswith("<extra_id_2>")])
        response = response + "<extra_id_2>"
        try:
            evaluate = get_reward([response], False, self.host, self.port)[0]

            # compute the distance between the two vectors
            distance = sum([int(bool(a - b)) for a, b in zip(numbers, evaluate)])

            # normalize the distance to be between 0 and 1
            distance = distance / (len(numbers))

            score = 1 - distance
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score


def get_raw_reward(
    sentences: List[str], add_EOS=False, host="localhost", port=5555, model_name="reward_model",
):
    sentences = _str_list2numpy(sentences)

    futures = []

    with FuturesModelClient(f"{host}:{port}", model_name) as client:
        for sen in np.split(sentences, sentences.shape[0]):
            add_EOS_arr = np.ones_like(sen, dtype=bool) * add_EOS
            future = client.infer_batch(sentences=sen, add_EOS=add_EOS_arr)
            futures.append(future)

    all_result_dicts = [f.result() for f in futures]

    all_rewards, all_exceeded = [], []

    for output_dict in all_result_dicts:
        reward_out = output_dict["rewards"].flatten().tolist()

        all_rewards.append(reward_out)
        all_exceeded += output_dict["exceeded"].tolist()

    # return all_rewards, all_exceeded
    outputs = []
    for reward in all_rewards:
        outputs.append(reward)
    return outputs


class HelpSteerFeedback(Feedback):
    def __init__(self, weights):
        # local_rank = os.getenv("local_rank", "0")
        self.host = os.getenv("REWARD_SERVER_HOST", "localhost")
        self.port = os.getenv("REWARD_SERVER_PORT", "1234")
        self.weights = weights
        assert len(self.weights) == 9

    def score(self, response, data_id):
        """
        score the response
        """
        # remove the trailing extra_id_1
        if response.endswith("<extra_id_1>"):
            response = response[: -len("<extra_id_1>")]
        # # get the expected answer, e.g. 'quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2'
        # attribute_str = response.split("<extra_id_2>")[-1].split("\n")[0]
        # # extract the numbers
        # attributes = attribute_str.split(",")
        # numbers = [int(attr.split(":")[-1]) for attr in attributes]
        # remove the <extra_id_2> line
        if "<extra_id_2>" in response:
            response = "\n".join([i for i in response.split("\n") if not i.startswith("<extra_id_2>")])
        response = response + "<extra_id_2>"
        try:
            evaluate = get_raw_reward([response], False, self.host, self.port)[0]

            total = sum([w * e for w, e in zip(self.weights, evaluate)])

            max_value = sum([w * 4.0 for w in self.weights])

            score = total / max_value
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            # print stacktrace 
            import traceback
            traceback.print_exc()
            score = 0.0
        finally:
            return score


class GSK8KFeedback(Feedback):
    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedback(Feedback):
    def __init__(self):
        ...

    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedbackHF(Feedback):
    def __init__(self, split):
        super().__init__()
        self.ds = load_dataset("gsm8k", "main")
        self.split = split

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.ds[self.split][data_id]["answer"].lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0
