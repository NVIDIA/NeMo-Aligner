import json
import os
import re

import pandas as pd
import requests
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import LocalSandbox

from nemo_aligner.data.nlp.datasets import TEMPLATES
from nemo_aligner.utils.deep_search.mcts.reward_functions import get_reward


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
        assert self.ds[data_id]["data_id"] == data_id
        response = response.lower()
        answer = self.ds[data_id]["expected_answer"]
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


class RegressionRMFeedback(Feedback):
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
        # # get the expected answer, e.g. 'quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2'
        # attribute_str = response.split("<extra_id_2>")[-1].split("\n")[0]
        # # extract the numbers
        # attributes = attribute_str.split(",")
        # numbers = [int(attr.split(":")[-1]) for attr in attributes]
        # # remove the <extra_id_2> line
        # response = "\n".join([i for i in response.split("\n") if not i.startswith("<extra_id_2>")])
        response = response + "<extra_id_2>"
        try:
            evaluate = get_reward([response], False, self.host, self.port)[0]

            print(evaluate)
            score = (evaluate[0] - 1) / 4  # score from 1 to 5
            # compute the distance between the two vectors
            # distance = sum([int(bool(a - b)) for a, b in zip(numbers, evaluate)])

            # normalize the distance to be between 0 and 1
            # distance = distance / (len(numbers))

            # score = 1 - distance
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score


class LLMJudgementFeedback(Feedback):
    def __init__(self, template="extra_sft_empty_sys"):
        # local_rank = os.getenv("local_rank", "0")
        self.host = os.getenv("JUDGE_SERVER_HOST", "localhost")
        self.port = os.getenv("JUDGE_SERVER_PORT", "1234")
        self.template = TEMPLATES[template]

    def score(self, response, data_id):
        """
        score the response
        """
        prompt_to_sent = self.format_prompt(response)
        print("prompt_to_sent", prompt_to_sent)
        try:
            eval_output = self.get_generation(
                prompt_to_sent, True, False, 1000, 1, 1.0, 1.0, 0, 1.0
            )  # get_reward([response], False, self.host, self.port)[0]
            if eval_output.find("<extra_id_0>") < 0:
                # hack due to the problem that huggingface's tokenizer strips out the <extra_id_x> token
                prompt_to_sent = (
                    prompt_to_sent.replace("<extra_id_0>", "").replace("<extra_id_1>", "").replace("<extra_id_2>", "")
                )
            evaluation = eval_output[len(prompt_to_sent) :]
            print("evaluation", evaluation)
            rating_matches = re.findall(r"\[\[(\d+)\]\]", evaluation)
            score = float(rating_matches[-1])  # Get the last match
            score = score / 5  # scale it to 0-1
            print("score", score)
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score

    def text_generation(self, data, ip, port):
        headers = {"Content-Type": "application/json"}
        resp = requests.put(f"http://{ip}:{port}/generate", data=json.dumps(data), headers=headers)
        return resp.json()

    def get_generation(self, prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition):
        data = {
            "sentences": [prompt],
            "tokens_to_generate": int(token_to_gen),
            "temperature": temp,
            "add_BOS": add_BOS,
            "top_k": top_k,
            "top_p": top_p,
            "greedy": greedy,
            "all_probs": False,
            "repetition_penalty": repetition,
            "min_tokens_to_generate": int(min_tokens),
            "end_strings": ["<|endoftext|>", "<extra_id_1>", "\x11"],
        }
        sentences = self.text_generation(data, ip=self.host, port=self.port)["sentences"]
        return sentences[0]

    def format_prompt(self, response):
        # remove the trailing extra_id_1
        if response.endswith("<extra_id_1>"):
            response = response[: -len("<extra_id_1>")]
        response = response[response.find("<extra_id_1>") :]
        response = response.replace("<extra_id_1>User", "### User:").replace("<extra_id_1>Assistant", "### Assistant:")
        mt_bench_multi_turn = f"""[Instruction]\nPlease act as an impartial judge and evaluate the quality of the responses provided by an AI assistant to the user's questions displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on all the assistant turns. Begin your evaluation by providing a short explanation. Be as objective and harsh as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format:\"[[rating]]\", for example: \"Rating: [[5]]\".
<|The Start of Assistant's Conversation with User|>
{response}
<|The End of Assistant's Conversation with User|>"""
        self_rewarding_prompt = """Given the conversation below, review the assistant's response to the user's question using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.
<|The Start of Assistant's Conversation with User|>
{response}
<|The End of Assistant's Conversation with User|>
After examining the assistant's response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: \"[[rating]]\", for example: \"Rating: [[5]]\""""
        formatted = self.template.format(prompt=self_rewarding_prompt)
        return formatted


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
