import re

import pandas as pd
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer, math_equal
from nemo_skills.code_execution.sandbox import LocalSandbox


class Feedback(object):
    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError


class GSK8KFeedbackDataset(Feedback):
    def score(self, response, answer):
        """
        score the response
        """
        # this needs to be on a seperate server for anything
        # complicated but for GSM8K this is fine
        response = extract_answer(response)
        return float(math_equal(response, answer))


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


class MathSandBoxedFeedBack:
    def __init__(self, host, port, test_on_init=True):
        self.sandbox = LocalSandbox(host=host, port=port)

        if test_on_init:
            assert self.sandbox.is_output_correct("123", 123), "sandbox output should be correct! on 123 string vs 123"
            assert self.sandbox.is_output_correct("\\frac{1}{4}", "\\frac{2}{8}"), "sandbox should reduce fractions!"

    def score(self, response, answer):
        # NOTE: response must be in boxed format
        response = extract_answer(response)

        return self.sandbox.is_output_correct(response, answer)
