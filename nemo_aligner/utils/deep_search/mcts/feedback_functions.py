import pandas as pd
import re


class Feedback(object):

    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError


class GSK8KFeedback(Feedback):

    def __init__(self, gsk8k_path='train-00000-of-00001.parquet'):
        super().__init__()
        self.gsk8k_path = gsk8k_path
        self.gsk8k = pd.read_parquet(self.gsk8k_path)

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.gsk8k.iloc[data_id]['answer'].lower().split('####')[1].strip()
        # predicted answer matches the answer pattern
        numbers = re.findall(r'\{{([\d,]+)\}}', response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0