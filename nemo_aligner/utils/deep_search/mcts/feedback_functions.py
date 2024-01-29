import re

import pandas as pd


class Feedback(object):
    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError

class GSK8KFeedbackDataset(Feedback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.dataset['answer'][data_id].lower().split("####")[1].strip().replace(',', '')

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

    def __init__(self, gsk8k_path='train-00000-of-00001.parquet'):
        super().__init__()
        self.gsk8k_path = gsk8k_path
        self.gsk8k = pd.read_parquet(self.gsk8k_path)
#        new_row= {'question': "what is 3+2?", "answer": "3+2=5 #### 5"}
#        self.gsk8k.loc[0] = new_row

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.gsk8k.iloc[data_id]['answer'].lower().split('####')[1].strip().replace(',', '')
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
