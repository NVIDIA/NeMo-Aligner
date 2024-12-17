import numpy as np
from typing import List
from nemo_aligner.servers.http_communicator import HTTPCommunicator

communicator = HTTPCommunicator.create_http_communicator_from_dict(
    {'math_grader': ('0.0.0.0', 5555)}
)

preds = ["\\boxed{.5}", "There is a lot of random reasoning done here, but eventually, the answer is \\boxed{123/456}", "\\boxed{15}"]
gt = ["1/2", "0.2697368421", "7"]

def triton_textencode(text_batch: List[str]):
    enc = np.array([[np.char.encode(i, 'utf-8')] for i in text_batch])
    enc = np.reshape(enc, (enc.shape[0], 1))

    return enc

data = {
    "pred_responses": triton_textencode(preds),
    "ground_truth": triton_textencode(gt),
}

future = communicator.send_data_to_server("math_grader", data)
print(future)
v = future.result()
print(v)