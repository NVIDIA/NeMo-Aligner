import base64
import json

import numpy as np
import sentencepiece
import torch
from pytriton.client import ModelClient

from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""


def encode_context_data(context_data):
    context_data = [np.array(t, dtype=np.int32).tostring() for t in context_data]
    str_context = [base64.b64encode(t).decode() for t in context_data]
    str_ndarray = np.array(str_context)[..., np.newaxis]
    context_data = np.char.encode(str_ndarray, "utf-8")
    return context_data


tokenizer = sentencepiece.SentencePieceProcessor(
    "/datasets/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model"
)
### first evaluation to construct the root node and its children nodes
input_text = ["how are you?", "how large is the universe?"]


# ### sub sequent evaluation to construct the children node

max_depth = 500
termination_condition = TerminationCondition(max_depth, end_strings=["<extra_id_1>"])
score_fun = GSK8KFeedbackHF()
total_data = 500

batch_size = 2

for data_id in range(0, total_data, batch_size):

    input_text = []
    for i in range(batch_size):
        question = score_fun.ds['train'][data_id + i]["question"]
        input_text.append(steerlm_template.format(prompt=question))

    str_ndarray = np.array(input_text)[..., np.newaxis]
    input1_data = np.char.encode(str_ndarray, "utf-8")
    context_data = [tokenizer.encode(t) for t in input_text]
    encoded_context_data = encode_context_data(context_data)

    with ModelClient("localhost:2323", "search") as client:
        result_dict = client.infer_batch(
            sentences=input1_data, context_ids=encoded_context_data, parameters={"session": "test"}
        )
        actions = result_dict["action"]  # [batch, top_k]
        policy = result_dict["policy"]  # [batch, top_k]

    # select the tokens based on policy probablity
    tokens = [tokenizer.EncodeAsIds(i.item()) for i in str_ndarray]

    done = [False] * batch_size
    for depth in range(max_depth):
        if depth % 100 == 0:
            print(depth, "out of", max_depth)

        actions = actions[:, 0:1]
        for j, is_done in zip(range(len(actions)), done):
            if is_done:
                continue
            tokens[j].append(actions[j].item())
        #
        with ModelClient("localhost:2323", "search") as client:
            result_dict = client.infer_batch(
                action=actions, context_ids=encoded_context_data, parameters={"session": "test"}
            )

        # append actions to context_ids
        for b in range(len(actions)):
            context_data[b].append(actions[b][0].item())

        encoded_context_data = encode_context_data(context_data)

        actions = result_dict["action"]
        policy = result_dict["policy"]  # [batch, top_k]

        decoded = [tokenizer.DecodeIds(i) for i in tokens]

        for j in range(len(actions)):
            if termination_condition(decoded[j], depth):
                done[j] = True

        if all(done):
            break

    # save decoded to jsonl file
    for i in range(batch_size):
        answer = score_fun.ds['train'][data_id + i]["answer"]
        output = {}
        output["question"] = input_text[i]
        output["answer"] = answer
        output["decoded"] = decoded[i]
        output["data_id"] = data_id + i
        with open("output.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
