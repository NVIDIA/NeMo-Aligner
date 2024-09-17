import base64
import copy

import numpy as np
import sentencepiece
import torch
from jinja2 import Template
from pytriton.client import ModelClient

from test_reward_client import get_reward, process_sample

template = Template(
    """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
{% for item in record.conversations %}{% if item.from == 'User' %}
<extra_id_1>User
{{ item.value }}
{% elif item.from == 'Assistant' %}<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
{{ item.value }}{% endif %}{% endfor %}"""
)


def encode_context_data(context_data):
    context_data = [np.array(t, dtype=np.int32).tostring() for t in context_data]
    str_context = [base64.b64encode(t).decode() for t in context_data]
    str_ndarray = np.array(str_context)[..., np.newaxis]
    context_data = np.char.encode(str_ndarray, "utf-8")
    return context_data


# tokenizer = sentencepiece.SentencePieceProcessor('/dataset/models/unpack_2b_mcore/d153550ae65b4c6c94815752f654e0f5_2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model')
# tokenizer = sentencepiece.SentencePieceProcessor('/dataset/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model')
tokenizer = sentencepiece.SentencePieceProcessor(
    "/datasets/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model"
)
# tokenizer = sentencepiece.SentencePieceProcessor('/datasets/models/unpack_843m_mcore/adfd4c68d8444aa790c2e65eab362a9f_a184c0997f35446cac66e8e2d63f7853_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model')


### first evaluation to construct the root node and its children nodes
# input_text = ['how are you?']
# input_text = ['how are you?', 'how large is the universe?', 'what is your name', 'what is the time now?'] * 4
# input_text = ['how are you? I am']
record = {
    "conversations": [
        {"from": "User", "value": "how are you?"},
        {"from": "Assistant", "value": "I am fine, thank you."},
        {"from": "User", "value": "why the sky is blue?"},
        {"from": "Assistant", "value": ""},
    ]
}

print(template.render(record=record))
prompt_only = copy.deepcopy(record)
prompt_only["conversations"] = prompt_only["conversations"][:-1]
prompt_only["conversations"].append({"from": "Assistant", "value": ""})
print(template.render(record=prompt_only))
input_text = [template.render(record=record)] * 1
prompt_text = [template.render(record=prompt_only)] * 1

str_ndarray = np.array(input_text)[..., np.newaxis]
input1_data = np.char.encode(str_ndarray, "utf-8")

context_data = [tokenizer.encode(t) for t in input_text]

encoded_context_data = encode_context_data(context_data)


def get_combined_reward(reward):
    # just use helpfulness for now
    return reward["helpfulness"]


with ModelClient("localhost:2323", "search") as client:
    result_dict = client.infer_batch(
        sentences=input1_data, context_ids=encoded_context_data, parameters={"session": "test"}
    )
    actions = result_dict["action"]  # [batch, top_k]
    policy = result_dict["policy"]  # [batch, top_k]

# select the tokens based on policy probablity
tokens = [tokenizer.EncodeAsIds(i.item()) for i in str_ndarray]
# ### sub sequent evaluation to construct the children node
batch_size, top_k = actions.shape

for i in range(50):
    if i % 10 == 0:
        print(i, "out of", 50)

    value_records = []
    for bid, one_action in enumerate(actions):
        prompt_len = len(prompt_text[bid])
        for token in one_action:
            new_seq = tokens[bid] + [token.item()]
            decoded = tokenizer.DecodeIds(new_seq)
            response = decoded[prompt_len:]
            record = {"conversations": prompt_only["conversations"][:-1], "response": response}
            text = process_sample(record)
            value_records.append(text)
    rewards = get_reward(value_records, port=1424)

    best_actions = []
    for bid in range(batch_size):
        sub_rewards = rewards[bid * top_k : (bid + 1) * top_k]
        sub_rewards = [get_combined_reward(r) for r in sub_rewards]
        # get the best action
        best_action = np.argmax(sub_rewards)
        print("batch id", bid, "best reward", sub_rewards[best_action])
        best_actions.append(actions[bid][best_action].item())

    for j in range(batch_size):
        tokens[j].append(best_actions[j])

    decoded = [tokenizer.DecodeIds(i) for i in tokens]
    for bid in range(len(decoded)):
        prompt_str = prompt_text[bid]
        print("step", i, bid, decoded[bid][len(prompt_str) :])
    #
    with ModelClient("localhost:2323", "search") as client:
        result_dict = client.infer_batch(
            action=np.array(best_actions).reshape(batch_size, -1).astype(np.int32),
            context_ids=encoded_context_data,
            parameters={"session": "test"},
        )

    # append actions to context_ids
    for b in range(batch_size):
        context_data[b].append(best_actions[b])

    encoded_context_data = encode_context_data(context_data)

    actions = result_dict["action"]
    policy = result_dict["policy"]  # [batch, top_k]


for i in range(len(tokens)):
    print(tokens[i], len(tokens[i]))

decoded = [tokenizer.DecodeIds(i) for i in tokens]

for i in range(len(decoded)):
    prompt_str = prompt_text[i]
    print(i, decoded[i][len(prompt_str) :])
