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
threshold = 0.01
end_word = "<extra_id_1>"
max_tokens = 400


def adjust_actions(output, threshold=0.1):
    probablities = output["policy"]
    actions = output["action"]
    update_probablities = []
    update_actions = []
    for prob, one_actions in zip(probablities, actions):
        selected = prob >= threshold
        if sum(selected) > 0:
            # not empty
            select_prob = prob[selected]
            select_action = one_actions[selected].tolist()
            update_probablities.append(select_prob)
            update_actions.append(select_action)
        else:
            # if all the probablities are less than the threshold
            # use all the probablities
            update_probablities.append(prob)
            update_actions.append(one_actions.tolist())
    output["policy"] = update_probablities
    output["action"] = update_actions
    return output


def get_prompt_only(record):
    prompt_only = copy.deepcopy(record)
    prompt_only["conversations"] = prompt_only["conversations"][:-1]
    prompt_only["conversations"].append({"from": "Assistant", "value": ""})
    return prompt_only


### first evaluation to construct the root node and its children nodes
# input_text = ['how are you?']
# input_text = ['how are you?', 'how large is the universe?', 'what is your name', 'what is the time now?'] * 4
# input_text = ['how are you? I am']
record0 = {
    "conversations": [
        {"from": "User", "value": "how are you?"},
        {"from": "Assistant", "value": "I am fine, thank you."},
        {"from": "User", "value": "why the sky is blue?"},
        {"from": "Assistant", "value": ""},
    ]
}
record1 = {
    "conversations": [
        {"from": "User", "value": "how are you?"},
        {"from": "Assistant", "value": "I am fine, thank you."},
        {"from": "User", "value": "why the ocean is blue?"},
        {"from": "Assistant", "value": ""},
    ]
}

print(template.render(record=record0))
prompt_only0 = get_prompt_only(record0)
prompt_only1 = get_prompt_only(record1)
print(template.render(record=prompt_only0))
prompts = [prompt_only0, prompt_only1]
input_text = [template.render(record=record0), template.render(record=record1)]
prompt_text = [template.render(record=prompt_only0), template.render(record=prompt_only1)]

# prompts = prompts[0:1]
# input_text = input_text[0:1]
# prompt_text = prompt_text[0:1]

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
    result_dict = adjust_actions(result_dict, threshold=threshold)
    actions = result_dict["action"]  # [batch, top_k]
    policy = result_dict["policy"]  # [batch, top_k]

# select the tokens based on policy probablity
tokens = [tokenizer.EncodeAsIds(i.item()) for i in str_ndarray]
batch_size = len(actions)
finished = [False] * batch_size

for i in range(max_tokens):
    # ### sub sequent evaluation to construct the children node

    if i % 10 == 0:
        print(i, "out of", 50)

    value_records = []
    count = 0
    for bid in range(batch_size):
        if finished[bid]:
            continue
        one_action = actions[count]
        prompt_len = len(prompt_text[bid])
        print("bid", bid, "action len", len(one_action))
        for token in one_action:
            new_seq = tokens[bid] + [token]
            decoded = tokenizer.DecodeIds(new_seq)
            response = decoded[prompt_len:]
            record = {"conversations": prompts[bid]["conversations"][:-1], "response": response}
            text = process_sample(record)
            value_records.append(text)
        count += 1
    rewards = get_reward(value_records, port=1424)

    best_actions = []
    beg_index = 0
    for bid in range(len(actions)):
        top_k = len(actions[bid])
        end_index = beg_index + top_k
        sub_rewards = rewards[beg_index:end_index]
        beg_index = end_index
        sub_rewards = [get_combined_reward(r) for r in sub_rewards]
        # get the best action
        best_action = np.argmax(sub_rewards)
        print("batch id", bid, "best reward", sub_rewards[best_action])
        best_actions.append(actions[bid][best_action])

    decoded = []
    count = 0
    next_actions = []
    for j in range(batch_size):
        if finished[j]:
            continue
        tokens[j].append(best_actions[count])
        text = tokenizer.DecodeIds(tokens[j])
        decoded.append(text)
        if text.endswith(end_word):
            finished[j] = True
        else:
            next_actions.append(best_actions[count])
        prompt_str = prompt_text[j]
        print("step", i, bid, text[len(prompt_str) :])
        count += 1

    if len(next_actions) == 0:
        break

    # for bid in range(len(decoded)):
    #     prompt_str = prompt_text[bid]
    #     print('step', i, bid, decoded[bid][len(prompt_str):])
    #
    with ModelClient("localhost:2323", "search") as client:
        result_dict = client.infer_batch(
            action=np.array(next_actions).reshape(len(next_actions), -1).astype(np.int32),
            context_ids=encoded_context_data,
            parameters={"session": "test"},
        )
        result_dict = adjust_actions(result_dict, threshold=threshold)

    # append actions to context_ids
    count = 0
    next_context_data = []
    for b in range(batch_size):
        if finished[b]:
            continue
        context_data[b].append(best_actions[count])
        next_context_data.append(context_data[b])
        count += 1

    encoded_context_data = encode_context_data(next_context_data)

    actions = result_dict["action"]
    policy = result_dict["policy"]  # [batch, top_k]


for i in range(len(tokens)):
    print(tokens[i], len(tokens[i]))

decoded = [tokenizer.DecodeIds(i) for i in tokens]

for i in range(len(decoded)):
    prompt_str = prompt_text[i]
    print(i, decoded[i][len(prompt_str) :])
