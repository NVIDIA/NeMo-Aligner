from typing import List

import numpy as np
from pytriton.client import FuturesModelClient

__all__ = ["get_reward"]

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_TEMPLATE = f"<extra_id_0>System\n{SYSTEM_PROMPT}\n"

USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

# had to delete the value here because we concat the prompts
ASSISTANT_TURN_TEMPLATE_FINAL = "<extra_id_1>Assistant\n"

ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"


# TODO: what to do with this?
LABEL_PREFIX = "<extra_id_2>"

OPEN_ASSISTANT_ATTRIBUTES = ["quality", "toxicity", "humor", "creativity"]

HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

ALL_STEERLM_ATTRIBUTES = OPEN_ASSISTANT_ATTRIBUTES + HELPSTEER_ATTRIBUTES


def process_sample(record):
    conversations = record["conversations"]
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)

    last_turn_is_user = False
    for turn in conversations:
        last_turn_is_user = False
        value = turn["value"]
        if turn["from"] in {"User", "user"}:
            text += USER_TURN_TEMPLATE.format(value=value)
            last_turn_is_user = True
        else:
            text += ASSISTANT_TURN_TEMPLATE.format(value=value)

    assert last_turn_is_user
    text += ASSISTANT_TURN_TEMPLATE_FINAL
    text += record["response"]
    return text


def _str_list2numpy(str_list: List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def get_reward(
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
        reward_out = output_dict["rewards"][0]
        reward_record = {key: value for key, value in zip(ALL_STEERLM_ATTRIBUTES, reward_out)}

        all_rewards.append(reward_record)
        all_exceeded += output_dict["exceeded"].tolist()
    return all_rewards


# data_record = {"conversations": [{"from": "User", "text": "how are you?"}], "response": ".asdf asf23r asdf"}
#
# text = process_sample(data_record)
# print(text)
# x = get_reward([text], port=1424)
# print(x)
