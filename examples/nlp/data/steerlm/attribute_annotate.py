# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is for annotating attributes for a dataset by sending requests to a regression reward model server. 
"""


import argparse
import json
import os
from typing import List

import jsonlines
import numpy as np
from common import (
    ALL_STEERLM_ATTRIBUTES,
    ASSISTANT_TURN_TEMPLATE,
    LABEL_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TURN_TEMPLATE,
)
from pytriton.client import FuturesModelClient
from tqdm import tqdm, trange


def _str_list2numpy(str_list: List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model_name", type=str, default="reward_model")
    parser.add_argument("--add-eos", action="store_true")
    return parser.parse_args()


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
        reward_out = output_dict["rewards"].flatten().tolist()

        all_rewards.append(reward_out)
        all_exceeded += output_dict["exceeded"].tolist()

    return all_rewards, all_exceeded


def get_key(l):
    convs = [c["value"] for c in l["conversations"]]
    return "".join(convs)


def main(args):
    inference_output = args.output_file

    exist = set()
    if os.path.exists(inference_output):
        with jsonlines.open(inference_output) as reader:
            for obj in tqdm(reader):
                exist.add(get_key(obj))

    fout = open(inference_output, "a", encoding="utf-8")

    # to warm up the jit
    _ = get_reward(["hello world!"], add_EOS=args.add_eos, host=args.host, port=args.port, model_name=args.model_name)

    all_samples, inputs = [], []

    with jsonlines.open(args.input_file) as reader:
        for obj in tqdm(reader):
            if get_key(obj) in exist:
                continue
            user = obj["mask"]
            turns = []
            text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
            for turn in obj["conversations"]:
                value = turn["value"]
                if turn["from"] == user:
                    text += USER_TURN_TEMPLATE.format(value=value)
                else:
                    text += ASSISTANT_TURN_TEMPLATE.format(value=value)
                if "label" in turn and turn["label"] is not None:
                    out_text = text + LABEL_PREFIX
                    turns.append(out_text)

            all_samples.append(turns)
            inputs.append(obj)

    print(f"exist {len(exist)}, rest {len(inputs)}")
    if len(inputs) == 0:
        exit(0)

    for idx in trange(0, len(all_samples)):
        input = inputs[idx]
        sample = all_samples[idx]
        rewards_all, _ = get_reward(
            sample, add_EOS=args.add_eos, host=args.host, port=args.port, model_name=args.model_name
        )

        t = 0
        for turn in input["conversations"]:
            if "label" in turn and turn["label"] is not None:
                reward = rewards_all[t]
                t += 1

                reward_each = [min(4.0, max(0.0, float(r))) for r in reward]
                reward_each = [round(r) for r in reward_each]

                reward_string = ",".join(f"{a}:{r}" for a, r in zip(ALL_STEERLM_ATTRIBUTES, reward_each))
                turn["label"] = reward_string

        assert t == len(rewards_all)

        fout.write(json.dumps(input) + "\n")

    print("all annotations finished")
    fout.close()


if __name__ == "__main__":
    main(prepare_args())
