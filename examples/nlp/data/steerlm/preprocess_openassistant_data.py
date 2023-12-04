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
This script is to preprocess OpenAssistant dataset from HuggingFace format into Attribute-conditioned SFT training format.
"""

import argparse
import gzip
import json
import os
import random

import requests
from common import OPEN_ASSISTANT_ATTRIBUTES, SYSTEM_PROMPT

likert_scale = 5


def encode_labels(labels):
    items = []
    for key in OPEN_ASSISTANT_ATTRIBUTES:
        if key in labels:
            value = labels[key]["value"]
            items.append(f"{key}:{round(value*(likert_scale-1))}")
    return ",".join(items)


def parse_conversations(tree_obj):
    """ recusive function that returns all the sub converstaions in a list starting from node tree_obj

    Args:
        tree_obj (obj): current conversation node

    Returns:
        a list of sub conversation threads including the current conversation node
    """
    if "prompt" in tree_obj:
        prompt_obj = tree_obj["prompt"]
    elif "text" in tree_obj and "role" in tree_obj:
        prompt_obj = tree_obj
    else:
        return [[]]

    if prompt_obj["role"] == "prompter":
        role = "User"
    elif prompt_obj["role"] == "assistant":
        role = "Assistant"
    else:
        raise ValueError(f'unknown role {prompt_obj["role"]}')

    turn = {"value": prompt_obj["text"], "from": role}

    if "labels" in prompt_obj:
        turn["label"] = encode_labels(prompt_obj["labels"])
    all_conversations = []
    multiple_sub_threads = []
    for next_obj in prompt_obj["replies"]:
        multiple_threads = parse_conversations(next_obj)
        multiple_sub_threads.extend(multiple_threads)
    if len(multiple_sub_threads) != 0:
        for sub_thread in multiple_sub_threads:
            all_conversations.append([turn] + sub_thread)
    else:
        all_conversations.append([turn])
    return all_conversations


def get_data_records(objs, mask_role, type):
    output = []
    for obj in objs:
        multi_conversations = parse_conversations(obj)
        for conversations in multi_conversations:
            if len(conversations) <= 1:
                # remove single turn conversations
                continue

            # mask out labels from user turns
            updated_conversation = []
            for turn in conversations:
                if turn["from"] == "User":
                    turn["label"] = None
                updated_conversation.append(turn)

            conversation_obj = {
                "conversations": updated_conversation,
                "system": SYSTEM_PROMPT,
                "mask": mask_role,
                "type": type,
            }
            output.append(conversation_obj)
    return output


def download_open_assistant(output_directory):
    filename = f"{output_directory}/2023-04-12_oasst_all.trees.jsonl.gz"

    # only download if doesn't exist
    if not os.path.isfile(filename):
        url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_all.trees.jsonl.gz"
        response = requests.get(url)
        with open(filename, mode="wb") as fw:
            fw.write(response.content)

    with gzip.open(filename) as f:
        file_content = f.readlines()

    data = [json.loads(dp.decode("utf-8")) for dp in file_content]
    return data


def main(output_directory, proportion_of_train=0.95, seed=10):
    os.makedirs(args.output_directory, exist_ok=True)
    all_objs = download_open_assistant(output_directory)

    # Note that we manually shuffle and split the dataset into train / valid sets as we do not use
    # the official train / valid splits from Hugging Face. This is because we use the full dataset that
    # also includes low-quality data (since SteerLM can still learn from such data), instead of
    # the smaller "ready for export" dataset.
    random.seed(seed)
    random.shuffle(all_objs)

    train_num = int(len(all_objs) * proportion_of_train)
    train_objs = all_objs[:train_num]
    val_objs = all_objs[train_num:]
    train_records = get_data_records(train_objs, "User", "VALUE_TO_TEXT")
    val_records = get_data_records(val_objs, "User", "VALUE_TO_TEXT")

    with open(f"{output_directory}/train.jsonl", "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(f"{output_directory}/val.jsonl", "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dir",
        "--output_directory",
        required=True,
        help="folder to store the created train.jsonl and val.jsonl; will be created if not exist",
    )
    args = parser.parse_args()
    main(args.output_directory)
