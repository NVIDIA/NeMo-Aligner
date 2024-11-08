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
This script is to preprocess HelpSteer2 dataset from HuggingFace format into Attribute-conditioned SFT training format.
"""

import argparse
import json
import os

from common import ALL_STEERLM_ATTRIBUTES, SYSTEM_PROMPT
from datasets import load_dataset


def download_helpsteer2():
    ds = load_dataset("nvidia/HelpSteer2")
    train = ds["train"]
    val = ds["validation"]
    return train, val


def download_helpsteer2_preference():
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference")["train"]
    train = []
    val = []

    for dp in ds:
        new_dp1 = {"prompt": dp["prompt"], "response": dp["response_1"], "quality": dp["preference_strength"]}

        new_dp2 = {"prompt": dp["prompt"], "response": dp["response_2"], "quality": dp["preference_strength"]}

        if dp["split"] == "train":
            train.append(new_dp1)
            train.append(new_dp2)
        else:
            val.append(new_dp1)
            val.append(new_dp2)

    return train, val


def format_label(dp, only_helpfulness=False):
    label_list = []
    for attr in ALL_STEERLM_ATTRIBUTES:
        if attr in dp:
            if only_helpfulness and attr != "helpfulness":
                continue
            label_list.append(f"{attr}:{dp[attr]}")
    return ",".join(label_list)


def process_dataset(data, only_helpfulness=False):
    output = []
    for dp in data:
        conversation_obj = {}
        conversation_obj["conversations"] = [
            {"value": dp["prompt"], "from": "User", "label": None},
            {
                "value": dp["response"],
                "from": "Assistant",
                "label": format_label(dp, only_helpfulness=only_helpfulness),
            },
        ]
        conversation_obj["system"] = SYSTEM_PROMPT
        conversation_obj["mask"] = "User"
        conversation_obj["type"] = "VALUE_TO_TEXT"
        output.append(conversation_obj)
    return output


def main(output_dir, preference=False, only_helpfulness=False):
    if preference:
        train, val = download_helpsteer2_preference()
    else:
        train, val = download_helpsteer2()

    os.makedirs(output_dir, exist_ok=True)
    processed_train = process_dataset(train, only_helpfulness=only_helpfulness)

    with open(f"{output_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for record in processed_train:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    processed_val = process_dataset(val, only_helpfulness=only_helpfulness)
    with open(f"{output_dir}/val.jsonl", "w", encoding="utf-8") as f:
        for record in processed_val:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dir",
        "--output_directory",
        required=True,
        help="folder to store the created train.jsonl and val.jsonl; will be created if it does not exist",
    )

    parser.add_argument(
        "-oh", "--only_helpfulness", action="store_true", help="Use only the Helpfulness attribute",
    )

    parser.add_argument(
        "-pref",
        "--preference",
        action="store_true",
        help="Use HelpSteer2-preference meant for Bradley-Terry reward modelling instead of regular HelpSteer2",
    )
    args = parser.parse_args()

    main(args.output_directory, preference=args.preference, only_helpfulness=args.only_helpfulness)
