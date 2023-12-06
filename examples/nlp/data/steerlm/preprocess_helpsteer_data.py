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
This script is to preprocess HelpSteer dataset from HuggingFace format into Attribute-conditioned SFT training format.
"""

import argparse
import json
import os

from common import HELPSTEER_ATTRIBUTES, SYSTEM_PROMPT
from datasets import load_dataset


def download_helpsteer():
    ds = load_dataset("nvidia/HelpSteer")
    train = ds["train"]
    val = ds["validation"]
    return train, val


def format_label(dp):
    label_list = []
    for attr in HELPSTEER_ATTRIBUTES:
        label_list.append(f"{attr}:{dp[attr]}")
    return ",".join(label_list)


def process_dataset(data):
    output = []
    for dp in data:
        conversation_obj = {}
        conversation_obj["conversations"] = [
            {"value": dp["prompt"], "from": "User", "label": None},
            {"value": dp["response"], "from": "Assistant", "label": format_label(dp)},
        ]
        conversation_obj["system"] = SYSTEM_PROMPT
        conversation_obj["mask"] = "User"
        conversation_obj["type"] = "VALUE_TO_TEXT"
        output.append(conversation_obj)
    return output


def main(output_dir):
    train, val = download_helpsteer()

    os.makedirs(output_dir, exist_ok=True)
    processed_train = process_dataset(train)
    with open(f"{output_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for record in processed_train:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    processed_val = process_dataset(val)
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
    args = parser.parse_args()

    main(args.output_directory)
