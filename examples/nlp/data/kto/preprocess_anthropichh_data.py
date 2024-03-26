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

"""A script to process the Anthropic Dataset"""
import argparse
import json
import warnings
from pathlib import Path

from datasets import load_dataset


def prepare_args():
    parser = argparse.ArgumentParser(description="generate dataset")
    parser.add_argument(
        "--output-dir", type=str, default="./",
    )
    return parser.parse_args()


START_PROMPT_FORMAT = "Human:\n{body}\nAssistant:\n{response}"
PROMPT_CONTINUATION_FORMAT = "Human:\n{body}\nAssistant:\n{response}"


def process_hh(split):
    if split == "validation":
        warnings.warn("anthropic HH has no validation set, so using test set instead")
        split = "test"

    ds = load_dataset("Anthropic/hh-rlhf")[split]

    def convert_string_format(string):
        split_string = string.split("\n\nHuman: ")

        string_to_use = ""
        prompt_string_to_use = ""

        for item in split_string:
            if len(item) == 0:
                continue

            output = item.split("\n\nAssistant: ")

            if len(output) != 2:
                return None
            else:
                body, response = output

            if len(string_to_use) == 0:
                prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
                string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
            else:
                prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
                string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)

        # for prompt, remove the space at the end
        return string_to_use, prompt_string_to_use[:-1]

    list_of_dicts = []

    chosen = list(map(convert_string_format, ds["chosen"]))
    rejected = list(map(convert_string_format, ds["rejected"]))

    for c, r in zip(chosen, rejected):
        if c is None or r is None:
            continue

        chosen_response, chosen_prompt = c
        rejected_response, rejected_prompt = r

        chosen_prompt = chosen_prompt + "\n"
        rejected_prompt = rejected_prompt + "\n"
        chosen_response = chosen_response.replace(chosen_prompt, "")
        rejected_response = rejected_response.replace(rejected_prompt, "")
        if chosen_prompt != rejected_prompt:
            continue

        chosen_dict = {
            "prompt": chosen_prompt,
            "response": chosen_response,
            "preference": "chosen",
        }

        rejected_dict = {
            "prompt": chosen_prompt,
            "response": rejected_response,
            "preference": "rejected",
        }

        list_of_dicts.append(chosen_dict)
        list_of_dicts.append(rejected_dict)

    return list_of_dicts


def convert_list_of_dict_to_jsonl(list_of_dict):
    return "\n".join(json.dumps(item) for item in list_of_dict)


def save_dataset_for_kto(list_of_dicts, split, save_dir):
    with open(Path(save_dir) / f"{split}.jsonl", "w") as file:
        for data_dict in list_of_dicts:
            json.dump(data_dict, file)
            file.write("\n")


if __name__ == "__main__":
    args = prepare_args()

    for split in ["train", "test"]:
        list_of_dicts = process_hh(split)
        save_dataset_for_kto(list_of_dicts, split, args.output_dir)
