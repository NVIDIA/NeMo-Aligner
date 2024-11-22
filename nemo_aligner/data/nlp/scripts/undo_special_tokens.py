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

"""Script to remove special tokens from dpo datasets 
and convert them into list of messages format"""

import argparse
import json
import re


def format_conversation(input_string):
    # Define roles and patterns
    role_patterns = {"<extra_id_0>System": "system", "<extra_id_1>User": "user", "<extra_id_1>Assistant": "assistant"}

    # Initialize an empty output list
    conversation = []

    # Use regex to find each segment's role and content
    segments = re.findall(r"(<extra_id_[0-1]>[^\n]+)\n(.*?)((?=<extra_id_)|$)", input_string, re.DOTALL)

    for segment in segments:
        role_tag, content, _ = segment
        role = role_patterns.get(role_tag.strip(), "unknown")
        conversation.append({"role": role, "content": content.strip()})

    return conversation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file.")
    parser.add_argument("input_jsonl", type=str, help="Path to the input JSONL file.")
    # Parse the arguments
    args = parser.parse_args()

    input_jsonl = args.input_jsonl
    output_jsonl = input_jsonl.replace(".jsonl", ".no_special_toks.jsonl")

    with open(input_jsonl, "r") as f, open(output_jsonl, "w") as w:
        for line in f:
            j = json.loads(line)
            prompt = j["prompt"]
            undo_spl_prompt = format_conversation(prompt)
            empty_assistant = undo_spl_prompt.pop()
            chosen, rejected = j["chosen_response"], j["rejected_response"]
            chosen = chosen.split("\n<extra_id_1>")[0]
            rejected = rejected.split("\n<extra_id_1>")[0]
            chosen_message = {"role": empty_assistant["role"], "content": chosen}
            rejected_message = {"role": empty_assistant["role"], "content": rejected}
            j_out = {
                "prompt": undo_spl_prompt,
                "chosen_response": chosen_message,
                "rejected_response": rejected_message,
                "chosen_reward": j["chosen_reward"],
                "rejected_reward": j["rejected_reward"],
            }
            w.write(json.dumps(j_out) + "\n")
