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
This script is for processing data from Attribute-conditioned SFT training format into regression reward model training format.
"""


import argparse
import json

from common import (
    ALL_STEERLM_ATTRIBUTES,
    ASSISTANT_TURN_TEMPLATE,
    LABEL_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TURN_TEMPLATE,
)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file", type=str, required=True,
    )
    parser.add_argument(
        "--input-file", type=str, required=True,
    )
    return parser.parse_args()


def parse(s):
    # Split the string by comma
    try:
        pairs = s.split(",")

        # Split each pair by colon to separate key and value
        result = {pair.split(":")[0]: pair.split(":")[1] for pair in pairs}
        assert len(result) > 0, "At least one attribute should be present"
        return result
    except Exception:
        raise Exception("invalid sample", s)


def process_sample(line, fout):
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
    conversations = line["conversations"]
    user = line["mask"]
    for turn in conversations:
        value = turn["value"]
        if turn["from"] == user:
            text += USER_TURN_TEMPLATE.format(value=value)
        else:
            text += ASSISTANT_TURN_TEMPLATE.format(value=value)

        if "label" in turn and turn["label"]:  # label field is present and not None or empty
            out_text = text + LABEL_PREFIX
            given_attrs = parse(turn["label"])
            labels = [float(given_attrs.get(a, -100)) for a in ALL_STEERLM_ATTRIBUTES]
            newline = {"text": out_text, "label": labels}

            fout.write(json.dumps(newline, ensure_ascii=False) + "\n")


def main(args):
    f = open(args.input_file, "r", encoding="utf-8")
    fout = open(args.output_file, "w", encoding="utf-8")

    lines = f.readlines()

    for line in lines:
        jline = json.loads(line)
        process_sample(jline, fout)

    f.close()
    fout.close()


if __name__ == "__main__":
    main(prepare_args())
