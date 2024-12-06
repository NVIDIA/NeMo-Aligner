# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

"""
A script for removing problematic long dialogues from a chat dataset prior to training.
Any example that does not have any non-masked tokens after truncation is removed.

Usage:
  python3 remove_long_dialogues.py \
    --tokenizer_path <PATH TO TOKENIZER MODEL> \
    --tokenizer_type sentencepiece
    --dataset_file <PATH TO DATASET TO PREPROCESS> \
    --output_file <WHERE TO SAVE PREPROCESSED DATASET> \
    --seq_len <MAX_SEQ_LEN TO USE DURING TRAINING>

Note:
  - Tokenizer path supports SentencePiece tokenizer and HF tokenizer.
    For SentencePiece tokenizer, specify the file /path/to/tokenizer.model.
    For HF tokenizer, specify a folder /path/to/hf_folder which contains tokenizer.json, tokenizer_config.json
    and special_tokens_map.json, or the HF name of the tokenizer to use (e.g. "meta-llama/Meta-Llama-3-8B")
"""


def test_index(tokenizer_path, tokenizer_type, dataset_file, output_file, seq_len=4096):
    if tokenizer_type == "huggingface":
        # pass in either a local Hugging Face folder which contains tokenizer.json or a path to the tokenizer on huggingface
        tokenizer = get_nmt_tokenizer(library="huggingface", model_name=tokenizer_path, use_fast=True)
    elif tokenizer_type == "sentencepiece":
        tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=tokenizer_path)
    else:
        raise ValueError(f"unsupported tokenizer type {tokenizer_type}")
    d = GPTSFTChatDataset(dataset_file, tokenizer, seq_len, 1)
    total_records = len(d)
    removed_ids = set()

    num_removed = 0
    for i in range(total_records):
        if i % 1000 == 0 and i != 0:
            print(f"Processing {i + 1}/{total_records}")
            print(f"% removed so far {num_removed}/{i + 1} = {(num_removed / (i + 1)) * 100:.2f}%")

        try:
            sample = d[i]
        except:
            num_removed += 1
            removed_ids.add(i)
            continue

        if d[i]["mask"][: seq_len + 1].sum().item() == 0:
            num_removed += 1
            removed_ids.add(i)
            continue

    with open(dataset_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            if i in removed_ids:
                continue
            j = json.loads(line)
            for conv in j["conversations"]:
                conv["canoncial_form"] = conv.get("canoncial_form", "")
                conv["label"] = conv.get("label", None)
            invalid_keys = []
            for key in j.keys():
                if key not in ["system", "mask", "conversations", "dataset"]:
                    invalid_keys.append(key)
            for k in invalid_keys:
                j.pop(k)

            o.write(json.dumps(j) + "\n")
            o.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="sentencepiece",
        help="The tokenizer type to use. Should be one of 'huggingface' or 'sentencepiece'.",
    )
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=False, default=4096)
    args = parser.parse_args()
    test_index(
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        dataset_file=args.dataset_file,
        output_file=args.output_file,
        seq_len=args.seq_len,
    )
