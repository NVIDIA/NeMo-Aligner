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
import os
import warnings
from pathlib import Path

from cai_utils import UserAssistantPromptTemplate
from datasets import DatasetDict, concatenate_datasets, load_dataset

"""

example for preprocessing Anthropic helpful-only dataset:

python process_anthropic_hh_using_chat_prompt.py
    --output-dir <output-dir>
    --dataset-dir-name helpful-base helpful-online helpful-rejection-sampled
    --output-file-name-prefix anthropic_helpful_only    
    
example for preprocessing Anthropic HH full dataset:

python process_anthropic_hh_using_chat_prompt.py
    --output-dir <output-dir>
    --output-file-name-prefix anthropic_hh_full
"""


def prepare_args():
    parser = argparse.ArgumentParser(
        description="downloads anthropic-hh dataset and converts to chat prompt template."
    )
    parser.add_argument("--output-dir", type=str, required=True, default=None)
    parser.add_argument("--dataset-dir-name", type=str, default=None, nargs="*")
    parser.add_argument("--output-file-name-prefix", type=str, default=None)
    parser.add_argument(
        "--add-eos",
        type=str,
        default="True",
        choices=["False", "True"],
        help="add eos-token at the end of a comparison prompt?",
    )

    # prompt template
    group_prompt_template = parser.add_argument_group("prompt_template", "prompt template")
    group_prompt_template.add_argument(
        "--user_format", type=str, default="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n"
    )
    group_prompt_template.add_argument("--assistant_format", type=str, default="{MESSAGE}\n")
    group_prompt_template.add_argument("--system_format", type=str, default="<extra_id_0>System\n{MESSAGE}\n")
    group_prompt_template.add_argument("--system_default_message", type=str, default="")
    group_prompt_template.add_argument("--bos_token", type=str, default=None)
    group_prompt_template.add_argument("--eos_token", type=str, default="<extra_id_1>")
    group_prompt_template.add_argument("--response_extract_pattern", type=str, default="<extra_id_1>Assistant\n")

    args = parser.parse_args()
    args.add_eos = args.add_eos in ["True", "true"]

    if (
        (args.dataset_dir_name is None)
        or (len(args.dataset_dir_name) == 0)
        or (len(args.dataset_dir_name) == 0 and args.dataset_dir_name[0] is None)
    ):
        if args.output_file_name_prefix is None or args.output_file_name_prefix == "":
            args.output_file_name_prefix = "anthropic_hh"

        args.dataset_dir_name = [None]

    def _process_string(s: str):
        return s.encode("utf-8").decode("unicode_escape")

    args_dict = vars(args)
    prompt_template_config = {
        k: _process_string(v) if v is not None else v
        for k, v in args_dict.items()
        if k
        in {
            "user_format",
            "assistant_format",
            "system_format",
            "system_default_message",
            "bos_token",
            "eos_token",
            "response_extract_pattern",
        }
    }

    return args, prompt_template_config


def _process_samples(dataset, add_eos: bool, prompt_template_config: dict):
    def convert_string_format(string):
        split_string = [s.strip() for s in string.split("\n\nHuman: ")]
        split_string = [s for s in split_string if len(s) > 0]

        prompt_template = UserAssistantPromptTemplate(**prompt_template_config)
        messages = []

        for i, item in enumerate(split_string):

            output = item.split("\n\nAssistant: ")
            if len(output) != 2:
                return None
            body, response = output

            # handle special case
            for a in ["\n\nHuman", "\n\nHuman:", "\n\nHuman: "]:
                if response.endswith(a):
                    response = response[: -len(a)]
                    break

            body = body.strip().strip("\n")
            response = response.strip().strip("\n")
            if len(body) == 0 or len(response) == 0:
                return None

            messages.append(prompt_template.create_user_message(body))
            messages.append(prompt_template.create_assistant_message(response))

        if len(messages) == 0:
            return None

        prompt_string_to_use = prompt_template.format_messages(messages[:-1])
        string_to_use = prompt_template.format_messages(messages)

        if add_eos:
            string_to_use += prompt_template.eos_token

        # for prompt, remove the space at the end
        return string_to_use, prompt_string_to_use

    chosen = list(map(convert_string_format, dataset["chosen"]))
    rejected = list(map(convert_string_format, dataset["rejected"]))

    samples = []
    for c, r in zip(chosen, rejected):
        if c is None or r is None:
            continue

        chosen_response, chosen_prompt = c
        rejected_response, rejected_prompt = r

        if chosen_prompt != rejected_prompt:
            continue
        if len(chosen_prompt) == 0:
            continue

        chosen_response, chosen_prompt = c
        rejected_response, rejected_prompt = r
        if len(chosen_response) == 0 or len(rejected_response) == 0:
            continue

        comparison_dict = {
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

        samples.append(comparison_dict)

    return samples


def _convert_list_of_dict_to_jsonl(list_of_dict):
    return "\n".join(json.dumps(item) for item in list_of_dict)


def _save_dataset(list_of_dict, split: str, save_dir: str, output_file_name_prefix: str, add_eos: bool):
    prompts_to_save = _convert_list_of_dict_to_jsonl({"text": item["prompt"]} for item in list_of_dict)
    prompts_file_name = f"{split}_prompts_with_chat_prompt"
    if output_file_name_prefix is not None and output_file_name_prefix != "":
        prompts_file_name = f"{output_file_name_prefix}_{prompts_file_name}"
    if not add_eos:
        prompts_file_name += "_no_extra_id_eos"
    prompts_file_name += ".jsonl"

    with open(Path(save_dir) / prompts_file_name, "w") as f:
        f.write(prompts_to_save)

    comparisons_to_save = []

    for item in list_of_dict:
        comparisons_to_save.append({"text": item["chosen"]})
        comparisons_to_save.append({"text": item["rejected"]})

    comparisons_to_save = _convert_list_of_dict_to_jsonl(comparisons_to_save)
    comparisons_file_name = f"{split}_comparisons_with_chat_prompt"
    if output_file_name_prefix is not None and output_file_name_prefix != "":
        comparisons_file_name = f"{output_file_name_prefix}_{comparisons_file_name}"
    if not add_eos:
        comparisons_file_name += "_no_extra_id_eos"
    comparisons_file_name += ".jsonl"

    with open(Path(save_dir) / comparisons_file_name, "w") as f:
        f.write(comparisons_to_save)


def _load_anthropic_dataset(dataset_dir_name=None, split_names=None):
    assert isinstance(dataset_dir_name, list)
    if len(dataset_dir_name) == 0 or (len(dataset_dir_name) == 1 and dataset_dir_name[0] is None):
        return load_dataset("Anthropic/hh-rlhf")
    assert len(set([n for n in dataset_dir_name])) == len(dataset_dir_name)

    anthropic_supported_dir_names = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    for dir_name in dataset_dir_name:
        assert dir_name in anthropic_supported_dir_names

    anthropic_supported_split_names = ["test", "train"]
    if split_names is None or len(split_names) == 0:
        split_names = ["train", "test"]
    else:
        assert len(set([n for n in split_names])) == len(split_names)
        for i, split_name in enumerate(split_names):
            if split_name == "validation":
                warnings.warn("anthropic HH has no validation set, using test set instead")
                split_names[i] = "test"

    split_names_set = set(split_names)
    datasets = []
    dataset_splits = []
    dataset_dir_names = []
    for i, dir_name in enumerate(dataset_dir_name):
        ds_i = load_dataset("Anthropic/hh-rlhf", data_dir=dir_name)
        ds_i_splits = set(ds_i.keys())
        ds_i_splits = set.intersection(split_names_set, ds_i_splits)
        assert len(set.intersection(split_names_set, ds_i_splits)) == len(split_names_set)

        dataset_dir_names.append(dir_name)
        datasets.append(ds_i)
        dataset_splits.append(ds_i_splits)

    ds_concat_by_split = {}
    for ds_split in split_names:
        ds_concat_i = concatenate_datasets([ds_i[ds_split] for ds_i in datasets])
        ds_concat_by_split[ds_split] = ds_concat_i

    # Combine the concatenated splits into a single DatasetDict
    merged_dataset = DatasetDict(ds_concat_by_split)
    return merged_dataset


def main():
    args, prompt_template_config = prepare_args()
    os.makedirs(args.output_dir, exist_ok=True)

    anthropic_dataset = _load_anthropic_dataset(args.dataset_dir_name, split_names=["train", "test"])

    for split in ["train", "test"]:
        list_of_dicts = _process_samples(
            anthropic_dataset[split], add_eos=args.add_eos, prompt_template_config=prompt_template_config
        )
        _save_dataset(list_of_dicts, split, args.output_dir, args.output_file_name_prefix, add_eos=args.add_eos)


if __name__ == "__main__":
    main()
