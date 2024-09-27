# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
import os.path
import random
from typing import List, Union

import numpy as np
import tqdm
from cai_utils import UserAssistantPromptTemplate, remote_inference


def generate_chat_prompt(sample: dict):
    dataset_name = "sl_cai_samples"
    assert isinstance(sample, dict)
    assert all(key in sample for key in ["initial_prompt", "revision_response"])
    assert dataset_name is not None and dataset_name != ""

    def _message_formatter(role: str, message: str):
        assert role in ["User", "Assistant"]

        return {"from": f"{role}", "value": f"{message}", "label": None}

    def _conversation_formatter(messages: list, system_message: str = ""):
        assert isinstance(dataset_name, str)
        assert isinstance(messages, list) and len(messages) > 0

        return {"system": f"{system_message}", "conversations": messages, "mask": "User", "dataset": f"{dataset_name}"}

    prompt = _message_formatter(role="User", message=sample["initial_prompt"])
    revision = _message_formatter(role="Assistant", message=sample["revision_response"])

    return _conversation_formatter(messages=[prompt, revision])


def model_remote_inference(prompt, inference_config: dict):
    sentences = remote_inference(prompt=prompt, **inference_config)
    return sentences


def generate_cai_batch_sample(
    prompt_list: list,
    few_shot_dataset_path: str,
    critique_list,
    revision_list,
    inference_config: dict,
    prompt_template_config: dict,
    apply_chat_template: bool,
):
    assert isinstance(prompt_list, list)
    if not isinstance(critique_list, list):
        critique_list = [critique_list] * len(prompt_list)
    if not isinstance(revision_list, list):
        revision_list = [revision_list] * len(prompt_list)

    num_prompts = len(prompt_list)
    assert len(critique_list) == num_prompts
    assert len(revision_list) == num_prompts

    prompt_template = UserAssistantPromptTemplate(**prompt_template_config)

    with open(few_shot_dataset_path, "r") as f:
        few_shot_samples = json.load(f)
        few_shot_prompts = [item for sublist in few_shot_samples["samples"] for item in sublist]

    # get initial response
    print(f"\nGenerating initial response for {num_prompts} prompts...")
    initial_prompt_batch = [
        few_shot_prompts + [{"content": p, "role": UserAssistantPromptTemplate.Role.User}] for p in prompt_list
    ]

    chat_batch = model_remote_inference(
        [prompt_template.format_messages(p) if apply_chat_template else p for p in initial_prompt_batch],
        inference_config=inference_config,
    )

    assert len(chat_batch) == num_prompts
    initial_response_batch = [prompt_template.extract_response(chat) for chat in chat_batch]
    print("Done")

    # generate a single critique
    print(f"\nGenerating a critique for {num_prompts} initial responses...")
    critique_request_batch = [
        initial_prompt_batch[i]
        + [{"content": initial_response_batch[i], "role": UserAssistantPromptTemplate.Role.Assistant}]
        + [{"content": cr_p, "role": UserAssistantPromptTemplate.Role.User}]
        for i, cr_p in enumerate(critique_list)
    ]

    chat_batch = model_remote_inference(
        [prompt_template.format_messages(p) if apply_chat_template else p for p in critique_request_batch],
        inference_config=inference_config,
    )
    assert len(chat_batch) == num_prompts
    critique_response_batch = [prompt_template.extract_response(chat) for chat in chat_batch]
    print("Done")

    # generate a single revision
    print(f"\nGenerating {num_prompts} revisions ...")
    revision_request_prompt_batch = [
        critique_request_batch[i]
        + [{"content": critique_response_batch[i], "role": UserAssistantPromptTemplate.Role.Assistant}]
        + [{"content": rev_p, "role": UserAssistantPromptTemplate.Role.User}]
        for i, rev_p in enumerate(revision_list)
    ]

    chat_batch = model_remote_inference(
        [prompt_template.format_messages(p) if apply_chat_template else p for p in revision_request_prompt_batch],
        inference_config=inference_config,
    )
    assert len(chat_batch) == num_prompts
    revision_response_batch = [prompt_template.extract_response(chat) for chat in chat_batch]
    print("Done")

    s_batch = []
    for i in range(num_prompts):
        s = dict(
            initial_prompt=prompt_list[i],
            initial_response=initial_response_batch[i],
            critique_prompt=critique_list[i],
            critique_response=critique_response_batch[i],
            revision_prompt=revision_list[i],
            revision_response=revision_response_batch[i],
        )
        s_batch.append(s)
    return s_batch


def load_critique_revision_instructions(file_path: str):
    instructions = []
    with open(file_path, "r") as file:
        json_object = json.load(file)
        for i in range(len(json_object)):
            inst = json_object[f"harmful{i}"]
            assert len(inst) == 2 and "prompt" in inst and "edit_request" in inst
            assert len(inst["prompt"]) == 1

            critique_prompt = inst["prompt"][0]
            revision_prompt = inst["edit_request"]

            instructions.append(dict(critique_prompt=critique_prompt, revision_prompt=revision_prompt))

    return instructions


def get_red_team_train_human_prompts(red_teaming_dataset_path: str) -> list:
    def strip_first_red_team_prompt(data_item):
        human_index = data_item["transcript"].find("Human:")
        assistant_index = data_item["transcript"].find("Assistant:")
        assert human_index >= 0
        assert assistant_index >= 0 and assistant_index > human_index

        human_index += len("Human:")
        prompt = data_item["transcript"][human_index:assistant_index]
        prompt = prompt.strip()
        return prompt

    red_teaming_prompts = []
    with open(red_teaming_dataset_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            red_teaming_prompt = strip_first_red_team_prompt(json_object)
            red_teaming_prompts.append(red_teaming_prompt)

    return red_teaming_prompts


def generate_cai_dataset(
    red_teaming_dataset_path: str,
    few_shot_samples_filepath: str,
    critique_revision_instructions_filepath: str,
    num_examples: int,
    batch_size: int,
    save_to_file_interval: int,
    save_file_path: str,
    inference_config: dict,
    prompt_template_config: dict,
    apply_chat_template: bool,
):
    """
    @param batch_size: inference batch size
    @param save_to_file_interval: saves generated samples to the disk after 'save_to_file_interval' batches.
    @param save_file_path:
    @param red_teaming_dataset_path:
    @param few_shot_samples_filepath:
    @param critique_revision_instructions_filepath:
    @param num_examples:
    @param inference_config:
    @param prompt_template_config:
    @param apply_chat_template:
    @return:
    """
    assert batch_size > 0
    assert save_to_file_interval > 0
    assert save_file_path is not None and save_file_path != ""
    assert not os.path.exists(save_file_path), f"{save_file_path} already exists"

    log_dir = os.path.join(os.path.dirname(save_file_path), "cai_output")
    os.makedirs(log_dir, exist_ok=True)
    output_path_critique_revision = os.path.join(log_dir, f"cai_critique_revision_samples.json")
    output_path_cai_samples = os.path.join(log_dir, f"cai_samples.jsonl")

    # load constitution critique/revision instructions
    critique_revision_instructions_set = load_critique_revision_instructions(critique_revision_instructions_filepath)

    def sample_random_critique_revision_set():
        a_critique_list = []
        a_revision_list = []
        for i in range(batch_size):
            rnd_index = random.randint(0, len(critique_revision_instructions_set) - 1)
            a_critique_list.append(critique_revision_instructions_set[rnd_index]["critique_prompt"])
            a_revision_list.append(critique_revision_instructions_set[rnd_index]["revision_prompt"])
        return a_critique_list, a_revision_list

    red_teaming_prompts = get_red_team_train_human_prompts(red_teaming_dataset_path)

    num_examples = len(red_teaming_prompts) if num_examples < 0 else min(num_examples, len(red_teaming_prompts))
    assert num_examples > 0

    cai_samples = []
    critique_revision_samples = []
    num_batches = 0
    for index in tqdm.tqdm(range(0, num_examples, batch_size), desc="Generating CAI samples, batch #"):
        red_teaming_prompts_list = red_teaming_prompts[index : index + batch_size]
        if len(red_teaming_prompts_list) < batch_size:
            break

        # sample random critique/revision instruction set
        critique_list, revision_list = sample_random_critique_revision_set()
        critique_list = [c.removesuffix("\n\nCritique:").removeprefix("\n\nCritiqueRequest: ") for c in critique_list]
        revision_list = [c.removesuffix("\n\nRevision:").removeprefix("\n\nRevisionRequest: ") for c in revision_list]

        # call model
        cai_batch_sample = generate_cai_batch_sample(
            red_teaming_prompts_list,
            few_shot_dataset_path=few_shot_samples_filepath,
            critique_list=critique_list,
            revision_list=revision_list,
            inference_config=inference_config,
            prompt_template_config=prompt_template_config,
            apply_chat_template=apply_chat_template,
        )
        assert len(cai_batch_sample) == len(red_teaming_prompts_list)

        print(f"Reformatting CAI samples according to a chat prompt template...")
        for sample in cai_batch_sample:
            chat_formatted_cai_sample = generate_chat_prompt(sample=sample)
            cai_samples.append(chat_formatted_cai_sample)
            critique_revision_samples.append(sample)
        print("Done")

        num_batches += 1

        if num_batches % save_to_file_interval == 0:
            with open(save_file_path, "w") as f:
                for sample in cai_samples:
                    json_line = json.dumps(sample)
                    f.write(json_line + "\n")

    # final save
    with open(save_file_path, "w") as f:
        for sample in cai_samples:
            json_line = json.dumps(sample)
            f.write(json_line + "\n")

    # logs (saving all raw files)
    with open(output_path_critique_revision, "w") as f:
        json.dump(critique_revision_samples, f, indent=4)

    with open(output_path_cai_samples, "w") as f:
        for sample in cai_samples:
            json_line = json.dumps(sample)
            f.write(json_line + "\n")

    return cai_samples


def remove_long_dialogs_wrapper(
    input_file_path: str, max_seq_length: int, tokenizer_model: str, tokenizer_library: str
):
    from cai_utils import remove_long_dialogs

    if max_seq_length is None or max_seq_length <= 0:
        return
    assert os.path.isfile(tokenizer_model)
    assert tokenizer_library is not None

    remove_long_out_dir = os.path.join(os.path.dirname(input_file_path), "removed_long_dialogs")
    statistics_file_path = os.path.join(remove_long_out_dir, "remove_long_dialogs_statistics.json")
    os.makedirs(remove_long_out_dir, exist_ok=True)

    import shutil

    copy_input_file_path = os.path.join(remove_long_out_dir, os.path.basename(input_file_path))
    shutil.copy(input_file_path, copy_input_file_path)

    d = remove_long_dialogs(
        input_file_path=copy_input_file_path,
        max_seq_length=max_seq_length,
        tokenizer_model=tokenizer_model,
        tokenizer_library=tokenizer_library,
        output_dir=remove_long_out_dir,
        use_pool=True,
    )

    # save output statistics
    with open(statistics_file_path, "w") as f:
        json.dump(d, f, indent=4)

    # delete files
    if os.path.exists(copy_input_file_path):
        os.remove(copy_input_file_path)

    return d["output_file"]


def validate_dataset_structure(item: dict):
    """
    expected format:

        prompt = {
            'system': f"{system_message}",
            'conversations': [
                {'from': 'User',
                 'value': f"{user_message}",
                 'label': default_label},

                {'from': 'Assistant',
                 'value': f"{assistant_message}",
                 'label': default_label}
            ],
            "mask": "User",
            "dataset": f"{dataset}",
        }
    """

    required_fields = ["system", "conversations", "mask", "dataset"]
    for field in required_fields:
        if field not in item:
            return False

    for message in item["conversations"]:
        if "from" not in message or "value" not in message:
            return False

    return True


def blend_cai_sft_dataset(helpfulness_dataset_path, cai_samples_filepath, blended_sl_cai_filename, summary_filename):
    input_files = [helpfulness_dataset_path, cai_samples_filepath]
    assert all(os.path.isfile(input_file) for input_file in input_files)

    samples = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)

    # make sure all samples have a valid structure
    assert all(validate_dataset_structure(s) for s in samples)

    # permute samples
    samples_permuted = [samples[i] for i in np.random.permutation(len(samples))]

    # Each sample is in a separate line
    with open(blended_sl_cai_filename, "w") as f:
        for sample in samples_permuted:
            json_line = json.dumps(sample)
            f.write(json_line + "\n")

    # create a summary file
    with open(summary_filename, "w") as f:
        summary_dict = dict(input_files=input_files, output_file=blended_sl_cai_filename, prompt_template="chat")
        json.dump(summary_dict, f, indent=4)


def prepare_args():
    parser = argparse.ArgumentParser(description="Generate CAI dataset")
    parser.add_argument("--max-seq-length", type=int, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--tokenizer-model", type=str, required=False, default=None)
    parser.add_argument("--tokenizer-library", type=str, required=False, default=None)
    parser.add_argument(
        "--num-examples",
        type=int,
        required=False,
        default=-1,
        help="Number of samples to generate. Default (-1) generates for all the samples in the dataset.",
    )
    parser.add_argument("--save-to-file-interval", type=int, required=False, default=1)
    parser.add_argument("--red-teaming-prompts-dataset-path", type=str, required=True, default=None)
    parser.add_argument("--few-shot-prompts-dataset-path", type=str, required=True, default=None)
    parser.add_argument("--critique-revision-instructions-path", type=str, required=True, default=None)
    parser.add_argument(
        "--output-filepath", type=str, required=False, default="cai_revisions_aligner_chat_template.jsonl"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--helpfulness-dataset-path", type=str, required=True, default=None)

    group_inference = parser.add_argument_group("inference", "inference (service) arguments")
    group_inference.add_argument("--add_bos", type=str, choices=["True", "False"], default="False")
    group_inference.add_argument("--top_k", type=int, default=1)
    group_inference.add_argument("--top_p", type=float, default=0.9)
    group_inference.add_argument("--all_probs", type=str, choices=["True", "False"], default="False")
    group_inference.add_argument("--repetition_penalty", type=float, default=1.2)
    group_inference.add_argument("--min_tokens_to_generate", type=int, default=1)
    group_inference.add_argument("--temperature", type=float, default=1.0)
    group_inference.add_argument("--greedy", type=str, choices=["True", "False"], default="True")
    group_inference.add_argument("--tokens_to_generate", type=int, default=1024)
    group_inference.add_argument("--end_strings", type=str, nargs="*", default=None)
    group_inference.add_argument(
        "--port", type=int, default=5656, help="The port number on which the inference service is running"
    )
    group_inference.add_argument(
        "--host", type=str, default="localhost", help="The hostname or IP address of the inference service"
    )

    # prompt template
    group_prompt_template = parser.add_argument_group("prompt_template", "prompt template")
    group_prompt_template.add_argument("--apply_chat_template", type=str, choices=["True", "False"], default="False")
    group_prompt_template.add_argument("--user_format", type=str, default=None)
    group_prompt_template.add_argument("--assistant_format", type=str, default=None)
    group_prompt_template.add_argument("--system_format", type=str, default=None)
    group_prompt_template.add_argument("--system_default_message", type=str, default=None)
    group_prompt_template.add_argument("--bos_token", type=str, default=None)
    group_prompt_template.add_argument("--eos_token", type=str, default=None)
    group_prompt_template.add_argument("--response_extract_pattern", type=str, default="[/INST]")

    """
    prompt template configuration is going to be applied to chat conversation messages when invoking remote
    inference with megatron_gpt_eval.py service.
    
    
    prompt template configuration example: <extra_id_*> template
        --apply_chat_template True
        --user_format "<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n"
        --assistant_format "{MESSAGE}\n"
        --system_format "<extra_id_0>System\n{MESSAGE}\n"
        --system_default_message ""
        --eos_token "<extra_id_1>"
        --response_extract_pattern "<extra_id_1>Assistant\n"
        
    
    prompt template configuration example: mistral-instruct-7B
        --apply_chat_template False
        --response_extract_pattern "[/INST]"
        
    NOTE: setting 'apply_chat_template' to False as chat template is going to be applied when 
    invoking remote inference with megatron_gpt_eval.py service.
    
    
    prompt template configuration example: Mistral-Instruct-7B, converted to nemo, 
        using huggingface tokenizer and not using nemo tokenizer:
    
    -apply_chat_template True
        --user_format "[INST] {MESSAGE} [/INST]"
        --assistant_format "{MESSAGE}</s> "
        --bos_token "<s>"
        --eos_token "</s>"
        --response_extract_pattern "[/INST]"
    """

    args = parser.parse_args()
    args.add_bos = args.add_bos == "True"
    args.all_probs = args.all_probs == "True"
    args.greedy = args.greedy == "True"
    args.apply_chat_template = args.apply_chat_template == "True"

    assert os.path.isfile(args.red_teaming_prompts_dataset_path)
    assert os.path.isfile(args.few_shot_prompts_dataset_path)
    assert os.path.isfile(args.helpfulness_dataset_path)
    assert args.num_examples >= -1 and args.num_examples != 0

    if args.max_seq_length is not None:
        assert args.max_seq_length > 0
        assert os.path.isfile(args.tokenizer_model)
        assert args.tokenizer_library is not None

    def _process_string(s: str):
        return s.encode("utf-8").decode("unicode_escape")

    # Convert parsed arguments to dictionary
    args_dict = vars(args)
    inference_config = {
        k: v
        for k, v in args_dict.items()
        if k
        in {
            "add_bos",
            "top_k",
            "top_p",
            "all_probs",
            "repetition_penalty",
            "min_tokens_to_generate",
            "temperature",
            "greedy",
            "tokens_to_generate",
            "end_strings",
            "port",
            "host",
        }
    }
    if inference_config["end_strings"] is not None:
        inference_config["end_strings"] = list(map(_process_string, inference_config["end_strings"]))

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

    return args, inference_config, prompt_template_config


def main():
    args, inference_config, prompt_template_config = prepare_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Generating CAI samples...")
    generate_cai_dataset(
        red_teaming_dataset_path=args.red_teaming_prompts_dataset_path,
        few_shot_samples_filepath=args.few_shot_prompts_dataset_path,
        critique_revision_instructions_filepath=args.critique_revision_instructions_path,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        save_to_file_interval=args.save_to_file_interval,
        save_file_path=args.output_filepath,
        inference_config=inference_config,
        prompt_template_config=prompt_template_config,
        apply_chat_template=args.apply_chat_template,
    )

    print("Blending CAI samples with the helpfulness dataset...")
    blend_cai_sft_dataset(
        helpfulness_dataset_path=args.helpfulness_dataset_path,
        cai_samples_filepath=args.output_filepath,
        blended_sl_cai_filename=args.output_filepath,
        summary_filename=args.output_filepath.replace(".json", ".summary.json"),
    )

    if args.max_seq_length is not None:
        print("Removing long dialogs from dataset...")
        remove_long_dialogs_wrapper(
            input_file_path=args.output_filepath,
            max_seq_length=args.max_seq_length,
            tokenizer_model=args.tokenizer_model,
            tokenizer_library=args.tokenizer_library,
        )

    print("Done")


if __name__ == "__main__":
    main()
