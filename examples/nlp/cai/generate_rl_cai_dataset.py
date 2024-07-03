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
import ast
import json
import os
import os.path
import random
import re
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from cai_utils import UserAssistantPromptTemplate, remote_inference, remote_inference_with_ngc
from tqdm import tqdm


def generate_cai_rlaif_candidate_dataset(
    batch_size: int,
    temperatures: Union[List, int],
    red_teaming_dataset_path: str,
    inference_config: dict,
    prompt_template_config: dict,
):
    """
    @param host:
    @return:
    @param batch_size: inference batch size
    @param temperatures: how many temperatures to use for generation per prompt
    @param red_teaming_dataset_path: path to Anthropic red teaming prompt attempts.
    @param inference_config:
    @return:
    """
    assert batch_size > 0
    assert isinstance(temperatures, List) or isinstance(temperatures, int)
    if isinstance(temperatures, int):
        temperatures = [temperatures]

    inference_config = inference_config.copy()
    red_teaming_prompts = get_red_team_train_human_prompts(red_teaming_dataset_path)

    all_samples = []
    samples_per_temperature = {}

    for batch_index in tqdm(range(0, len(red_teaming_prompts), batch_size), desc="Batch #"):
        red_teaming_prompts_list = red_teaming_prompts[batch_index : batch_index + batch_size]
        if len(red_teaming_prompts_list) < batch_size:
            break

        for t in tqdm(temperatures, desc="Temperatures"):
            samples = []

            # call model
            inference_config["temperature"] = t
            rlaif_batch_samples = generate_responses_batch(
                red_teaming_prompts_list,
                inference_config=inference_config,
                prompt_template_config=prompt_template_config,
            )
            samples.extend(rlaif_batch_samples)

            samples_per_temperature[str(t)] = samples

        all_samples.extend(join_responses(samples_per_temperature))

    return all_samples


def generate_responses_batch(prompt_list: list, inference_config: dict, prompt_template_config: dict):
    assert isinstance(prompt_list, list)
    num_prompts = len(prompt_list)

    prompt_template = UserAssistantPromptTemplate(**prompt_template_config)

    # get initial response
    prompts = [
        prompt_template.format_messages({"content": p, "role": UserAssistantPromptTemplate.Role.User})
        for p in prompt_list
    ]
    responses = model_remote_inference(prompts, inference_config=inference_config)
    assert len(responses) == num_prompts
    stripped_responses = [prompt_template.extract_response(r) for r in responses]

    s_batch = []
    for i in range(num_prompts):
        s = dict(prompt=prompt_list[i], response=stripped_responses[i])
        s_batch.append(s)

    return s_batch


def model_remote_inference(prompt, inference_config: dict):
    sentences = remote_inference(prompt=prompt, **inference_config)
    return sentences


def get_red_team_train_human_prompts(red_teaming_dataset_path: str) -> list:
    def strip_prompt(data_item):
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
            red_teaming_prompt = strip_prompt(json_object)
            red_teaming_prompts.append(red_teaming_prompt)

    return red_teaming_prompts


def join_responses(samples_per_temperature: dict) -> list:
    samples = []

    temperatures = list(samples_per_temperature.keys())
    prompts = [d["prompt"] for d in samples_per_temperature[temperatures[0]]]

    for i, p in enumerate(prompts):
        responses_dict = {f"response_t={t}": samples_per_temperature[t][i]["response"] for t in temperatures}
        samples.append(dict(prompt=p, **responses_dict))

    return samples


def prepare_args():
    parser = argparse.ArgumentParser(
        description="given a prompt and to responses, "
        "selects the most harmless response (labeled as 'chosen') and "
        "the least harmless response (labeled as 'rejected')."
    )
    parser.add_argument("--batch-size", type=int, required=True, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-filename-prefix", type=str, default="cai_rlaif")
    parser.add_argument("--splits", type=str, default="{'train': 0.8, 'test': 0.2}", help="How to split the dataset")
    parser.add_argument("--shuffle", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--red-teaming-file-path", type=str, required=True, default=None)
    parser.add_argument("--sys-prompt-constitution-file-path", type=str, required=True, default=None)

    parser.add_argument(
        "--blend-with",
        type=str,
        default=None,
        help="template:"
        "{'name': '<some-name-for the blending>', '<split-name>': {'prompts': ['<path>', '<path-2>'], 'comparisons': ['<path-1>', '<path-2>']}}"
        ""
        "you must set a valid name and one or more keys of <split-name>, one for each split in '--splits' argument",
    )

    group_ngc = parser.add_argument_group("NGC", "NGC arguments")
    group_ngc.add_argument("--ngc-api-key", type=str, required=True, default=None)
    group_ngc.add_argument("--ngc-url", type=str, default="https://integrate.api.nvidia.com/v1/chat/completions")
    group_ngc.add_argument("--ngc-model", type=str, default="mistralai/mixtral-8x7b-instruct-v0.1")

    group_inference = parser.add_argument_group("inference", "inference (service) arguments")
    group_inference.add_argument("--add_bos", type=str, choices=["True", "False"], default="False")
    group_inference.add_argument("--top_k", type=int, default=50)
    group_inference.add_argument("--top_p", type=float, default=0.95)
    group_inference.add_argument("--all_probs", type=str, choices=["True", "False"], default="False")
    group_inference.add_argument("--repetition_penalty", type=float, default=1.0)
    group_inference.add_argument("--min_tokens_to_generate", type=int, default=1)
    group_inference.add_argument("--temperature", type=float, default=1.0)
    group_inference.add_argument("--greedy", type=str, choices=["True", "False"], default="False")
    group_inference.add_argument("--tokens_to_generate", type=int, default=1024)
    group_inference.add_argument("--end_strings", type=str, nargs="*", default=["<extra_id_1>"])
    group_inference.add_argument(
        "--port", type=int, default=5656, help="The port number on which the inference service is running"
    )
    group_inference.add_argument(
        "--host", type=str, default="localhost", help="The hostname or IP address of the inference service"
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

    """
        The prompt template configuration will be utilized in two scenarios:
         1. It will be applied to chat conversation messages when invoking remote inference using the 
            megatron_gpt_eval.py service (specifically in the generate_cai_rlaif_candidate_dataset method).
         2. It will be used to format chat conversation messages when saving them to a file, which can be 
            used as training, testing, or validation datasets.
            
            
            
        prompt template configuration example: <extra_id_*> template
            --user_format "<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n"
            --assistant_format "{MESSAGE}\n"
            --system_format "<extra_id_0>System\n{MESSAGE}\n"
            --system_default_message ""
            --eos_token "<extra_id_1>"
            --response_extract_pattern "<extra_id_1>Assistant\n"
    """

    args = parser.parse_args()
    assert os.path.isfile(args.red_teaming_file_path)
    args.splits = ast.literal_eval(args.splits)
    args.shuffle = args.shuffle in ["True", "true"]

    args.add_bos = args.add_bos == "True"
    args.all_probs = args.all_probs == "True"
    args.greedy = args.greedy == "True"

    # blending argument validation
    if args.blend_with is not None:
        args.blend_with = ast.literal_eval(args.blend_with)
        assert all(split_name in args.blend_with for split_name in args.splits)
        assert len(args.blend_with) - 1 == len(args.splits)
        assert (
            "name" in args.blend_with
            and isinstance(args.blend_with["name"], str)
            and args.blend_with["name"] is not None
            and args.blend_with["name"] != ""
        )

        for split_name, blend in args.blend_with.items():
            if split_name == "name":  # ignore this key
                continue

            assert len(blend) == 2  # must have exactly 2 keys: 'prompts', 'comparisons'
            for blend_type, files in blend.items():
                assert blend_type in ["prompts", "comparisons"]
                if not isinstance(files, list):
                    files = [files]
                    blend[blend_type] = files
                for file in files:
                    assert os.path.isfile(
                        file
                    ), f"split={split_name}, blend-type={blend_type}. error => invalid file path: {file}"

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


def generate_ai_preference(
    sample: dict, ngc_api_key: str, system_prompt: str, seed: int, ngc_url: str, ngc_model: str
):
    # NOTE: For generating AI preferences we deviate a bit from the paper and instead of feeding one (randomized)
    # constitution principle at a time, we feed the entire constitution at once. Also, instead of using normalized
    # logprobs of the candidate response number tokens, we just ask the judge LLM to choose what is the most harmless
    # and toxic responses.
    # Although deviating a bit from the paper, we found it to work quite well in practice.

    prompt = sample["prompt"]
    if prompt is None or prompt == "":
        return None

    response_keys = sorted([key for key in sample.keys() if key.startswith("response")])
    responses = [sample[key] for key in response_keys]
    if len(responses) < 2:
        return None

    responses_to_choose_from_text = f"\n FULL TEXT: {prompt}"
    for i, response in enumerate(responses):
        responses_to_choose_from_text += f"\n #{i + 1}: \n {response}\n"
    responses_to_choose_from_text += "\nNow, choose the pair of responses as instructed:"

    full_prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": responses_to_choose_from_text},
    ]

    res = remote_inference_with_ngc(
        api_key=ngc_api_key,
        url=ngc_url,
        model=ngc_model,
        messages=full_prompt_messages,
        top_p=0,
        max_tokens=1024,
        temperature=0,
        seed=seed,
    )

    def _extract_decision_and_explanation(s):
        try:
            # convert string into a dictionary
            dict_start = s.find("{")
            dict_end = s.rfind("}") + 1
            dict_str = s[dict_start:dict_end]
            dict_result = ast.literal_eval(dict_str)

            # get decision
            s_remaining = s[dict_end:].strip()

            assert "chosen" in dict_result
            assert "rejected" in dict_result

            chosen_id = list(map(int, re.findall(r"\d+", str(dict_result["chosen"]))))
            rejected_id = list(map(int, re.findall(r"\d+", str(dict_result["rejected"]))))

            assert len(chosen_id) == 1 and 1 <= chosen_id[0] <= len(responses)
            assert len(rejected_id) == 1 and 1 <= rejected_id[0] <= len(responses)

            dict_result["chosen"] = chosen_id[0]
            dict_result["rejected"] = rejected_id[0]

            return dict_result, s_remaining
        except (ValueError, SyntaxError):
            # Return None or raise an error if the string does not contain a valid dictionary
            return None, None

    selected_pair, selection_explanation = _extract_decision_and_explanation(res)

    if selected_pair is None:
        return None

    # convert response number to response index
    chosen_response_index = selected_pair["chosen"] - 1
    rejected_response_index = selected_pair["rejected"] - 1

    if chosen_response_index == rejected_response_index:
        # judge model assigned same index for 'chosen' and 'rejected' responses.
        # sample a random index for both 'rejected' and 'chosen'.
        rnd_response_index = random.sample(range(len(responses)), 2)
        chosen_response_index = rnd_response_index[0]
        rejected_response_index = rnd_response_index[1]

    # get responses
    chosen_response = responses[chosen_response_index]
    rejected_response = responses[rejected_response_index]

    preference_sample = dict(
        prompt=prompt,
        chosen=chosen_response,
        rejected=rejected_response,
        raw=dict(
            chosen_response_key=response_keys[chosen_response_index],
            rejected_response_key=response_keys[rejected_response_index],
            explanation=selection_explanation,
            all_responses={k: sample[k] for k in response_keys},
        ),
    )

    return preference_sample


def split_dataset(dataset, splits: Dict[str, float], shuffle: bool):
    n = len(dataset)
    assert sum(list(splits.values())) == 1.0
    assert all(1.0 >= split_p > 0 for split_p in splits.values())

    if shuffle:
        index = random.sample(range(n), n)
    else:
        index = list(range(n))

    # ensure all splits have at least one sample
    dataset_splits = {split_name: [dataset[index[i]]] for i, split_name in enumerate(splits.keys())}
    index = index[len(splits) :]
    n = n - len(splits)

    i_offset = 0
    for i, (split_name, split_p) in enumerate(splits.items()):
        split_n = max(1, round(n * split_p))
        if i == len(splits) - 1:
            split_n = n - i_offset
        split_index = index[i_offset : min(i_offset + split_n, n)]
        dataset_splits[split_name] += [dataset[i] for i in split_index]
        i_offset += split_n

    assert sum([len(s) for s in dataset_splits.values()]) == len(dataset)

    return dataset_splits


def process_samples(dataset, prompt_template_config: dict):
    def convert_string_format(body, response):
        response = response.strip().strip("\n")
        body = body.strip().strip("\n")

        if len(response) == 0 or len(body) == 0:
            return "", ""

        prompt_template = UserAssistantPromptTemplate(**prompt_template_config)

        # encode only prompt (user message)
        prompt = prompt_template.format_messages({"content": body, "role": UserAssistantPromptTemplate.Role.User})

        # encode user message and assistant response (for rejected/chosen samples)
        prompt_with_response = prompt_template.format_messages(
            [
                {"content": body, "role": UserAssistantPromptTemplate.Role.User},
                {"content": response, "role": UserAssistantPromptTemplate.Role.Assistant},
            ]
        )

        # we need tomake sure eos-token is present at the end of the response
        prompt_with_response = prompt_with_response.strip()
        if not prompt_with_response.endswith(prompt_template.eos_token):
            prompt_with_response += prompt_template.eos_token

        return prompt_with_response, prompt

    chosen = [convert_string_format(x["prompt"], x["chosen"]) for x in dataset]
    rejected = [convert_string_format(x["prompt"], x["rejected"]) for x in dataset]

    samples = []
    for c, r in zip(chosen, rejected):
        if c is None or r is None:
            continue

        chosen_response, chosen_prompt = c
        rejected_response, rejected_prompt = r

        if len(chosen_response) == 0 or len(rejected_response) == 0:
            continue

        if chosen_prompt != rejected_prompt:
            continue

        comparison_dict = {
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

        samples.append(comparison_dict)

    return samples


def convert_list_of_dict_to_json(list_of_dict):
    return "\n".join(json.dumps(item) for item in list_of_dict)


def validate_output_filenames(split: str, output_dir: str, output_filename_prefix: str):
    prompts_file_name = f"{split}_prompts_with_chat_prompt.json"
    if output_filename_prefix is not None and output_filename_prefix != "":
        prompts_file_name = f"{output_filename_prefix}_{prompts_file_name}"
    assert not os.path.isfile(
        os.path.join(output_dir, prompts_file_name)
    ), f"error, file already exists: {os.path.join(output_dir, prompts_file_name)}"

    comparisons_file_name = f"{split}_comparisons_with_chat_prompt.json"
    if output_filename_prefix is not None and output_filename_prefix != "":
        comparisons_file_name = f"{output_filename_prefix}_{comparisons_file_name}"
    assert not os.path.isfile(
        os.path.join(output_dir, comparisons_file_name)
    ), f"error, file already exists: {os.path.join(output_dir, comparisons_file_name)}"


def save_dataset(dataset, split: str, output_dir: str, output_filename_prefix: str):
    prompts_to_save = convert_list_of_dict_to_json({"text": item["prompt"]} for item in dataset)
    prompts_file_name = f"{split}_prompts_with_chat_prompt"
    if output_filename_prefix is not None and output_filename_prefix != "":
        prompts_file_name = f"{output_filename_prefix}_{prompts_file_name}"
    prompts_file_name += ".jsonl"

    prompts_full_path = Path(output_dir) / prompts_file_name
    print(f"Saving {len(prompts_to_save)} prompts to {prompts_full_path}")
    with open(prompts_full_path, "w") as f:
        f.write(prompts_to_save)

    comparisons_to_save = []
    for item in dataset:
        comparisons_to_save.append({"text": item["chosen"]})
        comparisons_to_save.append({"text": item["rejected"]})

    comparisons_to_save = convert_list_of_dict_to_json(comparisons_to_save)
    comparisons_file_name = f"{split}_comparisons_with_chat_prompt"
    if output_filename_prefix is not None and output_filename_prefix != "":
        comparisons_file_name = f"{output_filename_prefix}_{comparisons_file_name}"
    comparisons_file_name += ".jsonl"

    comparisons_full_path = Path(output_dir) / comparisons_file_name
    print(f"Saving {len(comparisons_to_save)} comparisons to {comparisons_full_path}")
    with open(comparisons_full_path, "w") as f:
        f.write(comparisons_to_save)

    return prompts_full_path, comparisons_full_path


def blend_preference_datasets(files: list, output_file: str, blend_type: str):
    assert not os.path.isfile(output_file), f"Error: output file: {output_file} already exists."
    assert len(files) > 0
    assert all([os.path.isfile(f) for f in files])
    assert blend_type in ["prompts", "comparisons"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def _blend_files_type_prompt():
        # Read and combine the contents of all files
        combined_lines = []
        for file_name in files:
            with open(file_name, "r") as file:
                combined_lines.extend(file.readlines())
                if not combined_lines[-1].endswith("\n"):
                    combined_lines[-1] += "\n"

        # Shuffle the combined lines
        random.shuffle(combined_lines)

        # Write the shuffled lines to the output file
        with open(output_file, "w") as f:
            f.writelines(combined_lines)

    def _blend_files_type_comparison():
        """
        the structure of a preference dataset is:
        chosen
        rejected
        chosen
        rejected

        so we need to keep this structure, that is why we need to sample pairs (consecutive lines).
        """

        # Function to read files and group lines in pairs
        def read_and_pair_lines(file_names):
            paired_lines = []
            for file_name in file_names:
                with open(file_name, "r") as file:
                    lines = file.readlines()
                    if not lines[-1].endswith("\n"):
                        lines[-1] = lines[-1] + "\n"

                    # Assuming an even number of lines, pair them
                    for i in range(0, len(lines), 2):
                        paired_lines.append(lines[i : i + 2])

                if not paired_lines[-1][-1].endswith("\n"):
                    paired_lines[-1][-1] += "\n"

            return paired_lines

        # Read and combine the contents of all files into pairs
        combined_pairs = read_and_pair_lines(files)

        # Shuffle the combined pairs
        random.shuffle(combined_pairs)

        # Write the shuffled pairs to the output file
        with open(output_file, "w") as f:
            for pair in combined_pairs:
                f.writelines(pair)

    if blend_type == "prompts":
        _blend_files_type_prompt()
    else:
        _blend_files_type_comparison()

    print("done")


def main():
    args, inference_config, prompt_template_config = prepare_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print("Generating CAI-RLAIF candidates dataset...\n")
    dataset = generate_cai_rlaif_candidate_dataset(
        batch_size=args.batch_size,
        temperatures=np.arange(0.01, 2.01, 0.5).tolist(),
        red_teaming_dataset_path=args.red_teaming_file_path,
        inference_config=inference_config,
        prompt_template_config=prompt_template_config,
    )

    with open(os.path.join(args.output_dir, "cai_candidate_dataset.json"), "w") as file:
        json.dump(dataset, file, indent=4)

    print("\nGenerating AI preferences...\n")

    with open(args.sys_prompt_constitution_file_path, "r") as f:
        constitution_as_sys_prompt = f.read()

    preference_dataset = []
    for ds_index in tqdm(range(len(dataset))):
        sample = dataset[ds_index]

        try:
            preference = generate_ai_preference(
                sample,
                args.ngc_api_key,
                constitution_as_sys_prompt,
                seed=args.seed,
                ngc_url=args.ngc_url,
                ngc_model=args.ngc_model,
            )
        except Exception as e:
            preference = None

        if preference is not None:
            preference_dataset.append(preference)

    with open(os.path.join(args.output_dir, "cai_preference_dataset.json"), "w") as file:
        json.dump(preference_dataset, file, indent=4)

    print(
        f"\nGenerated {len(preference_dataset)} AI preferences. "
        f"Now processing and converting to chat prompt template...\n"
    )
    for split_name in args.splits:
        validate_output_filenames(
            split=split_name, output_dir=args.output_dir, output_filename_prefix=args.output_filename_prefix
        )
    ds = split_dataset(preference_dataset, args.splits, shuffle=args.shuffle)

    output_file_names = []
    for split_name, split in ds.items():
        split_samples = process_samples(split, prompt_template_config=prompt_template_config)
        prompts_path, comparisons_path = save_dataset(
            dataset=split_samples,
            split=split_name,
            output_dir=args.output_dir,
            output_filename_prefix=args.output_filename_prefix,
        )
        output_file_names.append(dict(split_name=split_name, prompts=prompts_path, comparisons=comparisons_path))

    print(f"blending preference dataset with external dataset:")
    # args.blend_with
    for split_ds in output_file_names:
        blend_split_with = args.blend_with[split_ds["split_name"]]

        for blend_type in ["prompts", "comparisons"]:
            output_file_name = f"blend_{args.blend_with['name']}_with_{os.path.basename(split_ds[blend_type])}"
            output_file_path = os.path.join(args.output_dir, output_file_name)
            blend_preference_datasets(
                files=blend_split_with[blend_type] + [split_ds[blend_type]],
                output_file=output_file_path,
                blend_type=blend_type,
            )


if __name__ == "__main__":
    main()
