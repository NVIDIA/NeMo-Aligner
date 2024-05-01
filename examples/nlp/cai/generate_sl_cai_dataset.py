import argparse
import json
import os
import os.path
import random
from typing import Optional

import numpy as np
import requests
import tqdm
from utils import remove_long_dialogs


class MistralInstructChatTemplate:
    bos_token = "<s>"
    eos_token = "</s>"
    user_prompt_start = "[INST]"
    user_prompt_end = "[/INST]"
    prompt_template = """{BOS_TOKEN}{USER_PROMPT_START} {prompt} {USER_PROMPT_END}"""

    @staticmethod
    def apply_prompt_template(prompt):
        return MistralInstructChatTemplate.prompt_template.format(
            BOS_TOKEN=MistralInstructChatTemplate.bos_token,
            USER_PROMPT_START=MistralInstructChatTemplate.user_prompt_start,
            prompt=prompt,
            USER_PROMPT_END=MistralInstructChatTemplate.user_prompt_end,
        )

    @staticmethod
    def extract_response(prompt: str):
        response = prompt.rsplit(MistralInstructChatTemplate.user_prompt_end, 1)[-1]
        response = response.strip().removesuffix(MistralInstructChatTemplate.eos_token)
        return response


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


def model_remote_inference(
    prompt,
    port_num: int,
    host: str,
    temperature: Optional[float] = 1.0,
    greedy: Optional[bool] = True,
    tokens_to_generate: Optional[int] = 1024,
):
    """
    @param prompt:
    @param port_num: The port number on which the inference service is running.
    @param host: The hostname or IP address of the inference service.
    @param temperature:
    @param greedy:
    @param tokens_to_generate:
    @return:
    """
    assert port_num >= 0
    assert prompt is not None and isinstance(prompt, (str, list))
    if not isinstance(prompt, list):
        prompt = [prompt]
    assert all(isinstance(p, str) for p in prompt)

    headers = {"Content-Type": "application/json"}

    def request_data(request):
        resp = requests.put(f"http://{host}:{port_num}/generate", data=json.dumps(request), headers=headers)
        resp_json = resp.json()
        resp_sentences = resp_json["sentences"]
        return resp_sentences

    data = {
        "sentences": prompt,
        "add_BOS": True,
        "top_k": 1,
        "top_p": 0.9,
        "all_probs": False,
        "repetition_penalty": 1.2,
        "min_tokens_to_generate": 1,
    }

    if temperature is not None:
        data["temperature"] = temperature
    if greedy is not None:
        data["greedy"] = greedy
    if tokens_to_generate is not None:
        data["tokens_to_generate"] = tokens_to_generate

    sentences = request_data(data)
    sentences = [
        s + MistralInstructChatTemplate.eos_token if not s.endswith(MistralInstructChatTemplate.eos_token) else s
        for s in sentences
    ]

    return sentences


def generate_cai_batch_sample(
    prompt_list: list, few_shot_dataset_path: str, critique_list, revision_list, port_num: int, host: str
):
    assert isinstance(prompt_list, list)
    if not isinstance(critique_list, list):
        critique_list = [critique_list] * len(prompt_list)
    if not isinstance(revision_list, list):
        revision_list = [revision_list] * len(prompt_list)

    num_prompts = len(prompt_list)
    assert len(critique_list) == num_prompts
    assert len(revision_list) == num_prompts

    with open(few_shot_dataset_path, "r") as f:
        few_shot_samples = f.read().rstrip("\n")

    # get initial response
    print(f"\nGenerating initial response for {num_prompts} prompts...")
    initial_prompt_batch = [
        " ".join([few_shot_samples, MistralInstructChatTemplate.apply_prompt_template(p)]) for p in prompt_list
    ]
    chat_batch = model_remote_inference(initial_prompt_batch, port_num=port_num, host=host)
    assert len(chat_batch) == num_prompts
    initial_response_batch = [MistralInstructChatTemplate.extract_response(chat) for chat in chat_batch]
    print("Done")

    # generate a single critique
    print(f"\nGenerating a critique for {num_prompts} initial responses...")
    critique_request_batch = [
        " ".join([chat_batch[i], MistralInstructChatTemplate.apply_prompt_template(cr_p)])
        for i, cr_p in enumerate(critique_list)
    ]
    chat_batch = model_remote_inference(critique_request_batch, port_num=port_num, host=host)
    assert len(chat_batch) == num_prompts
    critique_response_batch = [MistralInstructChatTemplate.extract_response(chat) for chat in chat_batch]
    print("Done")

    # generate a single revision
    print(f"\nGenerating {num_prompts} revisions ...")
    revision_request_prompt_batch = [
        " ".join([chat_batch[i], MistralInstructChatTemplate.apply_prompt_template(rev_p)])
        for i, rev_p in enumerate(revision_list)
    ]
    chat_batch = model_remote_inference(revision_request_prompt_batch, port_num=port_num, host=host)
    assert len(chat_batch) == num_prompts
    revision_response_batch = [MistralInstructChatTemplate.extract_response(chat) for chat in chat_batch]
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
    port_num: int,
    host: str
):
    """
    @param batch_size: inference batch size
    @param save_to_file_interval: saves generated samples to the disk after 'save_to_file_interval' batches.
    @param save_file_path:
    @param red_teaming_dataset_path:
    @param few_shot_samples_filepath:
    @param critique_revision_instructions_filepath:
    @param num_examples:
    @param port_num: The port number on which the inference service is running.
    @param host: The hostname or IP address of the inference service.
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
            port_num=port_num,
            host=host
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
    parser.add_argument("--port-num", type=int, default=5656,
                        help="The port number on which the inference service is running")
    parser.add_argument("--host", type=str, default="localhost",
                        help="The hostname or IP address of the inference service")

    args = parser.parse_args()
    assert os.path.isfile(args.red_teaming_prompts_dataset_path)
    assert os.path.isfile(args.few_shot_prompts_dataset_path)
    assert os.path.isfile(args.helpfulness_dataset_path)
    assert args.num_examples >= -1 and args.num_examples != 0

    if args.max_seq_length is not None:
        assert args.max_seq_length > 0
        assert os.path.isfile(args.tokenizer_model)
        assert args.tokenizer_library is not None

    return args


def main():
    args = prepare_args()
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
        port_num=args.port_num,
        host=args.host
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
