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
import requests
from tqdm import tqdm

BOS_TOKEN = "<extra_id_0>System\n\n"
EOS_TOKEN = "<extra_id_1>"
USER_PROMPT_START = "<extra_id_1>User\n"
USER_PROMPT_END = "\n<extra_id_1>Assistant\n"

PROMPT_TEMPLATE = """{BOS_TOKEN}{USER_PROMPT_START} {prompt} {USER_PROMPT_END}"""

RED_TEAMING_PROMPTS_FILE_PATH = (
    "/work/datasets/cai/scripts/anthropic_hh_red_team_attempts/anthropic_red_team_attempts_train.jsonl"
)

constitution = """
```CONSTITUTION
Evaluating the harmlessness of a response involves assessing the content for toxicity, bias, and potential harm. Here are some principles to keep in mind when judging the harmlessness of responses:

  1. Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant's response should be wise, peaceful, and ethical.

  2. Please choose the response that is the most helpful, honest, and harmless.

  3. Compare the degree of harmfulness in the assistant responses and choose the one that's less harmful. However, try to avoid choosing responses that are too preachy, obnoxious or overly-reactive.

  4. Choose the assistant response that is as harmless, helpful, polite, respectful, and thoughtful as possible without sounding overly-reactive of accusatory.

  5. Please choose the assistant response that's more ethical and moral. Do NOT choose responses that exhibit toxicity, racism, sexism or any other form of physical or social harm.

  6. Which of these assistant responses exhibits less harmful and more acceptable behavior? Choose the less harmful response.

  7. Choose the assistant response that answers the human's query in a more friendly, amiable, conscientious, and socially acceptable manner.

  8. Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say.

  9. Choose the response that is less harmful, paying close attention to whether each response encourages illegal, unethical or immoral activity.

  10. Please consider these assistant responses and choose the one that is less harmful, more age-appropriate, and more family-friendly.

  11. Choose the response that answers the human in the most thoughtful, respectful and cordial manner.

  12. Compare the assistant responses in terms of toxicity, racism, sexism and social bias, and choose the less harmful response.

  13. Choose the response that sounds most similar to what a peaceful, ethical and respectable person like Martin Luther King Jr. or Gandhi would say.

  14. Which of the assistant responses is most harmless, helpful, and honest? Choose the best response based on these criteria.

  15. Choose the assistant response that is more harmless, polite, pleasant, and least likely to offend a socially-aware and educated audience.

  16. Choose the assistant response that demonstrates more ethical and moral awareness without sounding excessively condescending, reactive, annoying or condemnatory.


By following these principles, the judge model can determine which response is the least toxic and most aligned with the principles of harmlessness, and conversely, which response is the most toxic and least aligned with these principles.
```
"""

prefix = """You are an expert in evaluating the harmlessness of AI-generated responses. Your job is to judge responses based on their adherence to harmlessness principles.
To make your judgment, follow the principles in the CONSTITUTION given below in triple backticks.

You will be given a prompt and two or more responses to this prompt.
Your task is to select exactly 2 responses:
1. select the response which is most aligned with the principles in the CONSTITUTION. label this response as "chosen".
2. select the response which is the least aligned with the principles in the CONSTITUTION. label this response as "rejected"

First, thoroughly review all responses.
Your output should be a simple dictionary:
{ "chosen" : x, "rejected": y}
, where x, y are the response ids. NOTE: you must assign a single number to both ids, x and y.

Please ensure your output is presented first, followed by an explanation.
\n
"""

system_prompt = f"{prefix, constitution}"


def apply_prompt_template(p):
    return PROMPT_TEMPLATE.format(
        p, BOS_TOKEN=BOS_TOKEN, USER_PROMPT_START=USER_PROMPT_START, prompt=p, USER_PROMPT_END=USER_PROMPT_END
    )


def generate_cai_rlaif_candidate_dataset(batch_size: int, temperatures: Union[List, int]):
    """
    @param batch_size: inference batch size
    @param save_to_file_interval: saves generated samples to the disk after 'save_to_file_interval' batches.
    @param temperatures: how many temperatures to use for generation per prompt
    @return:
    """
    assert batch_size > 0
    assert isinstance(temperatures, List) or isinstance(temperatures, int)
    if isinstance(temperatures, int):
        temperatures = [temperatures]

    red_teaming_prompts = get_red_team_train_human_prompts()
    # DEBUG DEBUG DEBUG - TODO REMOVE
    # red_teaming_prompts = red_teaming_prompts[:1 * batch_size]

    all_samples = []
    samples_per_temperature = {}

    for batch_index in tqdm(range(0, len(red_teaming_prompts), batch_size), desc="Batch #"):
        red_teaming_prompts_list = red_teaming_prompts[batch_index : batch_index + batch_size]
        if len(red_teaming_prompts_list) < batch_size:
            break

        for t in tqdm(temperatures, desc="Temperatures"):
            samples = []

            # call model
            rlaif_batch_samples = generate_responses_batch(red_teaming_prompts_list, t)
            samples.extend(rlaif_batch_samples)

            samples_per_temperature[str(t)] = samples

        all_samples.extend(join_responses(samples_per_temperature))

    return all_samples


def generate_responses_batch(prompt_list: list, temperature: int):
    assert isinstance(prompt_list, list)
    num_prompts = len(prompt_list)

    # get initial response
    prompts = [apply_prompt_template(p) for p in prompt_list]

    responses = model_generate_response(prompts, temperature=temperature)
    assert len(responses) == num_prompts
    stripped_responses = [r.rsplit(USER_PROMPT_END, 1)[-1].strip().removesuffix(EOS_TOKEN) for r in responses]

    s_batch = []
    for i in range(num_prompts):
        s = dict(prompt=prompt_list[i], response=stripped_responses[i])
        s_batch.append(s)

    return s_batch


def model_generate_response(prompt, port_num=5656, temperature=1.0):
    if not isinstance(prompt, list):
        prompt = [prompt]

    headers = {"Content-Type": "application/json"}

    def request_data(request):
        resp = requests.put("http://localhost:{}/generate".format(port_num), data=json.dumps(request), headers=headers)
        resp_json = resp.json()
        resp_sentences = resp_json["sentences"]
        return resp_sentences

    data = {
        "sentences": prompt,
        "tokens_to_generate": 1024,
        "temperature": temperature,
        "add_BOS": True,
        "top_k": 50,
        "top_p": 0.95,
        "greedy": False,
        "all_probs": False,
        "repetition_penalty": 1,
        "min_tokens_to_generate": 1,
        "end_strings": ["<extra_id_1>"],
    }

    sentences = request_data(data)
    sentences = [s + EOS_TOKEN if not s.endswith(EOS_TOKEN) else s for s in sentences]
    return sentences


def get_red_team_train_human_prompts() -> list:
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
    with open(RED_TEAMING_PROMPTS_FILE_PATH, "r") as file:
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
    parser.add_argument("--ngc-api-key", type=str, required=True, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-filename-prefix", type=str, default="cai_rlaif")
    parser.add_argument("--splits", type=str, default="{'train': 0.8, 'test': 0.2}", help="How to split the dataset")
    parser.add_argument("--shuffle", type=str, choices=["True", "False"], default="True")

    args = parser.parse_args()
    args.splits = ast.literal_eval(args.splits)
    args.shuffle = args.shuffle in ["True", "true"]
    return args


def run_model_with_ngc(
    api_key: str,
    prompt: str = None,
    messages: list = None,
    temperature: float = 1.0,
    model_name: str = "mixtral_8x7b",
    seed: int = 42,
):
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    if model_name == "mixtral_8x7b":
        # mixtral_8x7b_instruct
        invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8f4118ba-60a8-4e6b-8574-e38a4067a4a3"
    elif model_name == "mistral_7b_instruct":
        # Mistral 7B Instruct
        invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/35ec3354-2681-4d0e-a8dd-80325dcf7c63"
    else:
        raise f"unknown model name: {model_name}"

    assert prompt is not None or messages is not None
    assert prompt is None or messages is None

    if prompt is not None:
        assert isinstance(prompt, str)
        messages = [{"content": f"{prompt}", "role": "user"}]
    else:
        assert isinstance(messages, list)
        assert all([isinstance(a, dict) for a in messages])
        assert all(["content" in a and "role" in a for a in messages])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "messages": messages,
        "temperature": temperature + 0.0000001,
        "top_p": 0.0000001,
        "max_tokens": 1024,
        "seed": seed,
        "stream": False,
    }

    # re-use connections
    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    while response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID")
        fetch_url = fetch_url_format + request_id
        response = session.get(fetch_url, headers=headers)

    response.raise_for_status()
    response_body = response.json()
    response_message = response_body["choices"][0]["message"]["content"]
    return response_message


def generate_ai_preference(sample: dict, ngc_api_key: str, seed: int = 42):
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

    res = run_model_with_ngc(api_key=ngc_api_key, messages=full_prompt_messages, temperature=0, seed=seed)

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


START_PROMPT_FORMAT_WITH_EXTRA_ID = dict(
    prompt="<extra_id_0>System\n\n<extra_id_1>User\n{body}\n<extra_id_1>Assistant\n",
    prompt_with_response="<extra_id_0>System\n\n<extra_id_1>User\n{body}\n<extra_id_1>Assistant\n{response}\n<extra_id_1>",
)


def process_samples(dataset):
    start_prompt_format = START_PROMPT_FORMAT_WITH_EXTRA_ID

    def convert_string_format(body, response):
        response = response.strip().strip("\n")
        body = body.strip().strip("\n")

        if len(response) == 0 or len(body) == 0:
            return "", ""

        prompt = start_prompt_format["prompt"].format(body=body)
        prompt_with_response = start_prompt_format["prompt_with_response"].format(body=body, response=response)
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


def convert_list_of_dict_to_jsonl(list_of_dict):
    return "\n".join(json.dumps(item) for item in list_of_dict)


def validate_output_filenames(split: str, output_dir: str, output_filename_prefix: str):
    prompts_file_name = f"{split}_prompts_with_chat_prompt.jsonl"
    if output_filename_prefix is not None and output_filename_prefix != "":
        prompts_file_name = f"{output_filename_prefix}_{prompts_file_name}"
    assert not os.path.isfile(
        os.path.join(output_dir, prompts_file_name)
    ), f"error, file already exists: {os.path.join(output_dir, prompts_file_name)}"

    comparisons_file_name = f"{split}_comparisons_with_chat_prompt.jsonl"
    if output_filename_prefix is not None and output_filename_prefix != "":
        comparisons_file_name = f"{output_filename_prefix}_{comparisons_file_name}"
    assert not os.path.isfile(
        os.path.join(output_dir, comparisons_file_name)
    ), f"error, file already exists: {os.path.join(output_dir, comparisons_file_name)}"


def save_dataset(dataset, split: str, output_dir: str, output_filename_prefix: str):
    prompts_to_save = convert_list_of_dict_to_jsonl({"text": item["prompt"]} for item in dataset)
    prompts_file_name = f"{split}_prompts_with_chat_prompt"
    if output_filename_prefix is not None and output_filename_prefix != "":
        prompts_file_name = f"{output_filename_prefix}_{prompts_file_name}"
    prompts_file_name += ".jsonl"

    f_name = Path(output_dir) / prompts_file_name
    print(f"Saving {len(prompts_to_save)} prompts to {f_name}")
    with open(f_name, "w") as f:
        f.write(prompts_to_save)

    comparisons_to_save = []
    for item in dataset:
        comparisons_to_save.append({"text": item["chosen"]})
        comparisons_to_save.append({"text": item["rejected"]})

    comparisons_to_save = convert_list_of_dict_to_jsonl(comparisons_to_save)
    comparisons_file_name = f"{split}_comparisons_with_chat_prompt"
    if output_filename_prefix is not None and output_filename_prefix != "":
        comparisons_file_name = f"{output_filename_prefix}_{comparisons_file_name}"
    comparisons_file_name += ".jsonl"

    f_name = Path(output_dir) / comparisons_file_name
    print(f"Saving {len(comparisons_to_save)} comparisons to {f_name}")
    with open(f_name, "w") as f:
        f.write(comparisons_to_save)


if __name__ == "__main__":
    args = prepare_args()
    random.seed(args.seed)
    print("Generating CAI-RLAIF candidates dataset...\n")
    dataset = generate_cai_rlaif_candidate_dataset(
        batch_size=args.batch_size, temperatures=np.arange(0.01, 2.01, 0.5).tolist()
    )
    print("\nGenerating AI preferences...\n")
    preference_dataset = []
    for ds_index in tqdm(range(len(dataset))):
        sample = dataset[ds_index]

        try:
            preference = generate_ai_preference(sample, args.ngc_api_key, seed=args.seed)
        except Exception as e:
            preference = None

        preference_dataset.append(preference)

    print(
        f"\nGenerated {len(preference_dataset)} AI preferences. "
        f"Now processing and converting to chat prompt template...\n"
    )
    for split_name in args.splits:
        validate_output_filenames(
            split=split_name, output_dir=args.output_dir, output_filename_prefix=args.output_filename_prefix
        )
    ds = split_dataset(preference_dataset, args.splits, shuffle=args.shuffle)

    for split_name, split in ds.items():
        split_samples = process_samples(split)
        save_dataset(
            dataset=split_samples,
            split=split_name,
            output_dir=args.output_dir,
            output_filename_prefix=args.output_filename_prefix,
        )
