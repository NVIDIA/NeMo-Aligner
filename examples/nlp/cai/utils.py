import os
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Optional, Union
import requests


def _pool_process_item(item_index: int, max_seq_length: int):
    global g_dataset

    item = g_dataset[item_index]
    item_mask = item["mask"]
    item_mask_len = item_mask.shape[0]
    need_to_remove = item_mask[: max_seq_length + 1].sum().item() == 0
    return item_index, item_mask_len, need_to_remove


def remove_long_dialogs(
    input_file_path: str,
    max_seq_length: int,
    tokenizer_model: str,
    tokenizer_library: str,
    output_dir: str,
    use_pool: bool,
):
    from tqdm import tqdm
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    assert os.path.isfile(input_file_path)
    input_file_name, input_file_extension = os.path.splitext(os.path.basename(input_file_path))

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = os.path.join(
        output_dir, f"{input_file_name}_remove_long_dialogs_max_seq_{max_seq_length}{input_file_extension}"
    )

    # load tokenizer model
    tokenizer = get_nmt_tokenizer(library=tokenizer_library, tokenizer_model=tokenizer_model)

    # create dataset object
    dataset = GPTSFTChatDataset(
        file_path=input_file_path, tokenizer=tokenizer, max_seq_length=max_seq_length, min_seq_length=1
    )

    removed_ids = set()
    length_statistics = defaultdict(int)

    if use_pool:

        def init_worker(shared_queue):
            # declare scope of a new global variable
            global g_dataset

            # store argument in the global variable for this process
            g_dataset = shared_queue

        with Pool(initializer=init_worker, initargs=(dataset,)) as pool:
            tasks = [pool.apply_async(_pool_process_item, (i, max_seq_length)) for i in range(len(dataset))]
            for task in tqdm(tasks):
                item_index, item_mask_len, need_to_remove = task.get()

                if need_to_remove:
                    removed_ids.add(item_index)
                length_statistics[item_mask_len] += 1
    else:
        for i in tqdm(range(len(dataset))):
            item_mask = dataset[i]["mask"]
            item_mask_len = item_mask.shape[0]
            need_to_remove = item_mask[: max_seq_length + 1].sum().item() == 0

            if need_to_remove:
                removed_ids.add(i)
            length_statistics[item_mask_len] += 1

    print(f"removed {(len(removed_ids) / len(tasks)) * 100:.2f}%")

    # note: we assume each sample is a single line.
    with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_name, "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            if i in removed_ids:
                continue
            o.write(line)

    return dict(
        output_file=output_file_name,
        num_removed_ids=len(removed_ids),
        removed_ids=list(removed_ids),
        length_statistics=length_statistics,
    )


def remote_inference(
    prompt: Union[List[str], str],
    port: int,
    host: str,
    temperature: Optional[float] = None,
    greedy: Optional[bool] = None,
    tokens_to_generate: Optional[int] = None,
    min_tokens_to_generate: Optional[int] = None,
    add_bos: Optional[bool] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    all_probs: Optional[bool] = None,
    repetition_penalty: Optional[float] = None,
    end_strings: Optional[Union[List[str], str]] = None,
):
    """
    @param prompt:
    @param port: The port number on which the inference service is running.
    @param host: The hostname or IP address of the inference service.
    @param temperature:
    @param greedy:
    @param tokens_to_generate:
    @param min_tokens_to_generate:
    @param add_bos:
    @param top_k:
    @param top_p:
    @param all_probs:
    @param repetition_penalty:
    @param end_strings:
    @return:
    """
    import json

    import requests

    assert port >= 0
    assert prompt is not None and isinstance(prompt, (str, list))
    if not isinstance(prompt, list):
        prompt = [prompt]
    assert all(isinstance(p, str) for p in prompt)

    if end_strings is not None:
        if not isinstance(end_strings, list):
            end_strings = [end_strings]

    def request_data(request):
        headers = {"Content-Type": "application/json"}
        resp = requests.put(f"http://{host}:{port}/generate", data=json.dumps(request), headers=headers)
        resp_json = resp.json()
        resp_sentences = resp_json["sentences"]
        return resp_sentences

    data = {
        "sentences": prompt,
    }

    if tokens_to_generate is not None:
        data["tokens_to_generate"] = tokens_to_generate
    if temperature is not None:
        data["temperature"] = temperature
    if add_bos is not None:
        data["add_BOS"] = add_bos
    if top_k is not None:
        data["top_k"] = top_k
    if top_p is not None:
        data["top_p"] = top_p
    if greedy is not None:
        data["greedy"] = greedy
    if all_probs is not None:
        data["all_probs"] = all_probs
    if repetition_penalty is not None:
        data["repetition_penalty"] = repetition_penalty
    if min_tokens_to_generate is not None:
        data["min_tokens_to_generate"] = min_tokens_to_generate
    if end_strings is not None:
        data["end_strings"] = end_strings

    sentences = request_data(data)
    return sentences


def remote_inference_with_ngc(
    api_key: str,
    prompt: str = None,
    messages: list = None,
    url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
    model: str = "mistralai/mixtral-8x7b-instruct-v0.1",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    source: https://build.nvidia.com/mistralai/mixtral-8x7b-instruct

    @param api_key:
    @param prompt:
    @param messages:
    @param url: (default: "https://integrate.api.nvidia.com/v1/chat/completions")
    @param model: (default: "mistralai/mixtral-8x7b-instruct-v0.1")
    @param temperature:
    @param top_p:
    @param max_tokens:
    @param seed:
    @return:

    examples:

    single prompt:
    remote_inference_with_ngc(api_key="<your-ngc-apu-key>",
                              prompt="calculate 3+4=?")

    a conversion:
    remote_inference_with_ngc(api_key="<your-ngc-apu-key>",
                              messages=[{"content": f"calculate 3+4=?", "role": "user"},
                                        {"content": f"3+4=8", "role": "assistant"},
                                        {"content": f"you are wrong, please correct your answer.", "role": "user"}])

    """
    assert (prompt is None) ^ (messages is None)

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
        "model": model,
        "messages": messages,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = 0.0000001 + temperature
    if top_p is not None:
        payload["top_p"] = 0.0000001 + top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed

    session = requests.Session()
    response = session.post(url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    response_message = response_body["choices"][0]["message"]["content"]
    return response_message
