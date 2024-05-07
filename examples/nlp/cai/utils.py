import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

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
    assert isinstance(prompt, (str, list))
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
    url: str,
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    This function is designed to interact with NVIDIA's GPU Cloud (NGC) to utilize a specific model hosted on the platform.
    The function requires two main arguments: the NGC API key and the name of the model you wish to use.

    @param api_key: Your NGC API key.
    This key is necessary for authentication and to gain access to the models hosted on NGC.
    You can generate an API key by following the instructions provided in the NGC documentation.
    @param url: e.g., https://integrate.api.nvidia.com/v1/chat/completions
    @param model: The name of the model you want to access.
    This should be the full name of the model as registered in NGC, such as "mistralai/mixtral-8x7b-instruct-v0.1".
    The model name is used to specify which particular model you intend to interact with within the NGC repository.
    @param prompt:
    @param messages:
    @param temperature:
    @param top_p:
    @param max_tokens:
    @param seed:
    @return:

    Here is an example of how to call the Mixtral-8x7B model:
    https://build.nvidia.com/mistralai/mixtral-8x7b-instruct


    Examples:

    1. single prompt:
    remote_inference_with_ngc(api_key="<your-ngc-apu-key>",
                              url="https://integrate.api.nvidia.com/v1/chat/completions",
                              model: str = "mistralai/mixtral-8x7b-instruct-v0.1",
                              prompt="calculate 3+4=?")

    2. a conversion prompt:
    remote_inference_with_ngc(api_key="<your-ngc-apu-key>",
                              url="https://integrate.api.nvidia.com/v1/chat/completions",
                              model: str = "mistralai/mixtral-8x7b-instruct-v0.1",
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


class PromptTemplate:
    def __init__(
        self,
        role_message_format: Dict[str, str],
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        response_extract_pattern: Optional[str] = None,
    ):
        """

        @param role_message_format: dict of (key:role-name, value:role-message-format).
        example (for mistraL): dict(User="[INST] {MESSAGE} [/INST]", Assistant="{MESSAGE}</s> ")
        @param bos_token: begin-of-sequence token (optional)
        @param eos_token: end-of-sequence token (optional)
        @response_extract_pattern: (optional)

        example: Mistral

        mistral_prompt_template = PromptTemplate(
            dict(
                User="[INST] {MESSAGE} [/INST]",
                Assistant="{MESSAGE}</s> "
            ),
            bos_token="<s>",
            eos_token="</s>",
            response_extract_pattern="[/INST]"
        )

        # single message (e.g., user prompt)
        prompt = mistral_prompt_template.format_message(
            {"role": "User", "content": "Calculate the sum of 2 and 3."}
        )

        # a conversation
        prompt = mistral_prompt_template.format_message(
            [
                {"role": "User", "content": "Calculate the sum of 2 and 3."},
                {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
                {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
            ]
        )
        """
        assert role_message_format is not None and isinstance(role_message_format, dict)
        assert all(
            PromptTemplate.is_valid_role_message_template(message_template)
            for message_template in role_message_format.values()
        )

        self.role_message_template = role_message_format.copy()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.response_extract_pattern = response_extract_pattern

    def format_messages(self, messages: Union[List[dict], dict]):
        assert isinstance(messages, (list, dict))
        if not isinstance(messages, list):
            messages = [messages]

        assert all([isinstance(m, dict) for m in messages])
        assert all(["content" in m and "role" in m for m in messages])
        assert all([m["role"] in self.role_message_template for m in messages])

        for i in range(1, len(messages)):
            assert messages[i]["role"] != messages[i - 1]["role"]

        prompt = self.bos_token if self.bos_token is not None else ""

        for m in messages:
            role = m["role"]
            content = m["content"]
            message = self.role_message_template[role].format(MESSAGE=content)
            prompt += message

        return prompt

    def extract_response(self, message: str):
        assert self.response_extract_pattern is not None
        message = message.strip()

        # Find the last occurrence of the pattern
        last_occurrence = message.rfind(self.response_extract_pattern)

        response = None
        if last_occurrence >= 0:
            index_after_pattern = last_occurrence + len(self.response_extract_pattern)
            response = message[index_after_pattern:]
            response = response.strip()

            # Find eos-token
            if self.eos_token is not None:
                eos_token_index = response.rfind(self.eos_token)
                if eos_token_index >= 0:
                    response = response[:eos_token_index]

        return response

    @staticmethod
    def is_valid_role_message_template(message_template: str):
        # checks for the presence of exactly one "{MESSAGE}" placeholder in the template
        return message_template.count("{MESSAGE}") == 1


class UserAssistantPromptTemplate(PromptTemplate):
    _user_role_name = "User"
    _assistant_role_name = "Assistant"
    _system_role_name = "System"

    @property
    def user_role_name(self):
        return UserAssistantPromptTemplate._user_role_name

    @property
    def assistant_role_name(self):
        return UserAssistantPromptTemplate._assistant_role_name

    @property
    def system_role_name(self):
        return UserAssistantPromptTemplate._system_role_name

    def __init__(
        self,
        user_format: str,
        assistant_format: str,
        system_format: Optional[str] = None,
        system_default_message: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        response_extract_pattern: Optional[str] = None,
    ):
        """
        @param user_format: user message format.
        @param assistant_format: assistant message format.
        @param system_format: system message format.
        @param system_default_message: system default message
        @param bos_token: begin-of-sequence token (optional)
        @param eos_token: end-of-sequence token (optional)
        @param response_extract_pattern: (optional)


        example: Mistral

        user_assistant_format = UserAssistantPromptTemplate(
            user_format="[INST] {MESSAGE} [/INST]",
            assistant_format="{MESSAGE}</s> ",
            bos_token="<s>",
            eos_token="</s>",
            response_extract_pattern="[/INST]"
        )
        """

        role_message_format = {self.user_role_name: user_format, self.assistant_role_name: assistant_format}

        # optionally, add system message format
        if system_format is not None:
            role_message_format[self.system_role_name] = system_format
        else:
            assert system_default_message is None

        super().__init__(
            role_message_format,
            bos_token=bos_token,
            eos_token=eos_token,
            response_extract_pattern=response_extract_pattern,
        )

        self.system_default_message = system_default_message

    def has_system_role(self):
        return self.system_role_name in self.role_message_template

    def format_messages(self, messages: Union[List[dict], dict]):
        assert messages is not None
        assert isinstance(messages, (list, dict))
        if not isinstance(messages, list):
            messages = [messages]

        if self.system_default_message is not None and self.has_system_role():
            # NOTE: It is assumed that if a system message exists, it should be the first message.
            if messages[0]["role"] != self.system_role_name:
                messages = [{"content": self.system_default_message, "role": self.system_role_name}] + messages

        return super().format_messages(messages)


if __name__ == "__main__":
    extra_id_prompt_template = PromptTemplate(
        dict(
            System="<extra_id_0>System\n{MESSAGE}\n",
            User="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
            Assistant="{MESSAGE}\n",
        ),
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )

    m11 = extra_id_prompt_template.format_messages(
        [{"role": "System", "content": ""}, {"role": "User", "content": "Calculate the sum of 2 and 3."}]
    )
    print(f"{'-' * 20}\n{m11}\n{'-' * 20}")

    m12 = extra_id_prompt_template.format_messages(
        [
            {"role": "System", "content": ""},
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        ]
    )
    print(f"{'-' * 20}\n{m12}\n{'-' * 20}")

    m13 = extra_id_prompt_template.format_messages(
        [
            {"role": "System", "content": ""},
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
            {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
        ]
    )
    print(f"{'-' * 20}\n{m13}\n{'-' * 20}")

    extract_m12 = extra_id_prompt_template.extract_response(m12)
    print(f"{'-' * 20}\n{extract_m12}\n{'-' * 20}")

    # mistral example:
    mistral_prompt_template = PromptTemplate(
        dict(User="[INST] {MESSAGE} [/INST]", Assistant="{MESSAGE}</s> "),
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )

    m21 = mistral_prompt_template.format_messages({"role": "User", "content": "Calculate the sum of 2 and 3."})
    print(f"{'-' * 20}\n{m21}\n{'-' * 20}")

    m22 = mistral_prompt_template.format_messages(
        [
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        ]
    )
    print(f"{'-' * 20}\n{m22}\n{'-' * 20}")

    m23 = mistral_prompt_template.format_messages(
        [
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
            {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
        ]
    )
    print(f"{'-' * 20}\n{m23}\n{'-' * 20}")

    extract_m22 = mistral_prompt_template.extract_response(m22)
    print(f"{'-' * 20}\n{extract_m22}\n{'-' * 20}")

    # example for using 'UserAssistantPromptTemplate'
    extra_id_user_assistant_format = UserAssistantPromptTemplate(
        user_format="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
        assistant_format="{MESSAGE}\n",
        system_format="<extra_id_0>System\n{MESSAGE}\n",
        system_default_message="",
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )

    m31 = extra_id_user_assistant_format.format_messages({"role": "User", "content": "Calculate the sum of 2 and 3."})
    print(f"{'-' * 20}\n{m31}\n{'-' * 20}")

    mistral_user_assistant_format = UserAssistantPromptTemplate(
        user_format="[INST] {MESSAGE} [/INST]",
        assistant_format="{MESSAGE}</s> ",
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )

    m41 = mistral_user_assistant_format.format_messages({"role": "User", "content": "Calculate the sum of 2 and 3."})
    print(f"{'-' * 20}\n{m41}\n{'-' * 20}")
