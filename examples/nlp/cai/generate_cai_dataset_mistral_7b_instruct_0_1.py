import json
import os.path
import random
import tqdm
import requests


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
USER_PROMPT_START = "[INST]"
USER_PROMPT_END = "[/INST]"

PROMPT_TEMPLATE = """{USER_PROMPT_START} {prompt} {USER_PROMPT_END}"""
FEW_SHOT_SAMPLES_FILE_PATH = "/work/datasets/cai/scripts/mistral7b_few_shot_samples"
with open(FEW_SHOT_SAMPLES_FILE_PATH, 'r') as f:
    FEW_SHOT_SAMPLES = f.read().rstrip('\n')

RED_TEAMING_PROMPTS_FILE_PATH = "/work/datasets/cai/scripts/anthropic_hh_red_team_attempts/anthropic_red_team_attempts_train.jsonl"


def apply_prompt_template(p):
    return PROMPT_TEMPLATE.format(p, USER_PROMPT_START=USER_PROMPT_START, prompt=p, USER_PROMPT_END=USER_PROMPT_END)


def model_generate_response(prompt, port_num=5656):
    if not isinstance(prompt, list):
        prompt = [prompt]

    headers = {"Content-Type": "application/json"}

    def request_data(request):
        resp = requests.put('http://localhost:{}/generate'.format(port_num),
                            data=json.dumps(request),
                            headers=headers)
        resp_json = resp.json()
        resp_sentences = resp_json['sentences']
        return resp_sentences

    data = {
        "sentences": prompt,  #### [""] * batch_size,
        "tokens_to_generate": 300,
        "temperature": 1.0,
        "add_BOS": True,
        "top_k": 1,
        "top_p": 0.9,
        "greedy": True,
        "all_probs": False,
        "repetition_penalty": 1.2,
        "min_tokens_to_generate": 1,
    }

    sentences = request_data(data)
    sentences = [s + EOS_TOKEN if not s.endswith(EOS_TOKEN) else s for s in sentences]
    return sentences


def generate_cai_batch_sample(prompt_list: list, critique_list, revision_list, verbose=False):
    assert isinstance(prompt_list, list)
    if not isinstance(critique_list, list):
        critique_list = [critique_list] * len(prompt_list)
    if not isinstance(revision_list, list):
        revision_list = [revision_list] * len(prompt_list)

    num_prompts = len(prompt_list)
    assert len(critique_list) == num_prompts
    assert len(revision_list) == num_prompts

    # get initial response
    initial_prompt_batch = [FEW_SHOT_SAMPLES + " " + apply_prompt_template(p) for p in prompt_list]
    chat_batch = model_generate_response(initial_prompt_batch)
    assert len(chat_batch) == num_prompts
    initial_prompt_response_batch = [chat.rsplit(USER_PROMPT_END, 1)[-1].strip().removesuffix(EOS_TOKEN) for chat in chat_batch]

    if verbose:
        print("===== initial prompt + response ====")
        print(chat_batch[0])

    # generate a single critique
    critique_request_prompt_batch = [chat_batch[i] + " " + apply_prompt_template(cr_p) for i, cr_p in enumerate(critique_list)]
    chat_batch = model_generate_response(critique_request_prompt_batch)
    assert len(chat_batch) == num_prompts
    critique_response_batch = [chat.rsplit(USER_PROMPT_END, 1)[-1].strip().removesuffix(EOS_TOKEN) for chat in chat_batch]

    if verbose:
        print("===== initial prompt + response + critique prompt + response ====")
        print(chat_batch[0])

    revision_request_prompt_batch = [chat_batch[i] + " " + apply_prompt_template(rev_p) for i, rev_p in enumerate(revision_list)]
    chat_batch = model_generate_response(revision_request_prompt_batch)
    assert len(chat_batch) == num_prompts
    revision_response_batch = [chat.rsplit(USER_PROMPT_END, 1)[-1].strip().removesuffix(EOS_TOKEN) for chat in chat_batch]

    if verbose:
        print("===== initial prompt + response + critique prompt + response + revision request + response ====")
        print(chat_batch[0])

    s_batch = []
    for i in range(num_prompts):
        s = dict(initial_prompt=prompt_list[i],
                 initial_response=initial_prompt_response_batch[i],
                 critic_prompt=critique_list[i],
                 critic_response=critique_response_batch[i],
                 revision_prompt=revision_list[i],
                 revision_response=revision_response_batch[i])
        s_batch.append(s)

    return s_batch


def load_critique_revision_instructions():
    file_path = "CritiqueRevisionInstructions.json"
    instructions = []
    with open(file_path, 'r') as file:
        json_object = json.load(file)
        for i in range(len(json_object)):
            inst = json_object[f"harmful{i}"]
            assert len(inst) == 2 and 'prompt' in inst and 'edit_request' in inst
            assert len(inst['prompt']) == 1

            critique_prompt = inst['prompt'][0]
            revision_prompt = inst['edit_request']

            instructions.append(dict(critique_prompt=critique_prompt, revision_prompt=revision_prompt))

    return instructions


def get_red_team_train_human_prompts() -> list:

    def strip_first_red_team_prompt(data_item):
        human_index = data_item['transcript'].find("Human:")
        assistant_index = data_item['transcript'].find("Assistant:")
        assert human_index >= 0
        assert assistant_index >= 0 and assistant_index > human_index

        human_index += len("Human:")
        prompt = data_item['transcript'][human_index:assistant_index]
        prompt = prompt.strip()
        return prompt

    red_teaming_prompts = []
    with open(RED_TEAMING_PROMPTS_FILE_PATH, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            red_teaming_prompt = strip_first_red_team_prompt(json_object)
            red_teaming_prompts.append(red_teaming_prompt)

    return red_teaming_prompts


def generate_cai_dataset(batch_size: int, save_to_file_interval: int , save_file_path: str):
    """
    @param batch_size: inference batch size
    @param save_to_file_interval: saves generated samples to the disk after 'save_to_file_interval' batches.
    @param save_file_path:
    @return:
    """
    assert batch_size > 0
    assert save_to_file_interval > 0
    assert save_file_path is not None and save_file_path != ''
    assert not os.path.exists(save_file_path)

    # load constitution critique/revision instructions
    critique_revision_instructions_set = load_critique_revision_instructions()

    def sample_random_critique_revision_set():
        a_critique_list = []
        a_revision_list = []
        for i in range(batch_size):
            rnd_index = random.randint(0, len(critique_revision_instructions_set) - 1)
            a_critique_list.append(critique_revision_instructions_set[rnd_index]['critique_prompt'])
            a_revision_list.append(critique_revision_instructions_set[rnd_index]['revision_prompt'])
        return a_critique_list, a_revision_list

    red_teaming_prompts = get_red_team_train_human_prompts()
    cai_samples = []
    num_batches = 0
    for index in tqdm.tqdm(range(0, len(red_teaming_prompts), batch_size)):
        red_teaming_prompts_list = red_teaming_prompts[index:index+batch_size]
        if len(red_teaming_prompts_list) < batch_size:
            break

        # sample random critique/revision instruction set
        critique_list, revision_list = sample_random_critique_revision_set()
        critique_list = [c.removesuffix("\n\nCritique:").removeprefix("\n\nCritiqueRequest: ") for c in critique_list]
        revision_list = [c.removesuffix("\n\nRevision:").removeprefix("\n\nRevisionRequest: ") for c in revision_list]

        # call model
        cai_batch_samples = generate_cai_batch_sample(red_teaming_prompts_list,
                                                      critique_list=critique_list,
                                                      revision_list=revision_list,
                                                      verbose=False)
        cai_samples.extend(cai_batch_samples)

        num_batches += 1

        if num_batches % save_to_file_interval == 0:
            with open(save_file_path, 'w') as file:
                json.dump(cai_samples, file)

    print('done')


if __name__ == '__main__':
    generate_cai_dataset(batch_size=32,
                         save_to_file_interval=1,
                         save_file_path='cai_samples_from_anthropic_red_team_attempts_train.json')

    print('done')
