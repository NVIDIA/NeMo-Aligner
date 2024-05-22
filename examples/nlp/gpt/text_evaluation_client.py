import functools
import itertools
import json
import os
import pathlib
import sys
import traceback
from typing import Any, Callable, Optional

import amqp
import fire
import numpy as np
from celery import Celery
from hydra import TaskFunction
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm


def get_text_generation_for_batch(url, backend_url):
    app = Celery("tasks", backend=f"{backend_url}", broker=f"{url}")

    # CELERY_ACKS_LATE = True
    app.conf.task_acks_late = True
    app.conf.worker_deduplicate_successful_tasks = True
    app.conf.worker_prefetch_multiplier = 1

    # 5 hrs timeout
    app.conf.update(broker_transport_options={"visibility_timeout": 18000})

    @app.task
    def text_generation_batch(batch):
        pass

    return text_generation_batch


special_tokens = {
    "system_turn_start": "<extra_id_0>",
    "end_of_name": "\n",
    "end_of_turn": "\n",
    "turn_start": "<extra_id_1>",
    "label_start": "<extra_id_2>",
}


def get_prompt(system_turn, prompt_turns):
    prompt = f"{special_tokens['system_turn_start']}System{special_tokens['end_of_name']}"
    prompt += f"{system_turn}{special_tokens['end_of_turn']}"
    for turn in prompt_turns:
        prompt += f"{special_tokens['turn_start']}{turn['from']}{special_tokens['end_of_name']}"
        prompt += f"{turn['value']}{special_tokens['end_of_turn']}"
    return prompt


def chunk_work(results, tmp_file_path, key_map, keys, chunk_id, micro_batch_size=1):
    global_pbar = tqdm(total=len(results), desc=f"Search Global Progress {chunk_id}")
    total_time = 0
    total_finished = 0
    with open(tmp_file_path, "a", encoding="utf-8") as f:
        while len(results) > 0:
            for subtask in results:  # Iterate over a copy of the list
                try:
                    if subtask.ready():
                        args = subtask.get()
                        time_spent = args["time"]
                        data_ids = args["data_ids"]
                        total_time += time_spent
                        total_finished += micro_batch_size
                        global_pbar.update(micro_batch_size)
                        global_pbar.write(
                            f"Finished in {time_spent} seconds, total_time: {total_time}, average time: {total_time / total_finished}"
                        )
                        for data_id, logp in zip(data_ids, args["sft_logprob"]):
                            key = (data_id[0], data_id[1])
                            record = key_map[key]
                            record["log(P(y|x))"] = logp
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()
                            keys[key] = record
                        results.remove(subtask)
                        # results.children.remove(subtask)  # Remove the subtask from the list
                except TimeoutError:
                    global_pbar.write("Timeout")
                    traceback.print_exc()
                    results.remove(subtask)
                except amqp.exceptions.PreconditionFailed:
                    global_pbar.write("RabbitMQ connection failed")
                    traceback.print_exc()
                    results.remove(subtask)
                except Exception as e:
                    global_pbar.write(f"Exception: {e}")
                    traceback.print_exc()
                    results.remove(subtask)


def main(
    micro_batch_size=1,
    input_file="input.jsonl",
    server_url="amqp://",
    backend_url="rpc://",
    output_file="output.jsonl",
    use_greedy=True,
    chunk_size=10,
):
    # calculate the number of lines of the file cfg.dataset.data_prefix["train"]
    # tmp file
    tmp_file_path = output_file + ".tmp"
    # get path of output file
    # path_name = os.path.dirname(output_file)
    # server output
    # path_name = path_name + "/server_output/"
    # mkdir if not exist
    # os.makedirs(path_name, exist_ok=True)
    keys = {}
    if os.path.exists(tmp_file_path):
        with open(tmp_file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = get_prompt(obj["system"], obj["prompt_turns"])
                response_from = "Assistant"
                prompt += f"{special_tokens['turn_start']}{response_from}{special_tokens['end_of_name']}"
                response = obj["response"]
                response += f"{special_tokens['end_of_turn']}{special_tokens['turn_start']}"
                key = (prompt, response)
                keys[key] = obj

    jobs = []
    # with open(input_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         obj = json.loads(line)
    #         prompt = get_prompt(obj["system"], obj["prompt_turns"])
    #         response_from = "Assistant"
    #         prompt += f"{special_tokens['turn_start']}{response_from}{special_tokens['end_of_name']}"
    #         response = obj["response"]
    #         response += f"{special_tokens['end_of_turn']}{special_tokens['turn_start']}"

    #         key = (prompt, response)
    #         key_map[key] = obj

    #         context = prompt + response
    #         if key in keys:
    #             continue
    #         jobs += [(context, key)]

    key_map = {}

    def generate_jobs(input_file, keys, special_tokens, key_map):
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = get_prompt(obj["system"], obj["prompt_turns"])
                response_from = "Assistant"
                prompt += f"{special_tokens['turn_start']}{response_from}{special_tokens['end_of_name']}"
                response = obj["response"]
                response += f"{special_tokens['end_of_turn']}{special_tokens['turn_start']}"

                key = (prompt, response)
                key_map[key] = obj
                context = prompt + response
                if key in keys:
                    continue
                yield (context, key)

    jobs = generate_jobs(input_file, keys, special_tokens, key_map)
    text_generation_for_batch = get_text_generation_for_batch(server_url, backend_url)

    results = []
    chunk_id = 0
    for id, job in enumerate(jobs):
        data = [job]
        inputs = [input[0] for input in data]
        p_r_pair = [input[1] for input in data]
        length_params = {
            "max_length": 1,
        }
        sampling_params = {
            "use_greedy": use_greedy,
            "end_strings": ["<extra_id_1>", "\x11"],
            "all_probs": False,
            "compute_logprob": True,
        }
        job = {
            "inputs": inputs,
            "length_params": length_params,
            "sampling_params": sampling_params,
            "data_ids": p_r_pair,
        }
        results.append(text_generation_for_batch.delay(job))
        if id != 0 and id % chunk_size == 0:
            chunk_work(results, tmp_file_path, key_map, keys, chunk_id, micro_batch_size)
            results = []
            key_map.clear()
            chunk_id += 1
    chunk_work(results, tmp_file_path, key_map, keys, chunk_id, micro_batch_size)


if __name__ == "__main__":
    fire.Fire(main)
