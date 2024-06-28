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

import torch.multiprocessing as mp

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo_aligner.data.nlp.datasets import MCTSDataset
from nemo_aligner.utils.deep_search.mcts.search_stop_criteria import SearchStopCriteria
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.serve_trt_for_treesearch import TensorRTLLMModelClient
from nemo_aligner.utils.trtllm.trtllm_inference import TRTLLMInference

mp.set_start_method("spawn", force=True)
import os
import random
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from celery import Celery
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.timers import NamedTimer
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import (
    DummyScore,
    GSK8KFeedbackDataset,
    HelpSteerFeedback,
    SteerLMFeedback,
)
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import broadcast_2d_tensor
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed, resolve_and_create_trainer
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, preemptable_save

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("not", lambda x: not x)


def groupby(key, output):
    grouped = defaultdict(list)

    for item in output:
        grouped[item[key]].append(item)

    return grouped


def compute_metric_from_output(output):
    return_memory, _, return_positive_negative_samples = output
    return_memory = groupby("data_id", return_memory)

    num_correct = 0
    num_total = 0

    for k, v in return_memory.items():
        is_correct = all(r["reward"] > 0 for r in v)

        num_correct += is_correct
        num_total += 1

    return {
        "num_correct": num_correct,
        "num_total": num_total,
        "accuracy": num_correct / num_total if num_total > 0 else 0,
    }


def collate_func(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(b["question"])
        new_dict["data_id"].append(b["data_id"])

    return new_dict


def get_cached_outputs(cache_dir, global_set):
    """get the cached outputs that we didn't finish, need to make sure the rank actually completes it
    """
    dp_rank = parallel_state.get_data_parallel_rank()

    local_batches_to_load = []
    global_batch_ids = set()

    if cache_dir is None:
        return local_batches_to_load, global_batch_ids

    to_delete = []

    for p in sorted(Path(cache_dir).glob("*.pt")):
        batches = list(map(int, p.name.split("_")[0].split("-")))
        fs_dp_rank = int(p.name.split("_")[2])

        if all(b in global_set for b in batches):
            to_delete.append(p)
        elif dp_rank == fs_dp_rank:
            local_batches_to_load.extend(batches)

        global_batch_ids.update(batches)

    if torch.distributed.get_rank() == 0:
        print("### DELETING FILES", to_delete)
        for p in to_delete:
            p.unlink()

    return local_batches_to_load, global_batch_ids


class SynGen:
    def __init__(self, score_fun, tokenizer, pad_id, wall_time_seconds, mcts_cfg) -> None:
        self.score_fn = score_fun
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.wall_time_seconds = wall_time_seconds
        self.mcts_cfg = mcts_cfg

    def reset_exit_search_timer(self):
        self.exit = False
        self.exit_search_timer = threading.Timer(self.wall_time_seconds, self.exit_search)
        self.exit_search_timer.daemon = True
        self.exit_search_timer.start()

    def exit_search(self):
        print("### TIMER TRIGGER")
        self.exit = True

    def gen(self, batch):
        self.reset_exit_search_timer()
        # self.exit_search_timer.start()
        inputs = batch["question"]
        data_ids = batch["data_id"]
        mcts_cfg = self.mcts_cfg

        termination_condition = TerminationCondition(
            mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings, end_tokens=[self.tokenizer.eos_id]
        )

        stop_criteria = SearchStopCriteria(self.score_fn, [termination_condition], threshold=mcts_cfg.value_threshold)

        gen_fun = GenFunction(
            self.tokenizer.tokenizer,
            stop_criteria,
            self.pad_id,
            mcts_cfg.top_k,
            mcts_cfg.top_p,
            mcts_cfg.add_bos_token,
        )
        samples = []
        number_of_tries = 0
        while True:
            number_of_tries += 1
            # clear the cache
            if len(inputs) == 0:
                break
            if self.exit:
                # save the best output
                for data_id in data_ids:
                    best = 0
                    best_text = ""
                    print(f"### data_id: {data_id} ### total samples {len(stop_criteria.evaluation_cache[data_id])}")
                    for text in stop_criteria.evaluation_cache[data_id]:
                        results, tokens = stop_criteria.evaluation_cache[data_id][text]
                        samples.append(
                            {"value": results[0], "text": text, "tokens": tokens, "data_id": data_id,}
                        )
                        if results[0] > best:
                            best = results[0]
                            best_text = text
                    if best >= stop_criteria.threshold:
                        print(f"### data_id: {data_id} FOUND A GOOD SAMPLE {best} ###")
                        print(f"{best_text}")
                    else:
                        print(f"### data_id: {data_id} THE BEST SAMPLE SO FAR {best} ###")
                        print(f"{best_text}")
                break
            # compute seed number based data_ids and timestamp and number of tries
            timestamp = time.time()
            # mode a large prime number
            large_prime = 2147462143
            seed = hash(str(data_ids) + str(timestamp) + str(number_of_tries)) % large_prime
            outputs = gen_fun(inputs=inputs, data_ids=data_ids, seed=seed)
            new_inputs = []
            new_data_ids = []
            for data_id in data_ids:
                if data_id in stop_criteria.max_value:
                    # print out the maximum value
                    max_value = stop_criteria.max_value[data_id]
                    print(f"### MAX VALUE FOR DATA ID {data_id} IS {max_value}")
            for input, data_id, output in zip(inputs, data_ids, outputs):
                if data_id in stop_criteria.terminate and stop_criteria.terminate[data_id]:
                    print(f"### data_id: {data_id} ### total samples {len(stop_criteria.evaluation_cache[data_id])}")
                    for text in stop_criteria.evaluation_cache[data_id]:
                        results, tokens = stop_criteria.evaluation_cache[data_id][text]
                        samples.append(
                            {"value": results[0], "text": text, "tokens": tokens, "data_id": data_id,}
                        )
                        if results[0] >= stop_criteria.threshold:
                            print(f"### FOUND A GOOD SAMPLE ###")
                            print(f"{text}")
                else:
                    new_inputs.append(input)
                    new_data_ids.append(data_id)
            inputs = new_inputs
            data_ids = new_data_ids
        return samples


class GenFunction(TRTLLMInference):
    def __init__(self, tokenizer, stop_criteria, pad_id, top_k, top_p=0.75, add_bos_token=False):

        host = os.getenv("TRTLLM_GEN_HOST", "localhost")
        port = os.getenv("TRTLLM_GEN_PORT", "5000")
        self.infer = TensorRTLLMModelClient(host=host, port=port)
        self.stop_criteria = stop_criteria
        self.add_bos_token = add_bos_token
        self.top_k = top_k
        self.top_p = top_p
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.max_depth_to_explore = 1024

    def __call__(
        self, inputs=None, data_ids=None, seed=0,  # add bos token at the beginning of the input text
    ):
        context_tokens = self.tokenize_batch(inputs, self.add_bos_token)
        context_tokens = [c.tolist() for c in context_tokens]

        results = []
        input_ids = []
        for context in context_tokens:
            results.append(None)
            input_ids.append(context)

        infer_results = []
        if len(input_ids) != 0:
            context_lengths = [len(c) for c in input_ids]
            out = self.infer.generate(
                batch_input_ids=input_ids,
                input_lengths=context_lengths,
                tokens_to_generate=self.max_depth_to_explore,
                temperature=1.0,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=1.0,
                random_seed=seed,
                stop_phrases=["<extra_id_1>"],
                remove_stop_phrases=False,
            )

            try:
                for i in range(len(out)):
                    # text = out[i]['generation']
                    generation_ids = out[i]["output_ids"]
                    full_ids = input_ids[i] + generation_ids
                    text = self.tokenizer.decode(full_ids)
                    value, terminate, end_properly, has_answer = self.stop_criteria.get_value_and_terminated(
                        text, data_ids[i], i, full_ids
                    )
                    value = torch.tensor(value)
                    infer_results.append(value)
            except Exception as e:
                print(f"Error in value estimation: {e}")
                print(out)
                print(len(out))
                print(out[0])
                print(type(out))
                print(type(out[0]))
                import traceback

                traceback.print_exc()
                infer_results = [torch.tensor(0.0)] * len(input_ids)
        # replace None with the results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = infer_results.pop(0)
        return results


class MCTSSearchOneBatch:
    def __init__(
        self, search_func_args, collate_func, save_path, dataset, cache_dir,
    ):
        self.search_func_args = search_func_args
        self.collate_func = collate_func

        self.dataset = dataset
        self.save_path = save_path
        self.cache_dir = cache_dir

        self.timer = NamedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

        self.filename_format = "{num}" + f"_.pt"
        # search for the files here
        self.step = 0

    def search(self, batch_idx: List[int], replica_idx: List[int]):
        self.data_ids = set()
        self.outputs = []
        print("###### START", batch_idx)
        # compute seed number based on batch_idx and replica_idx and timestamp using hash function
        timestamp = time.time()
        # mode a large prime number
        large_prime = 2147462143
        seed = hash(str(batch_idx) + str(replica_idx) + str(timestamp)) % large_prime
        # seed seed for math, pytorch and numpy, random
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        batch_file_name = "-".join([str(b) + "@" + str(r) for b, r in zip(batch_idx, replica_idx)])
        batch_data = []
        for idx, replica_id in zip(batch_idx, replica_idx):
            dp = self.dataset[idx]
            dp["data_id"] = str(dp["data_id"]) + "@" + str(replica_id)
            batch_data.append(dp)
        batch = self.collate_func(batch_data)

        metrics = {}
        self.timer.start("mcts_search_time")

        syn_gen = SynGen(*self.search_func_args)
        output = syn_gen.gen(batch=batch)

        # TODO(geshen): compute metrics
        self.timer.stop("mcts_search_time")

        metrics["search_time"] = self.timer.get("mcts_search_time")
        metrics["step"] = self.step

        print("##### Metrics", metrics)

        self.outputs.extend(output)
        self.step += 1

        ids = batch["data_id"]
        self.data_ids.update(ids)
        print("###### DONE", batch_idx)

        print("### Finish Job", "batch_idx", batch_idx, "at step", self.step)
        save_path = os.path.join(self.save_path, f"{batch_file_name}_.pt")
        self.save(save_path)

    def save(self, save_path):
        print("### RANK SAVING")
        preemptable_save(self.state_dict(), save_path)

    def state_dict(self):
        return {"data_ids": self.data_ids, "mcts_outputs": self.outputs}

    def load_state_dict(self, state_dict):
        self.data_ids = state_dict["data_ids"]
        self.outputs = state_dict["mcts_outputs"]


def compute_limit_batches(number_of_batches: int, limit_batches: Union[int, float, None]):
    if limit_batches is None:
        limit_batches = 1.0

    if isinstance(limit_batches, float):
        limit_batches = int(number_of_batches * limit_batches)
    elif isinstance(limit_batches, int):
        limit_batches = min(number_of_batches, limit_batches)
    else:
        raise TypeError(f"Invalid data type of {type(limit_batches)} cannot compute limit batches")

    return limit_batches


@dataclass
class DatasetWrapper:
    ds: torch.utils.data.Dataset
    template: str

    # just like a dataset but return idx
    def __getitem__(self, idx):
        data_item = self.ds[idx]
        data_item["question"] = self.template.format(prompt=data_item["question"])
        return {**data_item, "data_id": idx}

    def __len__(self):
        return len(self.ds)


def get_dataset(cfg):
    train_ds = MCTSDataset(cfg.dataset.data_prefix["train"], cfg.dataset.prompt_template_name)
    ds = train_ds.data_lookup
    if cfg.model.mcts.feedback == "math":
        score_fn = GSK8KFeedbackDataset(ds)
    elif cfg.model.mcts.feedback == "steerlm":
        score_fn = SteerLMFeedback()
    elif cfg.model.mcts.feedback == "helpsteer":
        score_fn = HelpSteerFeedback(cfg.model.mcts.reward_weights)
    elif cfg.model.mcts.feedback == "dummy":
        score_fn = DummyScore()
    else:
        raise ValueError(f"Invalid feedback function {cfg.model.mcts.feedback}")
    return train_ds, score_fn


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    ds, score_fn = get_dataset(cfg)

    library = "sentencepiece"
    model_name = "solar"
    tokenizer_model = cfg.trtllm.vocab_file
    tokenizer = get_nmt_tokenizer(
        library=library, model_name=model_name, tokenizer_model=tokenizer_model, use_fast=True
    )
    tokenizer = tokenizer
    pad_id = tokenizer.pad_id

    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")
    os.makedirs(save_dir, exist_ok=True)

    # syn_gen = SynGen(score_fn, tokenizer, pad_id, cfg.model.mcts.max_wall_time, cfg.model.mcts)

    # start the worker on the rank
    start_worker((score_fn, tokenizer, pad_id, cfg.model.mcts.max_wall_time, cfg.model.mcts), collate_func, save_dir, ds, cfg, cfg.server_url, cfg.backend_url)


def start_worker(search_func_args, collate_func, save_path, ds, cfg, url, backend_url):
    app = Celery("tasks", backend=f"{backend_url}", broker=f"{url}")

    app.conf.task_acks_late = True
    app.conf.worker_deduplicate_successful_tasks = True
    app.conf.worker_prefetch_multiplier = 1

    # 5 hrs timeout
    app.conf.update(broker_transport_options={"visibility_timeout": 18000},)

    @app.task(
        bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_jitter=True, retry_kwargs={"max_retries": 10},
    )
    def search_for_batch(self, job):
        try:
            beg_time = time.time()
            # job exmaple [(0, 0), (0, 1), (1, 0), (1, 1)], list of (batch_idx, replica_idx)
            job = torch.tensor(job)
            batch_idx = job[:, 0].tolist()
            replicat_idx = job[:, 1].tolist()
            searcher = MCTSSearchOneBatch(
                search_func_args=search_func_args,
                collate_func=collate_func,
                save_path=save_path,
                dataset=ds,
                cache_dir=cfg.model.mcts.cache_dir,
            )
            searcher.search(batch_idx, replicat_idx)
            end_time = time.time()
        except Exception as e:
            print("ERROR", e)
            # print the stack trace
            import traceback

            traceback.print_exc()
            raise self.retry(exc=e)
        return {"batch_ids": batch_idx, "time": end_time - beg_time}

    RAND_ID = os.environ.get("local_rank", random.randint(0, 10000))
    app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            "--concurrency=1",
            "--pool=threads",
            "--without-gossip",
            "-Ofair",
            f"--hostname=worker-{RAND_ID}@%%h",
        ]
    )


if __name__ == "__main__":
    main()
