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

from nemo_aligner.data.nlp.datasets import MCTSDataset

mp.set_start_method("spawn", force=True)
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import numpy as np
import tensorrt_llm
import tensorrt_llm.profiler as profiler
import torch
from celery import Celery
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import get_engine_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.timers import NamedTimer
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import DummyScore, GSK8KFeedbackDataset, SteerLMFeedback
from nemo_aligner.utils.deep_search.mcts.run import run_mcts, run_trtllm_mcts
from nemo_aligner.utils.distributed import broadcast_2d_tensor
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    init_distributed,
    resolve_and_create_trainer,
    temp_pop_from_config,
)
from nemo_aligner.utils.trtllm.trtllm_inference import TRTLLMInference, load_tokenizer, read_model_name
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
    return_memory, _ = output
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

    torch.distributed.barrier()

    return local_batches_to_load, global_batch_ids


def get_global_set(local_data_ids):
    output = [None for _ in range(parallel_state.get_data_parallel_world_size())]
    torch.distributed.all_gather_object(output, local_data_ids, group=parallel_state.get_data_parallel_group())
    global_set = set().union(*output)

    return global_set


def get_local_iterator(global_set, num_to_load, extra_filters=None):
    indices = [i for i in range(num_to_load) if i not in global_set]

    if extra_filters is not None:
        indices = list(filter(lambda x: x not in extra_filters, indices))

    rng = random.Random(len(indices))
    rng.shuffle(indices)

    # rollout_micro_batch_size
    indices = torch.as_tensor(indices).tensor_split(parallel_state.get_data_parallel_world_size())[
        parallel_state.get_data_parallel_rank()
    ]

    return indices


class MCTSSearchOneBatch:
    def __init__(
        self, search_func, collate_func, save_path, dataset, cache_dir,
    ):
        self.search_func = search_func
        self.collate_func = collate_func

        self.dataset = dataset
        self.save_path = save_path
        self.cache_dir = cache_dir

        self.timer = NamedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

        tp_rank = 0
        pp_rank = 0
        self.filename_format = "{num}" + f"_tp_{tp_rank}_pp_{pp_rank}.pt"
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

        output = self.search_func(batch=batch, filename=self.filename_format.format(num=batch_file_name))

        # TODO(geshen): compute metrics
        self.timer.stop("mcts_search_time")

        search_metrics = compute_metric_from_output(output)

        metrics.update(search_metrics)
        metrics["search_time"] = self.timer.get("mcts_search_time")
        metrics["step"] = self.step

        print("##### Metrics", metrics)

        self.outputs.extend(output)
        self.step += 1

        ids = batch["data_id"]
        self.data_ids.update(ids)
        print("###### DONE", batch_idx)

        print("### Finish Job", torch.distributed.get_rank(), "batch_idx", batch_idx, "at step", self.step)
        save_path = os.path.join(self.save_path, f"{batch_file_name}_.pt")
        self.save(save_path)

    def save(self, save_path):
        group = parallel_state.get_model_parallel_group()
        rank = torch.distributed.get_rank(group=group)

        assert rank >= 0

        if rank + 1 == torch.distributed.get_world_size(group):
            print("### RANK SAVING", torch.distributed.get_rank())
            preemptable_save(self.state_dict(), save_path)

        torch.distributed.barrier(group=group)

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
    elif cfg.model.mcts.feedback == "dummy":
        score_fn = DummyScore()
    else:
        raise ValueError(f"Invalid feedback function {cfg.model.mcts.feedback}")
    return train_ds, score_fn


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    ds, score_fn = get_dataset(cfg)
    logging.info(f"loaded {ds}")

    # cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    # if cfg.pretrained_checkpoint.has_value_head:
    #     cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # trainer = resolve_and_create_trainer(cfg, "deep_search")
    OmegaConf.resolve(cfg)
    with temp_pop_from_config(cfg.trainer, "deep_search"):
        trainer = trainer = Trainer(**cfg.trainer)

    # exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # if cfg.pretrained_checkpoint.has_value_head:
    #     hybrid_model_cls = MegatronGPTHybridModel
    # else:
    #     hybrid_model_cls = MegatronGPTModel

    # ptl_model = load_from_nemo(
    #     hybrid_model_cls,
    #     cfg.model,
    #     trainer,
    #     strict=True,
    #     load_base_model_only=not cfg.pretrained_checkpoint.from_mcts_trained,
    #     restore_path=cfg.pretrained_checkpoint.restore_from_path,
    # )
    model_name, model_version = read_model_name(cfg.trtllm.engine_path)
    tokenizer, pad_id, end_id, bos_id = load_tokenizer(
        tokenizer_dir=cfg.trtllm.tokenizer_dir,
        vocab_file=cfg.trtllm.vocab_file,
        model_name=model_name,
        model_version=model_version,
    )
    infer = TRTLLMInference(cfg, end_id, pad_id, bos_id, tokenizer)
    # input_text = ['hello?', 'hello? how are you?']

    # batch_ids = [ tokenizer.encode(i, return_tensors='pt').squeeze(0) for i in input_text]
    # infer.evaluate(batch_ids)
    # infer(inputs=['hello?', 'hello? how are you?'], action=None, context_ids=None, session_info=None)

    # init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # if cfg.pretrained_checkpoint.has_value_head:
    #     ptl_model.prepare_for_inference()
    # else:
    #     ptl_model._reset_activation_checkpointing_args()
    #     ptl_model._reset_sequence_parallelism_args()
    #     ptl_model.eval()

    # ptl_model.freeze()

    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")

    os.makedirs(save_dir, exist_ok=True)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    logger.log_metrics(
        {"dataset_length": len(ds)}, step=0, prefix="data/",
    )

    search_func = partial(
        run_trtllm_mcts,
        trtllm_infer=infer,
        mcts_cfg=cfg.model.mcts,
        score_fn=score_fn,
        eos_id=end_id,
        bos_id=bos_id,
        pad_id=pad_id,
        inference_only=False,
        has_value=cfg.pretrained_checkpoint.has_value_head,
        use_cpu=cfg.model.mcts.kv_cache_in_cpu,
    )

    # start the worker on the rank
    start_worker(search_func, collate_func, save_dir, ds, cfg, cfg.server_url, cfg.backend_url)


def start_worker(search_func, collate_func, save_path, ds, cfg, url, backend_url):
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
            # job = broadcast_2d_tensor(job, 0, None, dtype=torch.int64)
            batch_idx = job[:, 0].tolist()
            replicat_idx = job[:, 1].tolist()
            searcher = MCTSSearchOneBatch(
                search_func=search_func,
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
