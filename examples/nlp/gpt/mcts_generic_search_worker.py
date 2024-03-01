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

mp.set_start_method("spawn", force=True)
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from celery import Celery
from datasets import load_dataset
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.timers import NamedTimer
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import Timer, broadcast_python_obj
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed, resolve_and_create_trainer
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, preemptable_save

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("not", lambda x: not x)


prompt_template = """\x00System

\x11User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
\x11Assistant
"""

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""


def groupby(key, output):
    grouped = defaultdict(list)

    for item in output:
        grouped[item[key]].append(item)

    return grouped


def compute_metric_from_output(output):
    return_memory, _ = output
    return_memory = groupby("data_id", return_memory)

    total_value = 0
    num_total = 0

    for k, v in return_memory.items():
        total_value += sum([i["reward"] for i in v])
        num_total += 1

    return {
        "num_total": num_total,
        "average_reward": total_value / num_total if num_total > 0 else 0,
    }


def collate_func(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(b["question"])
        new_dict["answer"].append(b["answer"])
        new_dict["data_id"].append(b["data_id"])

    return new_dict


class MCTSSearchOneBatch:
    def __init__(
        self, search_func, collate_func, save_path, cache_dir,
    ):
        self.search_func = search_func
        self.collate_func = collate_func

        self.save_path = save_path
        self.cache_dir = cache_dir

        self.timer = NamedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.filename_format = "{num}" + f"_tp_{tp_rank}_pp_{pp_rank}.pt"
        # search for the files here
        self.step = 0

    def search(self, batch: List):
        self.data_ids = set()
        self.outputs = []
        batch_ids = [i["data_id"] for i in batch]
        print("###### START", batch_ids)
        batch_file_name = "-".join([str(b) for b in batch_ids])
        batch = self.collate_func(batch)

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

        self.data_ids.update(batch_ids)
        print("###### DONE", batch_ids)

        print("### Finish Job", torch.distributed.get_rank(), "batch_idx", batch_ids, "at step", self.step)
        save_path = os.path.join(self.save_path, f"{batch_file_name}_.pt")
        self.save(save_path)
        return self.outputs

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


class DummpyScore:
    def score(self, response, data_id):
        return 0.0


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    score_fn = DummpyScore()

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    assert cfg.model.mcts.oracle == False
    assert cfg.model.mcts.turn_off_value == False

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "deep_search")

    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    hybrid_model_cls = MegatronGPTHybridModel

    ptl_model = load_from_nemo(
        hybrid_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=not cfg.pretrained_checkpoint.from_mcts_trained,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    ptl_model.prepare_for_inference()
    ptl_model.freeze()

    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")

    if torch.distributed.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    torch.distributed.barrier()

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    search_func = partial(run_mcts, ptl_model=ptl_model, score_fn=score_fn, inference_only=True)

    # start the worker on the rank
    start_worker(search_func, collate_func, save_dir, cfg, cfg.server_url)


def start_worker(search_func, collate_func, save_path, cfg, url):
    if torch.distributed.get_rank() == 0:
        app = Celery("tasks", backend="rpc://", broker=f"{url}")

        app.conf.task_acks_late = True
        app.conf.worker_deduplicate_successful_tasks = True
        app.conf.worker_prefetch_multiplier = 1

        # 5 hrs timeout
        app.conf.update(broker_transport_options={"visibility_timeout": 18000},)

        @app.task
        def search_for_batch(batch):
            batch = broadcast_python_obj(batch, 0, None)
            # braodcast the
            searcher = MCTSSearchOneBatch(
                search_func=search_func,
                collate_func=collate_func,
                save_path=save_path,
                cache_dir=cfg.model.mcts.cache_dir,
            )
            output = searcher.search(batch)
            return output

        RAND_ID = os.environ.get("local_rank", random.randint(0, 10000))
        app.worker_main(
            [
                "worker",
                "--loglevel=INFO",
                "--concurrency=1",
                "--pool=threads",
                "-Ofair",
                f"--hostname=worker-{RAND_ID}@%%h",
            ]
        )
    else:
        while True:
            batch_idx = broadcast_python_obj(batch_idx, 0, None)
            searcher = MCTSSearchOneBatch(
                search_func=search_func,
                collate_func=collate_func,
                save_path=save_path,
                cache_dir=cfg.model.mcts.cache_dir,
            )
            searcher.search(batch_idx)


if __name__ == "__main__":
    main()
