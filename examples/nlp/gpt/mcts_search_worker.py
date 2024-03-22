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
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import pandas as pd
import requests
import torch
from datasets import load_dataset
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.timers import NamedTimer
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackDataset, GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import Timer
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

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.filename_format = "{num}" + f"_tp_{tp_rank}_pp_{pp_rank}.pt"
        # search for the files here
        self.step = 0

    def search(self, batch_idx: List[int]):
        self.data_ids = set()
        self.outputs = []
        print("###### START", batch_idx)
        batch_file_name = "-".join([str(b) for b in batch_idx])
        batch = self.collate_func([self.dataset[idx] for idx in batch_idx])
        save_path = os.path.join(self.save_path, f"{batch_file_name}_.pt")

        if os.path.exists(save_path):
            return

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

        self.data_ids.update(batch_idx)
        print("###### DONE", batch_idx)

        print("### Finish Job", torch.distributed.get_rank(), "batch_idx", batch_idx, "at step", self.step)
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


def get_dataset(cfg):
    train_ds = MCTSDataset(cfg.dataset.data_prefix["train"], cfg.dataset.prompt_template_name)
    ds = train_ds.data_lookup
    score_fn = GSK8KFeedbackDataset(ds)
    return train_ds, score_fn


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    ds, score_fn = get_dataset(cfg)
    logging.info(f"loaded {ds}")

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    if cfg.pretrained_checkpoint.has_value_head:
        cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "deep_search")

    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    if cfg.pretrained_checkpoint.has_value_head:
        hybrid_model_cls = MegatronGPTHybridModel
    else:
        hybrid_model_cls = MegatronGPTModel

    ptl_model = load_from_nemo(
        hybrid_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=not cfg.pretrained_checkpoint.from_mcts_trained,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    if cfg.pretrained_checkpoint.has_value_head:
        ptl_model.prepare_for_inference()
    else:
        ptl_model._reset_activation_checkpointing_args()
        ptl_model._reset_sequence_parallelism_args()
        ptl_model.eval()

    ptl_model.freeze()

    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")

    if torch.distributed.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    torch.distributed.barrier()

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    logger.log_metrics(
        {"dataset_length": len(ds)}, step=0, prefix="data/",
    )

    search_func = partial(
        run_mcts,
        ptl_model=ptl_model,
        score_fn=score_fn,
        inference_only=False,
        has_value=cfg.pretrained_checkpoint.has_value_head,
    )

    request_obj = json.dumps({"batch_size": cfg.model.mcts.rollout_micro_batch_size})

    while True:
        broadcast_list = [None]

        if torch.distributed.get_rank() == 0:
            batch_idx = requests.put(
                url=f"http://{cfg.server.host}:{cfg.server.port}/get_idx",
                data=request_obj,
                headers={"Content-Type": "application/json"},
            ).json()

            broadcast_list = [batch_idx]

        torch.distributed.broadcast_object_list(broadcast_list, 0, device=torch.cuda.current_device())

        batch_idx = broadcast_list[0]

        if len(batch_idx) == 0:
            print("### NO MORE BATCHES!")
            return

        searcher = MCTSSearchOneBatch(
            search_func=search_func,
            collate_func=collate_func,
            save_path=save_dir,
            dataset=ds,
            cache_dir=cfg.model.mcts.cache_dir,
        )

        searcher.search(batch_idx)


if __name__ == "__main__":
    main()
