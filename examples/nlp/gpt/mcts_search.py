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

import os
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Union

import torch
import torch.multiprocessing as mp
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
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed, resolve_and_create_trainer
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""


def preemptable_save(obj, save_path: Path):
    with tempfile.NamedTemporaryFile(dir=save_path.parent, delete=False) as temp_file:
        # do the expensive op before replace
        torch.save(obj, temp_file.name)

        # this should be atomic
        Path(temp_file.name).replace(save_path)


def compute_metric_from_output(output):
    return_memory, _ = output
    num_correct = 0
    num_total = 0

    for item in return_memory:
        reward = item["reward"]

        if reward > 0:
            num_correct += 1

        num_total += 1

    return {"num_correct": num_correct, "num_total": num_total, "accuracy": num_correct / num_total}


def collate_func(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(b["question"])
        new_dict["answer"].append(b["answer"])
        new_dict["data_id"].append(b["data_id"])

    return new_dict


class MCTSSearch:
    def __init__(self, search_func, collate_func, save_path, batch_chunks, dataset, logger, run_timer, save_interval):
        self.search_func = search_func
        self.collate_func = collate_func
        self.save_path = Path(save_path).resolve()
        self.batch_chunks = batch_chunks

        self.dataset = dataset
        self.logger = logger

        # has to be DP specific timer
        self.run_timer = run_timer

        self.idx = 0
        self.outputs = []

        self.timer = NamedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

        if self.save_path.exists():
            self.load_state_dict(torch.load(self.save_path))

        self.save_interval = save_interval

    def search(self):
        self.run_timer.start_time()

        loop_iter = range(self.idx, len(self.batch_chunks))

        global_pbar = tqdm(
            loop_iter, initial=self.idx, total=len(self.batch_chunks), leave=True, desc="Search Global Step"
        )

        for i in global_pbar:
            batch_idx = self.batch_chunks[i]

            batch = self.collate_func([self.dataset[idx] for idx in batch_idx.tolist()])

            metrics = {}
            self.timer.start("mcts_search_time")
            output = self.search_func(batch=batch)
            # TODO(geshen): compute metrics
            self.timer.stop("mcts_search_time")

            search_metrics = compute_metric_from_output(output)

            metrics.update(search_metrics)
            metrics["search_time"] = self.timer.get("mcts_search_time")
            metrics["step"] = self.idx

            global_pbar.set_postfix(metrics)

            self.logger.log_metrics(
                metrics, step=self.idx, prefix="search/",
            )

            self.outputs.extend(output)
            self.idx = i

            if self.run_timer.is_within_dp_finished() or i % self.save_interval == 0:
                self.save()

    def save(self):
        group = parallel_state.get_model_parallel_group()
        rank = torch.distributed.get_rank(group=group)

        assert rank >= 0

        if rank + 1 == torch.distributed.get_world_size(group):
            print("### RANK SAVING", torch.distributed.get_rank())
            preemptable_save(self.state_dict(), self.save_path)

        torch.distributed.barrier(group=group)

    def state_dict(self):
        return {"idx": self.idx, "mcts_outputs": self.outputs}

    def load_state_dict(self, state_dict):
        self.idx = state_dict["idx"]
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


class DatasetWrapper:
    def __init__(self, ds):
        self.ds = ds

    # just like a dataset but return idx
    def __getitem__(self, idx):
        data_item = self.ds[idx]
        data_item["question"] = steerlm_template.format(prompt=data_item["question"])
        return {**data_item, "data_id": idx}

    def __len__(self):
        return len(self.ds)


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    dataset = load_dataset("gsm8k", "main")

    train_ds = DatasetWrapper(dataset["train"])
    score_fn = GSK8KFeedbackHF(split="train")

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

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
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    breakpoint()

    ptl_model.prepare_for_inference()
    ptl_model.freeze()

    num_to_load = compute_limit_batches(len(train_ds), cfg.model.mcts.num_rollouts)

    dp_rank = parallel_state.get_data_parallel_rank()
    indices = torch.arange(num_to_load).tensor_split(parallel_state.get_data_parallel_world_size())[dp_rank]

    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")

    if torch.distributed.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    torch.distributed.barrier()

    # we only really need model parallel src to save the checkpoint
    save_path = os.path.join(save_dir, "dp_{}.pt".format(dp_rank))

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    search_func = partial(run_mcts, ptl_model=ptl_model, score_fn=score_fn)

    searcher = MCTSSearch(
        search_func=search_func,
        collate_func=collate_func,
        save_path=save_path,
        batch_chunks=indices.split(cfg.model.mcts.rollout_micro_batch_size),
        dataset=train_ds,
        logger=logger,
        run_timer=timer,
        save_interval=cfg.trainer.deep_search.save_interval,
    )

    searcher.search()


if __name__ == "__main__":
    main()
