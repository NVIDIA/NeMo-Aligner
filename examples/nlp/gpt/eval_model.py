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
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
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
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackDataset, GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed, resolve_and_create_trainer
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, preemptable_save
from pathlib import Path

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("not", lambda x: not x)

mp.set_start_method("spawn", force=True)

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


def run_inference(model, feedback, dataloader, limit_batches=1.0, num_to_log_to_table=10, desc="inference"):
    model.prepare_for_inference()

    total = 0
    num_correct = 0
    logged = 0

    tables = []

    limit_batches = compute_limit_batches(len(dataloader), limit_batches)

    loop_iter = zip(range(limit_batches), dataloader)
    inference_pbar = tqdm(loop_iter, total=min(len(dataloader), limit_batches), leave=True, desc=desc)

    data_ids_that_are_incorrect = set()

    for _, batch in inference_pbar:
        output = model.generate(batch["question"])

        for response, answer, data_id in zip(output["sentences"], batch["answer"], batch["data_id"], strict=True):
            score = feedback.score(response, answer)

            if score > 0:
                num_correct += 1
            else:
                data_ids_that_are_incorrect.add(data_id)

            if logged < num_to_log_to_table:
                table = {}
                table["reward"] = score
                table["response"] = response
                table["ground_truth_answer"] = answer

                tables.append(table)
                logged += 1

            total += 1

    model.finish_inference()

    metric_output = torch.as_tensor([num_correct, total], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.all_reduce(metric_output, group=parallel_state.get_data_parallel_group())
    num_correct, total = metric_output.tolist()

    df = pd.DataFrame(columns=["step", "response", "reward", "ground_truth_answer"])

    for table in tables:
        df.loc[len(df)] = [
            0,
            table["response"],
            table["reward"],
            table["ground_truth_answer"],
        ]

    return (
        {
            "global_total": total,
            "global_correct": num_correct,
            "global_accuracy": num_correct / total if total > 0 else 0,
        },
        df,
        data_ids_that_are_incorrect
    )


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


def collate_fn(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(b["question"])
        new_dict["answer"].append(b["answer"])
        new_dict["data_id"].append(b["data_id"])

    return new_dict


def get_prompt_template(template_name):
    if template_name == "steerlm":
        template = steerlm_template
    elif template_name == "mistral":
        template = prompt_template
    else:
        raise NotImplementedError(f"template {template_name} is not supported")

    return template


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    template = get_prompt_template(cfg.dataset.prompt_template_name)
    val_ds = DatasetWrapper(load_dataset("gsm8k", "main")["test"], template)
    train_ds = DatasetWrapper(load_dataset("gsm8k", "main")["train"], template)
    feedback = GSK8KFeedbackDataset()

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
        load_base_model_only=not cfg.pretrained_checkpoint.from_mcts_trained,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    ptl_model.prepare_for_inference()
    ptl_model.freeze()

    dp_size = parallel_state.get_data_parallel_world_size()

    # hack to allow using all of the validation dataset
    # TODO: partial this dataloader into the func
    val_dataloader_builder_func = partial(
        build_dataloader,
        cfg=cfg,
        dataset=val_ds,
        consumed_samples=0,
        mbs=cfg.model.inference.micro_batch_size,
        gbs=cfg.model.inference.micro_batch_size * dp_size,
        load_gbs=False,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False,
    )

    train_dataloader_builder_func = partial(
        build_dataloader,
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=cfg.model.inference.micro_batch_size,
        gbs=cfg.model.inference.micro_batch_size * dp_size,
        load_gbs=False,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False,
    )

    val_dataloader = val_dataloader_builder_func()
    train_dataloader = train_dataloader_builder_func()

    val_metrics, val_table, val_wrong = run_inference(ptl_model, feedback, val_dataloader, desc="val inference")
    logger.log_metrics(val_metrics, step=0, prefix="val/")
    logger.log_table("table/val", dataframe=val_table, step=0)

    train_metrics, train_table, train_wrong = run_inference(ptl_model, feedback, train_dataloader, desc="train inference")
    logger.log_metrics(train_metrics, step=0, prefix="train/")
    logger.log_table("table/train", dataframe=train_table, step=0)

    save_dir = Path(cfg.exp_manager.explicit_log_dir) / "sets"
    save_dir.mkdir(exist_ok=True)

    torch.save(train_wrong, save_dir / "train_wrong_{}.pt".format(parallel_state.get_data_parallel_rank()))
    torch.save(val_wrong, save_dir / "val_wrong_{}.pt".format(parallel_state.get_data_parallel_rank()))

if __name__ == "__main__":
    main()
