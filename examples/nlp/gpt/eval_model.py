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
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.inference.inference_strategy import CodeExecutionStrategy
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.timers import NamedTimer
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.data.nlp.datasets import MCTSDataset
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackDataset, MathSandBoxedFeedBack
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed, resolve_and_create_trainer
from nemo_aligner.utils.trainer_utils import compute_limit_batches
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, preemptable_save

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("not", lambda x: not x)

mp.set_start_method("spawn", force=True)

# Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}


def run_inference(
    model, feedback, dataloader, limit_batches=1.0, num_to_log_to_table=10, desc="inference", strategy=None
):
    model.prepare_for_inference()

    total = 0
    num_correct = 0
    logged = 0

    tables = []

    limit_batches = compute_limit_batches(len(dataloader), limit_batches)

    loop_iter = zip(range(limit_batches), dataloader)
    inference_pbar = tqdm(loop_iter, total=min(len(dataloader), limit_batches), leave=True, desc=desc)

    incorrect_samples = []

    for _, batch in inference_pbar:
        output = model.generate(batch["question"], strategy=strategy)

        for question, response, answer in zip(batch["question"], output["sentences"], batch["answer"], strict=True):
            score = feedback.score(response, answer)

            if score > 0:
                num_correct += 1
            else:
                incorrect_samples.append({"question": question, "expected_answer": answer})

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
        incorrect_samples,
    )


def collate_fn(batch):
    # applies the steerlm format and
    # transposes the list of dict to dict of lists
    new_dict = defaultdict(list)

    for b in batch:
        new_dict["question"].append(b["question"])
        new_dict["answer"].append(b["expected_answer"])

    return new_dict


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    train_ds = MCTSDataset(cfg.dataset.data_prefix["train"], cfg.dataset.prompt_template_name)
    val_ds = MCTSDataset(cfg.dataset.data_prefix["validation"], cfg.dataset.prompt_template_name)
    test_ds = MCTSDataset(cfg.dataset.data_prefix["test"], cfg.dataset.prompt_template_name)

    sandbox_cfg = {"host": os.getenv("NEMO_SKILLS_SANDBOX_HOST"), "port": os.getenv("NEMO_SKILLS_SANDBOX_PORT")}

    feedback = MathSandBoxedFeedBack(**sandbox_cfg,)

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

    code_strategy = CodeExecutionStrategy(sandbox_cfg=sandbox_cfg | {"sandbox_type": "local"}, model=ptl_model)

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
        drop_last=False,
        shuffle=False,
    )

    test_dataloader_builder_func = partial(
        build_dataloader,
        cfg=cfg,
        dataset=test_ds,
        consumed_samples=0,
        mbs=cfg.model.inference.micro_batch_size,
        gbs=cfg.model.inference.micro_batch_size * dp_size,
        load_gbs=False,
        collate_fn=collate_fn,
        drop_last=False,
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
        drop_last=False,
        shuffle=False,
    )

    val_dataloader = val_dataloader_builder_func()
    train_dataloader = train_dataloader_builder_func()
    test_dataloader = test_dataloader_builder_func()

    val_metrics, val_table, val_wrong = run_inference(
        ptl_model, feedback, val_dataloader, desc="val inference", strategy=code_strategy
    )
    logger.log_metrics(val_metrics, step=0, prefix="val/")
    logger.log_table("table/val", dataframe=val_table, step=0)

    test_metrics, test_table, test_wrong = run_inference(
        ptl_model, feedback, test_dataloader, desc="test inference", strategy=code_strategy
    )
    logger.log_metrics(test_metrics, step=0, prefix="test/")
    logger.log_table("table/test", dataframe=test_table, step=0)

    train_metrics, train_table, train_wrong = run_inference(
        ptl_model, feedback, train_dataloader, desc="train inference", strategy=code_strategy,
    )
    logger.log_metrics(train_metrics, step=0, prefix="train/")
    logger.log_table("table/train", dataframe=train_table, step=0)

    save_dir = Path(cfg.exp_manager.explicit_log_dir) / "samples"
    save_dir.mkdir(exist_ok=True)

    torch.save(train_wrong, save_dir / "train_wrong_{}.pt".format(parallel_state.get_data_parallel_rank()))
    torch.save(val_wrong, save_dir / "val_wrong_{}.pt".format(parallel_state.get_data_parallel_rank()))
    torch.save(test_wrong, save_dir / "test_wrong_{}.pt".format(parallel_state.get_data_parallel_rank()))


if __name__ == "__main__":
    main()
