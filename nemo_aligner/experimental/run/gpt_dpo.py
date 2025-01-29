# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import functools
from dataclasses import dataclass
from typing import Optional

import nemo_run as run
import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig

from nemo_aligner.algorithms.dpo import DPOTrainer
from nemo_aligner.data.nlp.config import DPODataConfig, RLHFDataConfig
from nemo_aligner.models.nlp.gpt.megatron_gpt_dpo_model import DPOConfig
from nemo_aligner.utils.nemo2.config_utils import GPTConfigOverrides, ParallelismOverrides
from nemo_aligner.utils.nemo2.optim import CosineAnnealingScheduler, MegatronOptimizer
from nemo_aligner.utils.nemo2.precision import MegatronMixedPrecision


@run.cli.factory
def default_dpo_config() -> run.Config[DPOConfig]:
    return run.Config(
        DPOConfig,
        ref_policy_kl_penalty=0.2,
        preference_avg_log_probs=False,
        gt_reward_scale=1.0,
        preference_loss="dpo",
        preference_loss_weight=1,
        sft_loss_weight=0,
    )


## need to be able to optionally specify any of the things in
## GPTConfig (https://github.com/NVIDIA/NeMo/blob/eb892ae14e204fda269c9a37c6d78578ffd9a0df/nemo/collections/llm/gpt/model/base.py#L162)
## to overwrite
def gpt_config_overrides():
    return GPTConfigOverrides(hidden_dropout=0.5,)  ## test overriding the config


def default_dpo_parallelism():
    return ParallelismOverrides(tensor_model_parallel_size=1,)  # 2,


## hparams not mapped
## bucket_cap_mb -- passed to NLPDDPStrategy
## overlap_grad_sync
## contiguous_grad_buffer
## ** scheduler **
@run.cli.factory
def megatron_adam_optimizer() -> run.Config[MegatronOptimizer]:
    ## setup optimizer and scheduler
    return run.Config(
        MegatronOptimizer,
        config=run.Config(
            OptimizerConfig,
            optimizer="adam",
            lr=9e-6,
            weight_decay=0.1,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
        ),
        ## TODO: bucket_cap_mb
        lr_scheduler=run.Config(
            CosineAnnealingScheduler,
            warmup_steps=10,
            constant_steps=1000,
            max_steps=10000,  ## TODO: make this match trainer.max_steps
            min_lr=9e-7,
        ),
    )


@run.cli.factory
def default_dpo_data_config() -> run.Config[DPODataConfig]:
    return run.Config(DPODataConfig, data_prefix="test", micro_batch_size=4, global_batch_size=32,)


@run.cli.factory
def default_rlhf_data_config() -> run.Config[RLHFDataConfig]:
    return run.Config(RLHFDataConfig, data_prefix="test", micro_batch_size=4, global_batch_size=32,)


## TODO: remove PTL dependency
## need to export WANDB_API_KEY in order for this to work
def wandb_logger() -> WandbLogger:
    return WandbLogger(name="llama3_dpo_nemo2", project="nemo_dpo_baseline", save_dir="/tmp/test_wandb",)


## config for NeMo-Aligner trainer
def default_dpo_trainer():
    return functools.partial(
        DPOTrainer,
        limit_val_batches=-1,  ## TODO: is this right?
        val_check_interval=0.1,
        gradient_clip_val=1.0,
        max_epochs=3,
        save_interval=100,
        limit_train_batches=1.0,
        max_steps=-1,
        # precision=run.Config(
        #    MegatronMixedPrecision,
        #    precision="bf16-mixed",
        #    params_dtype=torch.bfloat16,
        # ),
        precision=MegatronMixedPrecision(precision="bf16-mixed", params_dtype=torch.bfloat16,),
    )
