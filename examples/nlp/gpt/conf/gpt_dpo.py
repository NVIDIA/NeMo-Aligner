
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

from dataclasses import dataclass
from typing import Optional

from nemo_aligner.models.nlp.gpt.megatron_gpt_dpo_model import DPOConfig
from examples.nlp.gpt.conf.utils import (
    default_log,
    default_resume,
    StrategyConfig,
    TrainerConfig,
)

## TODO: support for overwriting these params
## NeMo-Run??
def dpo_config():

    nemo_logger = default_log(
        # TODO
    )
    parallelism_cfg = StrategyConfig(
        ## TODO! all arguments should be optional
        ## in which case, the restored value is used
        ## anything specified here overwrites the restored value
    )
    dpo_trainer_cfg = TrainerConfig(
        num_nodes=8,
        devices=8,
        accelerator="gpu",
        precision="bf16",
    )
    gpt_config = GPTConfig(
        ## TODO! all arguments should be optional
        ## in which case, the restored value is used
        ## anything specified here overwrites the restored value
    )
    dpo_config = DPOConfig(
        ref_policy_kl_penalty=0.2,
        preference_average_log_probs=False,
        gt_reward_scale=1,
        preference_loss="dpo",
        preference_loss_weight=1,
        sft_loss_weight=0,
    )

    return (
        nemo_logger,
        parallelism_cfg,
        dpo_trainer_cfg,
        gpt_config,
        dpo_config,
    )