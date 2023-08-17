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

import datetime
import os
import random

import numpy as np
import torch
from nemo_rlhf.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import mpu

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


def load_nemo_or_checkpoint(model_class, trainer, cfg):
    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        # override model config
        pretrained_cfg = model_class.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
                pretrained_cfg.megatron_amp_O2 = True

            # to improve the accuracy of reward model
            if model_class == MegatronGPTRewardModel and pretrained_cfg.get("megatron_amp_O2", False):
                pretrained_cfg.force_head_dtype = "float32"

            mcore_gpt = cfg.get("mcore_gpt", None)
            if mcore_gpt is not None:
                pretrained_cfg.mcore_gpt = mcore_gpt

            # hack batch_size in NeMo
            pretrained_cfg.micro_batch_size = cfg.data.micro_batch_size
            pretrained_cfg.global_batch_size = cfg.data.micro_batch_size * mpu.get_data_parallel_world_size()

            # set default parallel_size
            if cfg.get("tensor_model_parallel_size", -1) > 0:
                pretrained_cfg.tensor_model_parallel_size = cfg.get("tensor_model_parallel_size")
            if cfg.get("pipeline_model_parallel_size", -1) > 0:
                pretrained_cfg.pipeline_model_parallel_size = cfg.get("pipeline_model_parallel_size")
            if cfg.get("pipeline_model_parallel_split_rank", -1) >= 0:
                pretrained_cfg.pipeline_model_parallel_split_rank = cfg.get("pipeline_model_parallel_split_rank")

            assert (
                cfg.trainer.devices
                * cfg.trainer.num_nodes
                % pretrained_cfg.tensor_model_parallel_size
                * pretrained_cfg.pipeline_model_parallel_size
                == 0
            ), "devices * num_nodes should be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"

        model = model_class.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
        )
    elif cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = model_class.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    return model


def get_max_time_per_run(max_time_per_run):
    time_array = max_time_per_run.split(":")
    if len(time_array) == 4:
        hours = int(time_array[-4]) * 24 + int(time_array[-3])
    else:
        hours = int(time_array[-3])
    mins = int(time_array[-2])
    secs = int(time_array[-1])
    max_time_per_run = datetime.timedelta(hours=hours, minutes=mins, seconds=secs)
    return max_time_per_run


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
