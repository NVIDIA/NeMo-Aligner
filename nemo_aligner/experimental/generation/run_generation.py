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
import json
import os
import subprocess
from functools import partial

import torch
import torch.multiprocessing as mp
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_sft_dataset,
    collate_with_pad_to_max_batch,
    identity_collate,
)
from nemo_aligner.experimental.generation.generation import GenerationTrainer
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu

"""Script to start Aligner Generation"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="gpt_generation")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "generation")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        MegatronGPTModel, cfg.model, trainer, strict=True, restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    with open_dict(cfg):
        # overwrite the model config with the config from the checkpoint
        cfg.model.encoder_seq_length = ptl_model.cfg.encoder_seq_length

    # pull values from checkpoint
    # trainer_restore_path = trainer.ckpt_path

    if os.path.exists(gen_file := os.path.join(cfg.exp_manager.explicit_log_dir, "generations", "generations.jsonl")):
        js_line = json.loads(subprocess.check_output(["tail", "-1", gen_file]).decode("utf_8"))
        custom_trainer_state_dict = {"step": js_line["step"], "consumed_samples": js_line["consumed_samples"]}
        consumed_samples = js_line["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    train_data_cfg = cfg.model.data.train_ds

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.generation.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.generation.max_steps * cfg.model.global_batch_size
    else:
        num_samples = None
    train_ds = build_sft_dataset(
        train_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        collate_fn=identity_collate,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=False,
        load_gbs=True,
        use_random_sampler=False,
        limit_train_batches=cfg.trainer.generation.limit_train_batches,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)

    gen_trainer = GenerationTrainer(
        cfg=cfg.trainer.generation,
        model=ptl_model,
        train_dataloader=train_dataloader,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
        exp_manager=cfg.exp_manager,
    )

    if custom_trainer_state_dict is not None:
        gen_trainer.load_state_dict(custom_trainer_state_dict)

    gen_trainer.generate()


if __name__ == "__main__":
    main()
