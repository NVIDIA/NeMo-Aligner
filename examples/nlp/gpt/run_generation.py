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
from nemo_aligner.algorithms.generation import GenerationTrainer, eye
from nemo_aligner.data.nlp.builders import build_dataloader, build_sft_dataset, collate_with_pad_to_max_batch
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
    trainer_restore_path = trainer.ckpt_path

    # TODO: log this restore path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    if os.path.exists(gen_file := os.path.join(cfg.exp_manager.explicit_log_dir, "generations", "generations.jsonl")):
        js_line = json.loads(subprocess.check_output(["tail", "-1", gen_file]).decode("utf_8"))
        custom_trainer_state_dict = {"step": js_line["step"], "consumed_samples": js_line["consumed_samples"]}
        consumed_samples = js_line["consumed_samples"]

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    """
    dp_group = parallel_state.get_data_parallel_group()
    calc_gbs = cfg.model.generation.rollout_micro_batch_size * dp_group.size()
    with open_dict(cfg):
        cfg.model.global_batch_size = calc_gbs
    with open_dict(ptl_model.cfg):
        ptl_model.cfg.global_batch_size = calc_gbs
    if hasattr(ptl_model, "global_batch_size"):
        ptl_model.global_batch_size = calc_gbs
    """
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

    # eos_id = ptl_model.tokenizer.eos_id

    # collate fn to pad to the max seq length in the batch
    # collate_fn = partial(
    #    self_rewarding_custom_collate,
    #    eos_id=eos_id,
    #    reset_position_ids=cfg.model.data.get("reset_position_ids", False),
    #    reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
    #    eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
    # )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        collate_fn=eye,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=False,
        load_gbs=True,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    # optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

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
