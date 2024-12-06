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
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import get_prompt_template_example
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, build_sft_dataset
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel, MambaSFTModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="gpt_sft")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.model.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    ptl_model = load_from_nemo(
        MambaSFTModel if cfg.model.get("mamba_hybrid", False) else GPTSFTModel,
        cfg,
        trainer,
        strict=True,
        restore_path=cfg.model.restore_from_path,
    )

    init_peft(ptl_model, cfg.model)

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

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_data_cfg = cfg.model.data.train_ds
    val_data_cfg = cfg.model.data.validation_ds

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None
    train_ds = build_sft_dataset(
        train_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        is_mamba=cfg.model.get("mamba_hybrid", False),
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
        model_cfg=cfg.model,
    )
    if cfg.model.data.get("sample", False):
        num_samples = cfg.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
    else:
        num_samples = None
    validation_ds = build_sft_dataset(
        val_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        is_mamba=cfg.model.get("mamba_hybrid", False),
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
        model_cfg=cfg.model,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=train_data_cfg.micro_batch_size,
        gbs=train_data_cfg.global_batch_size,
        collate_fn=train_ds.collate_fn,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not train_data_cfg.drop_last,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=val_data_cfg.micro_batch_size,
        gbs=val_data_cfg.global_batch_size,
        collate_fn=validation_ds.collate_fn,
        drop_last=val_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not val_data_cfg.drop_last,
        load_gbs=True,
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)

    sft_trainer = SupervisedTrainer(
        cfg=cfg.trainer.sft,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        sft_trainer.load_state_dict(custom_trainer_state_dict)

    sft_trainer.fit()


if __name__ == "__main__":
    main()
