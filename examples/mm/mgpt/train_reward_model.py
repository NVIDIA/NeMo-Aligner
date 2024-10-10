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
from functools import partial

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.data.mm.datasets import rm_custom_collate

from nemo_aligner.data.mm.builders import (
    build_mm_train_valid_test_regression_rm_datasets,
    build_mm_train_valid_test_rm_datasets,
)
from nemo_aligner.models.mm.mgpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    #init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo #, retrieve_model_state_dict_in_cpu


"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="training_rm")
def main(cfg) -> None:
    """
    Binary ranking reward models use comparison based objective similar to the one found in the
    InstructGPT paper: https://arxiv.org/pdf/2203.02155.pdf and have no explicit labels.
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)
    
    # We need this to looking for the base LLM path, which is not needed
    with open_dict(cfg):
        cfg.model.mm_cfg.llm.from_pretrained = None
        cfg.model.mm_cfg.llm.freeze = False
        #cfg.model.mm_cfg.vision_encoder.freeze = True

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "rm")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        reward_model_cls,
        cfg.model,
        trainer,
        strict=False,
        load_base_model_only=False,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    with open_dict(cfg.model):
        cfg.model.dist_ckpt_format = "zarr"


	# Not sure, it looks like it is needed
    #init_peft(ptl_model, cfg.model)

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    # Set distributed checkpoint format to zarr, as torch_distr is currently failing with OOM
    # when using the default format

    print("Micro batch size: ", cfg.model.micro_batch_size)
    print("Global batch size: ", cfg.model.global_batch_size)
    print("Use CPU Initialization", cfg.model.use_cpu_initialization)
    print("Checkpoint format: ", cfg.model.dist_ckpt_format)

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # use the entire dataset
    train_valid_test_num_samples = [-1 * cfg.model.global_batch_size] * 3

    if reward_model_type == RewardModelType.BINARY_RANKING:
        dataset_builder = build_mm_train_valid_test_rm_datasets
    elif reward_model_type == RewardModelType.REGRESSION:
        dataset_builder = build_mm_train_valid_test_regression_rm_datasets
    else:
        raise ValueError(f"Only support binary_ranking and regression reward model, but get {reward_model_type} ")

    image_processor = ptl_model.model.module.image_processor if hasattr(ptl_model.model, "module") else ptl_model.model.image_processor
    train_ds, validation_ds, _ = dataset_builder(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=ptl_model.tokenizer,
        image_processor=image_processor,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        #pad_samples_to_global_batch_size=False,
        collate_fn=partial(
            rm_custom_collate,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        ),
    )


    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        #pad_samples_to_global_batch_size=False,
        use_random_sampler=False,
        collate_fn=partial(
            rm_custom_collate,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        ),
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    rm_trainer = SupervisedTrainer(
        cfg=cfg.trainer.rm,
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
        rm_trainer.load_state_dict(custom_trainer_state_dict)

    rm_trainer.fit()


if __name__ == "__main__":
    main()
