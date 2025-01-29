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
from functools import partial
from threading import local
from typing import Any

import nemo_run as run
import torch.multiprocessing as mp

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import io
from nemo.lightning._strategy_lib import set_model_parallel_attributes
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.dpo import DPOTrainer, dpo_custom_collate
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_dpo_datasets,
    build_train_valid_test_dpo_packed_datasets,
    identity_collate,
)
from nemo_aligner.data.nlp.config import DPODataConfig
from nemo_aligner.experimental.run.configs.validate import _validate_config

### nemo2 things
## TODO: move these elsewhere?
from nemo_aligner.experimental.run.gpt_dpo import (
    default_dpo_config,
    default_dpo_parallelism,
    default_dpo_trainer,
    gpt_config_overrides,
    wandb_logger,
)
from nemo_aligner.models.nlp.gpt.megatron_gpt_dpo_model import MegatronGPTDPOModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.nemo2.checkpoint import AlignerCheckpointIO
from nemo_aligner.utils.nemo2.config_utils import maybe_override
from nemo_aligner.utils.nemo2.optim import MegatronOptimizer
from nemo_aligner.utils.nemo2.train_script_utils import setup_distributed
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
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu


def dpo_loop(
    #
    restore_from_path: str,
    model_cls: type[GPTModel],  # TODO more precise typing since unexected args in __init__
    data_config: DPODataConfig,
    optimizer: MegatronOptimizer,
    # trainer_cls: type,  # TODO placeholder
    tp: int = 1,
    pp: int = 1,
    vp: int | None = None,
) -> str:
    mp.set_start_method("spawn", force=True)

    #################
    # MODEL RESTORE #
    #################
    ## load original config, initialize new dpo model, then restore weights from dir
    ## TODO: move this to a helper function?
    ## get this working. Getting serialization error right now
    """gpt_config = io.load_context(restore_from_path, subpath="model.config")
    parallelism_config = io.load_context(restore_from_path, subpath="trainer.strategy.parallelism")"""
    loaded = io.load_context(restore_from_path)
    gpt_config = loaded.model.config
    parallelism_config = loaded.trainer.strategy.parallelism
    tokenizer = loaded.model.tokenizer

    gpt_config = maybe_override(gpt_config, gpt_config_overrides())

    override_config = default_dpo_parallelism()
    override_config.tensor_model_parallel_size = tp
    override_config.pipeline_model_parallel_size = pp
    override_config.virtual_pipeline_model_parallel_size = vp

    parallelism_config = maybe_override(parallelism_config, override_config)

    ## assuming we are using the same tokenizer as the base model
    # tokenizer = io.load_context(restore_from_path, subpath="model.tokenizer")

    ################
    # LOGGER SETUP #
    ################
    logger = CustomLoggerWrapper([wandb_logger()])

    #############
    # INIT PEFT #
    #############
    ## init peft (TODO)

    # pull values from checkpoint (??)
    """trainer_restore_path = trainer.ckpt_path

    ## TODO: figure out what this is doing
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0"""

    ####################
    # INIT DISTRIBUTED #
    ####################
    ## intialize distributed
    # init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    setup_distributed(
        parallelism_config, data_config,
    )

    ## setup optimizer and scheduler
    # TODO: connect this to the model
    # opt_cfg, scheduler = default_dpo_optimizer()
    # optimizer = MegatronOptimizer(opt_cfg, lr_scheduler=scheduler,)

    ##############
    # INIT MODEL #
    ##############
    model = model_cls(
        config=gpt_config, dpo_config=default_dpo_config(), data_config=data_config, tokenizer=tokenizer,
    )

    ## make parallelism in model config match parallelism config
    gpt_config = set_model_parallel_attributes(model, parallelism_config)

    model.build_model(virtual_pipeline_model_parallel_size=parallelism_config.virtual_pipeline_model_parallel_size,)

    #####################
    # INIT CHECKPOINTER #
    #####################
    ## TODO: make configurable
    checkpointer = AlignerCheckpointIO(model, ckpt_load_strictness="log_all",)

    ## restore from base checkpoint
    ## do not restore the optimizer states
    checkpointer.load_checkpoint(
        restore_from_path, load_optim=False,
    )

    ## TODO: update once we have peft support
    if True:  # cfg.model.peft.peft_scheme == "none":
        ref_policy_state_dict = retrieve_model_state_dict_in_cpu(model, megatron_amp_O2=False,)  ## TODO: configure
        model.ref_policy_state_dict = ref_policy_state_dict

    ####################
    # INIT DATALOADERS #
    ####################
    train_dataloader, val_dataloader, global_collate_fn = data_config.build_dataloaders(model.tokenizer)

    ## initialize PTL-related things, call train start hooks
    # init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)

    ## log hparams (TODO)
    # logger.log_hyperparams(OmegaConf.to_container(cfg))

    ## set run timer (TODO)
    # timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)
    timer = None

    ####################
    # VALIDATE CONFIGS #
    ####################
    _validate_config(model=model, data=data_config)

    ################
    # INIT TRAINER #
    ################
    dpo_trainer = default_dpo_trainer()(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        collate_fn=partial(
            global_collate_fn,
            eos_id=model.tokenizer.eos_id,
            reset_position_ids=data_config.reset_position_ids,
            reset_attention_mask=data_config.reset_attention_mask,
            eod_mask_loss=data_config.eod_mask_loss,
            pad_length_to_multiple_of=data_config.pad_length_to_multiple_of,
        ),
        logger=logger,  ## TODO
        ckpt=checkpointer,
        run_timer=timer,
    )

    ## load custom trainer state dict (??)
    """if custom_trainer_state_dict is not None:
        dpo_trainer.load_state_dict(custom_trainer_state_dict)"""

    ###############
    # RUN TRAINER #
    ###############
    dpo_trainer.fit()

    return "TODO: return path"
