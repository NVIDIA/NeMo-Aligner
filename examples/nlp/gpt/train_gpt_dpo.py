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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.dpo import DPOTrainer, dpo_custom_collate
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_dpo_datasets,
    build_train_valid_test_dpo_packed_datasets,
    identity_collate,
)
from nemo_aligner.models.nlp.gpt.megatron_gpt_dpo_model import MegatronGPTDPOModel
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
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu

### nemo2 things
from examples.nlp.gpt.conf.nemo2.gpt_dpo import (
    default_dpo_config,
    default_dpo_data_config,
    default_dpo_optimizer,
    default_dpo_trainer,
    gpt_config_overrides,
)
from nemo_aligner.utils.nemo2.checkpoint import AlignerCheckpointIO
from nemo_aligner.utils.nemo2.config_utils import maybe_override
from nemo_aligner.utils.nemo2.train_script_utils import setup_distributed

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="gpt_dpo")
def main(cfg) -> None:

    ## load original config, initialize new dpo model, then restore weights from dir
    gpt_config = io.load_context(input_path, subpath="model.config")
    gpt_config = maybe_override(gpt_config, gpt_config_overrides())

    ## set logger and exp manager (TODO)

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

    ## intialize distributed
    #init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    setup_distributed()

    ## initialize the model
    model = MegatronGPTDPOModel(
        gpt_config,
        default_dpo_config,
    )

    model.build_model(
        ## TODO: parallel_config
        virtual_pipeline_model_parallel_size = parallel_config.virtual_pipeline_model_parallel_size,
    )

    ## TODO: make configurable
    checkpointer = AlignerCheckpointIO(
        model,
        ckpt_load_strictness="log_all",
    )

    ## build the dataset (should be mostly unchanged from before, except the config)
    data_cfg = default_dpo_data_config()
    if data_cfg.data_impl == "packed_jsonl":
        build_fn = build_train_valid_test_dpo_packed_datasets
    else:
        build_fn = build_train_valid_test_dpo_datasets
    train_ds, validation_ds, _ = build_fn(
        cfg=data_cfg,
        data_prefix=data_cfg.data_prefix,
        data_impl=data_cfg.data_impl,
        splits_string=data_cfg.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=gpt_config.seq_length, ## TODO: check
        seed=gpt_config.seed, ## TODO: check
        tokenizer=model.tokenizer, ## TODO: tokenizer
    )

    collate = train_ds.global_collate_fn if cfg.model.data.data_impl == "packed_jsonl" else dpo_custom_collate
    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=gpt_config.micro_batch_size,
        gbs=gpt_config.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=identity_collate,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=gpt_config.micro_batch_size,
        gbs=gpt_config.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=identity_collate,
        use_random_sampler=False,
    )

    ## initialize PTL-related things, call train start hooks
    #init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    
    ## setup optimizer and scheduler
    # TODO: connect this to the model
    opt_cfg, scheduler = default_dpo_optimizer()
    optimizer = MegatronOptimizer(
        opt_cfg,
        lr_scheduler=scheduler,
    )

    ## log hparams (TODO)
    #logger.log_hyperparams(OmegaConf.to_container(cfg))

    ## set run timer (TODO)
    #timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)
    timer = None

    ## initialize DPO trainer
    dpo_trainer = default_dpo_trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        collate_fn=partial(
            collate,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
            pad_length_to_multiple_of=cfg.model.data.get("pad_length_to_multiple_of", None),
        ),
        logger=None, ## TODO
        ckpt=checkpointer,
        run_timer=timer,
    )

    ## load custom trainer state dict (??)
    """if custom_trainer_state_dict is not None:
        dpo_trainer.load_state_dict(custom_trainer_state_dict)"""

    dpo_trainer.fit()


if __name__ == "__main__":
    main()
