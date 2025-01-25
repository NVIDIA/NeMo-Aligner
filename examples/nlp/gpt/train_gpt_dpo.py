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
import argparse
from functools import partial

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
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
## TODO: move these elsewhere?
from examples.nlp.gpt.conf.nemo2.gpt_dpo import (
    default_dpo_config,
    default_dpo_data_config,
    default_dpo_optimizer,
    default_dpo_parallelism,
    default_dpo_trainer,
    gpt_config_overrides,
)
from nemo_aligner.utils.nemo2.checkpoint import AlignerCheckpointIO
from nemo_aligner.utils.nemo2.config_utils import maybe_override
from nemo_aligner.utils.nemo2.optim import MegatronOptimizer
from nemo_aligner.utils.nemo2.train_script_utils import setup_distributed

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)

def get_args():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument('--restore-from-path', type=str, required=True, help="Path to the base model to restore")
    parser.add_argument('--tp', type=int, default=1, help="Tensor parallel size")
    parser.add_argument('--pp', type=int, default=1, help="Pipeline parallel size")
    parser.add_argument('--vp', type=int, default=None, help="VPP size")

    return parser.parse_args()

def main() -> None:

    args = get_args()

    restore_from_path = args.restore_from_path

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
    override_config.tensor_model_parallel_size = args.tp
    override_config.pipeline_model_parallel_size = args.pp
    override_config.virtual_pipeline_model_parallel_size = args.vp

    parallelism_config = maybe_override(parallelism_config, override_config)
    
    ## assuming we are using the same tokenizer as the base model
    #tokenizer = io.load_context(restore_from_path, subpath="model.tokenizer")

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

    data_config = default_dpo_data_config()

    ## hard-code data prefixes for now
    data_config.data_prefix = {
        "train": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"],
        "validation": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"],
        "test": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"]
    }

    ## intialize distributed
    #init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    setup_distributed(
        parallelism_config,
        data_config,
    )

    ## setup optimizer and scheduler
    # TODO: connect this to the model
    opt_cfg, scheduler = default_dpo_optimizer()
    optimizer = MegatronOptimizer(
        opt_cfg,
        lr_scheduler=scheduler,
    )

    ## initialize the model
    model = MegatronGPTDPOModel(
        config=gpt_config,
        dpo_config=default_dpo_config(),
        data_config=data_config,
        optim=optimizer, ## TODO: where is this used?
        tokenizer=tokenizer,
    )

    ## make parallelism in model config match parallelism config
    gpt_config = set_model_parallel_attributes(model, parallelism_config)

    model.build_model(
        virtual_pipeline_model_parallel_size=parallelism_config.virtual_pipeline_model_parallel_size,
    )

    ## TODO: make configurable
    checkpointer = AlignerCheckpointIO(
        model,
        ckpt_load_strictness="log_all",
    )

    ## restore from base checkpoint
    ## do not restore the optimizer states
    checkpointer.load_checkpoint(
        restore_from_path,
        load_optim=False,
    )

    ## TODO: update once we have peft support
    if True: #cfg.model.peft.peft_scheme == "none":
        ref_policy_state_dict = retrieve_model_state_dict_in_cpu(
            model, megatron_amp_O2=False, ## TODO: configure
        )
        model.ref_policy_state_dict = ref_policy_state_dict

    # use the entire dataset
    train_valid_test_num_samples = [-1 * data_config.global_batch_size] * 3
    ## build the dataset (should be mostly unchanged from before, except the config)
    if data_config.data_impl == "packed_jsonl":
        build_fn = build_train_valid_test_dpo_packed_datasets
    else:
        build_fn = build_train_valid_test_dpo_datasets
    train_ds, validation_ds, _ = build_fn(
        cfg=data_config,
        data_prefix=data_config.data_prefix,
        data_impl=data_config.data_impl,
        splits_string=data_config.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=gpt_config.seq_length, ## TODO: check
        seed=data_config.seed, ## TODO: check
        tokenizer=model.tokenizer, ## TODO: tokenizer
    )

    collate = train_ds.global_collate_fn if data_config.data_impl == "packed_jsonl" else dpo_custom_collate
    train_dataloader = build_dataloader(
        cfg=data_config,
        dataset=train_ds,
        consumed_samples=0, #consumed_samples, ## TODO: UPDATE
        mbs=data_config.micro_batch_size,
        gbs=data_config.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=identity_collate,
    )

    val_dataloader = build_dataloader(
        cfg=data_config,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=data_config.micro_batch_size,
        gbs=data_config.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=identity_collate,
        use_random_sampler=False,
    )

    ## initialize PTL-related things, call train start hooks
    #init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)

    ## log hparams (TODO)
    #logger.log_hyperparams(OmegaConf.to_container(cfg))

    ## set run timer (TODO)
    #timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)
    timer = None

    ## initialize DPO trainer
    dpo_trainer = default_dpo_trainer()(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        collate_fn=partial(
            collate,
            eos_id=model.tokenizer.eos_id,
            reset_position_ids=data_config.reset_position_ids,
            reset_attention_mask=data_config.reset_attention_mask,
            eod_mask_loss=data_config.eod_mask_loss,
            pad_length_to_multiple_of=data_config.pad_length_to_multiple_of,
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
