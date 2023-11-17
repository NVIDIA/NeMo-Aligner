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

from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.critic_server_trainer import CriticServerTrainer
from nemo_aligner.models.nlp.gpt.megatron_gpt_critic import MegatronGPTCriticModel
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import (
    load_and_override_model_config,
    load_from_nemo,
    retrieve_model_state_dict_in_cpu,
    set_autocast_gpu_dtype,
)

"""This is the script to start the critic inference and training server
"""


@hydra_runner(config_path="conf", config_name="gpt_ppo_critic")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "ppo")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # needed for autocasting BF16
    set_autocast_gpu_dtype(cfg.trainer.precision)

    # load the pretrained RM
    ptl_model = load_from_nemo(
        MegatronGPTCriticModel,
        cfg.model,
        trainer,
        strict=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    if cfg.trainer.ppo.combine_rm_and_critic_server:
        # to run the critic and RM together
        # we move to CPU and swap them
        # so we need to retrieve the state here before PTL load
        rm_state_dict = retrieve_model_state_dict_in_cpu(
            ptl_model, megatron_amp_O2=cfg.model.get("megatron_amp_O2", False)
        )
        ptl_model.rm_state_dict = rm_state_dict

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path

    custom_trainer_state_dict = None
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    init_using_ptl(trainer, ptl_model, None, None)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)
    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    critic_trainer = CriticServerTrainer(
        cfg=cfg.trainer.ppo,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        ckpt_callback=ckpt_callback,
        gbs=cfg.model.global_batch_size,
    )

    if custom_trainer_state_dict is not None:
        critic_trainer.load_state_dict(custom_trainer_state_dict)

    critic_trainer.run_server()


if __name__ == "__main__":
    main()
