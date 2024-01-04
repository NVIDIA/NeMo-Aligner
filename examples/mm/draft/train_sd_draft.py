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
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.draft import DraftTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, build_train_valid_test_rm_datasets
from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion, LatentDiffusion
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    retrieve_custom_trainer_state_dict,
)
from megatron.core.utils import divide
from megatron.core import parallel_state
import torch
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
import sys
from nemo_aligner.utils.utils import load_from_nemo
from nemo_aligner.models.mm.draft.alignable_sd_model import AlignableSDModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
import torch
from nemo_aligner.data.mm import text_webdataset
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu
from nemo_aligner.models.mm.draft.draft_RMs import get_reward_model

mp.set_start_method("spawn", force=True)

#TODO @geshen: to get the trainer builder for stable diffusion, training script for the Stable Diffusion has an override for training strategy,
#             the following is brought from sd_train.py. Is that okay?

class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        ddp_overlap = self.cfg.model.get('ddp_overlap', True)
        if ddp_overlap:
            return NLPDDPStrategy(
                no_ddp_communication_hook=False,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=True,
                bucket_cap_mb=256,
            )
        else:
            return NLPDDPStrategy(
                no_ddp_communication_hook=True,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
            )


@hydra_runner(config_path="conf", config_name="train_sd_draft")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    #TODO: Check if tf32 is necassary. Brought from stable diffusion training script (sd_train.py)
    torch.backends.cuda.matmul.allow_tf32 = True
    #TODO: @ataghibakhsh Fix the dataset_path issue
    cfg.model.data.train.dataset_path = [cfg.model.data.webdataset.local_root_path for _ in range(cfg.trainer.devices)]
    #TODO @geshen: Please check if the following is okay to get the trainer; this is how stable diffusion training script builds the trainer
    trainer = MegatronStableDiffusionTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)
    #TODO: @geshen: For the Stable Diffusion, I am currently getting the vae and unet separately
    ptl_model = MegatronLatentDiffusion(cfg.model, trainer).to(torch.cuda.current_device())

    if cfg.model.get('peft', None):
        if cfg.model.peft.enable:
            peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

            if cfg.model.peft.restore_from_path is not None:
                # initialize peft weights from a checkpoint instead of randomly
                # This is not the same as resume training because optimizer states are not restored.
                logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
                ptl_model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(cfg.model))

            elif peft_cfg_cls is not None:
                logging.info("Adding adapter weights to the model for PEFT")
                ptl_model.add_adapter(peft_cfg_cls(cfg.model))
            else:
                logging.info(f"Running full finetuning since no peft scheme is given.\n{ptl_model.summarize()}")
    
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    trainer_restore_path = trainer.ckpt_path

    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0
    
    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_dataset, val_dataset = text_webdataset.build_train_valid_datasets(cfg.model, consumed_samples = 0)
    train_dataset = [d['captions'] for d in list(train_dataset)] #TODO  make with torch Dataset 

    train_dataloader = build_dataloader(
        cfg,
        dataset = train_dataset,
        consumed_samples = consumed_samples,
        mbs = cfg.model.micro_batch_size,
        gbs = cfg.model.global_batch_size,
        load_gbs=True,
    )

    if cfg.model.get("transformer_engine", False):
        ptl_model.setup_transformer_engine_tp_groups()

    ptl_model.setup()
    trainer.strategy._lightning_module = ptl_model

    dummy_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=divide(cfg.model.global_batch_size, parallel_state.get_data_parallel_world_size())
    )

    init_using_ptl(trainer, ptl_model, dummy_train_dataloader, train_dataset)
    # make sure the dummy train dataloader is never used
    del ptl_model._train_dl
    del dummy_train_dataloader
    
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    #TODO: What would be cleanest way to add the RM? 
    #TODO: @ataghibakhsh Later replace with NeMo RMs

    reward_model = get_reward_model(cfg.RM, mbs=cfg.model.micro_batch_size, gbs=cfg.model.global_batch_size)
    

    alignable_model = AlignableSDModel(ptl_model.model, reward_model, ptl_model.tokenizer, optimizer, cfg.model, ptl_model, logger=logger)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    draft_trainer = DraftTrainer(
        cfg=cfg.model,
        model=alignable_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        test_dataloader=None,
        ckpt_callback=ckpt_callback,
    )

    if custom_trainer_state_dict is not None:
        draft_trainer.load_state_dict(custom_trainer_state_dict)

    draft_trainer.fit()


if __name__ == "__main__":
    main()
