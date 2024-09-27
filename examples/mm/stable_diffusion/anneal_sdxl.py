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

import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from megatron.core.utils import divide
from omegaconf.omegaconf import OmegaConf, open_dict
from packaging.version import Version
from PIL import Image
from torch import nn

# checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import (
    DiffusionEngine,
    MegatronDiffusionEngine,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder import (
    AutoencoderKL,
    AutoencoderKLInferenceWrapper,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    LatentDiffusion,
    MegatronLatentDiffusion,
)
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.model import (
    AttnBlock,
    Decoder,
    Encoder,
    ResnetBlock,
)
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel import (
    ResBlock,
    SpatialTransformer,
    TimestepEmbedSequential,
    UNetModel,
)
from nemo.collections.multimodal.modules.stable_diffusion.encoders.modules import (
    FrozenCLIPEmbedder,
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPEmbedder2,
)
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPFSDPStrategy

# from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronStableDiffusionTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.mm import text_webdataset
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import MegatronCLIPRewardModel, get_reward_model
from nemo_aligner.models.mm.stable_diffusion.megatron_sdxl_draftp_model import MegatronSDXLDRaFTPModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    retrieve_custom_trainer_state_dict,
    temp_pop_from_config,
)

mp.set_start_method("spawn", force=True)


# TODO: this functionality should go into NeMo
# Specifically, the NeMo MegatronTrainerBuilder must also accept extra FSDP wrap modules so that it doesnt need to be subclassed
class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a DDP or a FSDP strategy passed to Trainer.strategy.  Copied from `sd_xl_train.py`
        """
        if self.cfg.model.get("fsdp", False):
            logging.info("FSDP.")
            assert (
                not self.cfg.model.optim.get("name") == "distributed_fused_adam"
            ), "Distributed optimizer cannot be used with FSDP."
            if self.cfg.model.get("megatron_amp_O2", False):
                logging.info("Torch FSDP is not compatible with O2 precision recipe. Setting O2 `False`.")
                self.cfg.model.megatron_amp_O2 = False

            # Check if its a full-finetuning or PEFT
            return NLPFSDPStrategy(
                limit_all_gathers=self.cfg.model.get("fsdp_limit_all_gathers", True),
                sharding_strategy=self.cfg.model.get("fsdp_sharding_strategy", "full"),
                cpu_offload=self.cfg.model.get("fsdp_cpu_offload", False),  # offload on is not supported
                grad_reduce_dtype=self.cfg.model.get("fsdp_grad_reduce_dtype", 32),
                precision=self.cfg.trainer.precision,
                ## nn Sequential is supposed to capture the `t_embed`, `label_emb`, `out` layers in the unet
                extra_fsdp_wrap_module={
                    UNetModel,
                    TimestepEmbedSequential,
                    Decoder,
                    ResnetBlock,
                    AttnBlock,
                    nn.Sequential,
                    MegatronCLIPRewardModel,
                    FrozenOpenCLIPEmbedder,
                    FrozenOpenCLIPEmbedder2,
                    FrozenCLIPEmbedder,
                    ParallelLinearAdapter,
                },
                use_orig_params=False,
                set_buffer_dtype=self.cfg.get("fsdp_set_buffer_dtype", None),
            )

        return NLPDDPStrategy(
            no_ddp_communication_hook=(not self.cfg.model.get("ddp_overlap")),
            gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )


def resolve_and_create_trainer(cfg, pop_trainer_key):
    """resolve the cfg, remove the key before constructing the PTL trainer
        and then restore it after
    """
    OmegaConf.resolve(cfg)
    with temp_pop_from_config(cfg.trainer, pop_trainer_key):
        return MegatronStableDiffusionTrainerBuilder(cfg).create_trainer()


@hydra_runner(config_path="conf", config_name="draftp_sdxl")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # set cuda device for each process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # turn off wandb logging
    cfg.exp_manager.create_wandb_logger = False

    if Version(torch.__version__) >= Version("1.12"):
        torch.backends.cuda.matmul.allow_tf32 = True
    cfg.model.data.train.dataset_path = [
        cfg.model.data.webdataset.local_root_path for _ in range(cfg.trainer.devices * cfg.trainer.num_nodes)
    ]
    cfg.model.data.validation.dataset_path = [
        cfg.model.data.webdataset.local_root_path for _ in range(cfg.trainer.devices * cfg.trainer.num_nodes)
    ]

    trainer = resolve_and_create_trainer(cfg, "draftp_sd")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)
    # Instatiating the model here
    ptl_model = MegatronSDXLDRaFTPModel(cfg.model, trainer).to(torch.cuda.current_device())
    init_peft(ptl_model, cfg.model)  # init peft

    trainer_restore_path = trainer.ckpt_path

    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # use the validation ds if needed
    train_ds, validation_ds = text_webdataset.build_train_valid_datasets(
        cfg.model.data, consumed_samples=consumed_samples
    )
    validation_ds = [d["captions"] for d in list(validation_ds)]

    val_dataloader = build_dataloader(
        cfg,
        dataset=validation_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
    )
    init_using_ptl(trainer, ptl_model, val_dataloader, validation_ds)

    if cfg.model.get("activation_checkpointing", False):
        # call activation checkpointing here
        # checkpoint wrapper
        logging.info("Applying activation checkpointing on UNet and Decoder.")
        non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def checkpoint_check_fn(module):
            return isinstance(module, (Decoder, UNetModel, MegatronCLIPRewardModel))

        apply_activation_checkpointing(
            ptl_model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=checkpoint_check_fn
        )

    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    torch.distributed.barrier()

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)
    timer = Timer(cfg.exp_manager.get("max_time_per_run", None) if cfg.exp_manager else None)

    draft_p_trainer = SupervisedTrainer(
        cfg=cfg.trainer.draftp_sd,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=val_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=[],
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
        run_init_validation=True,
    )

    if custom_trainer_state_dict is not None:
        draft_p_trainer.load_state_dict(custom_trainer_state_dict)

    torch.cuda.empty_cache()

    if cfg.get("prompt") is not None:
        logging.info(f"Override val dataset with custom prompt: {cfg.prompt}")
        val_dataloader = [[cfg.prompt]]

    wt_types = cfg.get("weight_type", None)
    if wt_types is None:
        wt_types = ["base", "draft", "linear", "power_2", "power_4", "step_0.6"]
    else:
        wt_types = wt_types.split(",") if isinstance(wt_types, str) else wt_types
    logging.info(f"Running on types: {wt_types}")

    # run for all weight types
    for wt_type in wt_types:
        global_idx = 0
        if wt_type == "base":
            # dummy function that assigns a value of 0 all the time
            logging.info("using the base model")
            wt_draft = lambda sigma, sigma_next, i, total: 0
        elif wt_type == "linear":
            wt_draft = lambda sigma, sigma_next, i, total: i * 1.0 / total
        elif wt_type == "draft":
            wt_draft = lambda sigma, sigma_next, i, total: 1
        elif wt_type.startswith("power"):  # its of the form power_{power}
            pow = float(wt_type.split("_")[1])
            wt_draft = lambda sigma, sigma_next, i, total: (i * 1.0 / total) ** pow
        elif wt_type.startswith("step"):  # use a step function (step_{p})
            frac = float(wt_type.split("_")[1])
            wt_draft = lambda sigma, sigma_next, i, total: float((i * 1.0 / total) >= frac)
        else:
            raise ValueError(f"invalid weighing type: {wt_type}")
        logging.info(f"using weighing type for annealed outputs: {wt_type}.")

        # initialize generator
        annealed_out_dir = os.path.join(cfg.exp_manager.explicit_log_dir, f"annealed_outputs_sdxl_{wt_type}/")
        # generate random seed for reproducibility and make output dir
        gen = torch.Generator(device="cpu")
        gen.manual_seed((1243 + 1247837 * local_rank) % (int(2 ** 32 - 1)))
        os.makedirs(annealed_out_dir, exist_ok=True)

        for batch in val_dataloader:
            batch_size = len(batch)
            with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                latents = torch.randn(
                    [
                        batch_size,
                        ptl_model.in_channels,
                        ptl_model.height // ptl_model.downsampling_factor,
                        ptl_model.width // ptl_model.downsampling_factor,
                    ],
                    generator=gen,
                ).to(torch.cuda.current_device())
            images = ptl_model.annealed_guidance(batch, latents, weighing_fn=wt_draft)
            images = (
                images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            )  # outputs are already scaled from [0, 255]
            # save to pil
            for i in range(images.shape[0]):
                i = i + global_idx
                img_path = os.path.join(annealed_out_dir, f"img_{i:05d}_{local_rank:02d}.png")
                prompt_path = os.path.join(annealed_out_dir, f"prompt_{i:05d}_{local_rank:02d}.txt")
                Image.fromarray(images[i]).save(img_path)
                with open(prompt_path, "w") as fi:
                    fi.write(batch[i])
            # increment global index
            global_idx += batch_size
        logging.info("Saved all images.")


if __name__ == "__main__":
    main()
