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

from copy import deepcopy
from typing import Mapping

import numpy as np
import torch
import wandb
from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from tqdm import tqdm

import nemo.collections.multimodal.parts.stable_diffusion.pipeline as sampling_utils
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import (
    DiffusionEngine,
    MegatronDiffusionEngine,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    LatentDiffusion,
    MegatronLatentDiffusion,
)
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_helpers import (
    do_sample,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
)
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_pipeline import get_sampler_config
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
)
from nemo_aligner.utils.utils import (
    _get_autocast_dtype,
    adapter_control,
    configure_batch_sizes,
    get_iterator_k_split_list,
)

BatchType = Mapping[str, torch.tensor]


def calculate_gaussian_kl_penalty_shared_var(curr_eps, init_eps):
    diff = curr_eps - init_eps
    kl = torch.sum(diff ** 2, dim=(1, 2, 3, 4), keepdim=True).flatten()
    dimensionality = torch.numel(curr_eps[0])
    kl /= dimensionality

    return kl


class MegatronSDXLDRaFTPModel(MegatronDiffusionEngine, SupervisedInterface):
    def __init__(self, cfg, trainer):

        super().__init__(cfg, trainer=trainer)
        self.peft_scheme = cfg.peft.peft_scheme
        self.init_model = None
        if cfg.peft.peft_scheme == "none":
            logging.info("Full finetuning, initializing a copy of the base model.")
            self.init_model = DiffusionEngine(deepcopy(cfg), None).to(torch.cuda.current_device()).eval()
            for p in self.init_model.parameters():
                p.requires_grad = False
            self.init_model.train(mode=False)

        self.cfg = cfg
        self.with_distributed_adam = self.with_distributed_adam
        self.model.first_stage_model.requires_grad_(False)
        # delete encoder
        del self.model.first_stage_model.encoder

        self.distributed_adam_offload_manager = None
        self.in_channels = self.model.model.diffusion_model.in_channels
        self.height = self.cfg.sampling.base.get("height", 512)
        self.width = self.cfg.sampling.base.get("width", 512)
        self.downsampling_factor = 2 ** (
            len(self.cfg.first_stage_config.ddconfig.get("ch_mult", [0])) - 1
        )  # one less than the number of layers
        self.downsampling_factor = int(self.downsampling_factor)

        self.autocast_dtype = _get_autocast_dtype(self.cfg.precision)
        # Required by nemo_aligner/utils/train_utils
        self.initialize_ub = False
        self.model.initialize_ub = False
        self.model.rampup_batch_size = False
        self.model.with_distributed_adam = False

        params = self.cfg.sampling.base
        self.sampler = get_sampler_config(params)

    ##############
    ## Rewriting this to avoid ignoring the decoder, and just putting it in its own shard
    def configure_sharded_model(self):
        def find_frozen_submodules(model):
            # ignore quant modules in vae
            def _ignore_first_stage_quant(name):
                if "first_stage" in name:
                    if "quant" in name:
                        return True
                    return False
                return True

            # do not ignore clip embedders
            def _donot_ignore_embedders(name):
                if "conditioner" not in name:
                    return True
                if (
                    name == "conditioner" or name == "conditioner.embedders"
                ):  # do not ignore the main-level modules, we will ignore the submodules
                    return False
                else:
                    if "conditioner.embedders.0" in name:
                        return False
                    if "conditioner.embedders.1" in name:
                        return False
                    return True

            # do not ignore diffusion model, this has to be sharded
            def _donot_ignore_diffusion_model(name):
                if "diffusion_model" in name:
                    if "time_embed" in name or "label_emb" in name:  # ignore these modules for sharding
                        return True
                    # we will allow input_layers, middle_layers, output_layers, and out
                    return False
                else:
                    return True

            frozen_submodules = []
            frozen_submodule_names = []
            for name, module in model.named_modules():
                if (
                    isinstance(module, nn.Module)
                    and list(module.parameters())
                    and all(not param.requires_grad for param in module.parameters())
                    and (_ignore_first_stage_quant(name))
                    and (_donot_ignore_embedders(name))
                    and (_donot_ignore_diffusion_model(name))
                ):
                    frozen_submodule_names.append(name)
                    frozen_submodules.append(module)
            return frozen_submodule_names, frozen_submodules

        if self.use_fsdp:
            """Top-evel FSDP model sharding"""
            # Shard the top-level model hierarchically. We shard the strategy-unwrapped model not
            # to lose the structure of non-FSDP wrapped parameters (e.g, embedding)
            # TODO: Currently the main parameter data type is kept in fp32 (when O2=False). This needs to be
            # extended to support lower precision main parameters.
            frozen_submodule_names, frozen_submodules = find_frozen_submodules(self.model)
            for submodule in frozen_submodule_names:
                logging.debug(f"Ignoring state {submodule} in FSDP.")
            self.trainer.strategy.kwargs["ignored_states"] = frozen_submodules
            # FSDP requires uniform status of require_grads
            # Diffusion models like SD has frozen parts and needs to be added to 'ignored_states' from sharding for FSDP to work
            self.model = self.trainer.strategy._setup_model(self.model)
            # Move the CPU-initialized model (with `use_cpu_initialization=True`) to GPU, which is to avoid
            # out-of-memory carash before sharding. In case of GPU-initialized model, this is no-op.
            self.model = self.model.cuda(torch.cuda.current_device())

    def finish_inference(self):
        return

    def finish_training_step(self):
        grad_reductions(self)

    def infer(self, batch):
        return

    def prepare_for_inference(self):
        return

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self.model, zero_grad=False)

    def prepare_for_validation_step(self):
        """things to call to prepare for validation
        """
        prepare_for_validation_step(self)
        gbs = int(self.cfg.global_batch_size)
        mbs = int(self.cfg.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

    def finish_validation_step(self):
        """things to call after validation step ends
        """
        finish_validation_step(self)
        # restore the batch sizes for training
        gbs = int(self.cfg.global_batch_size)
        mbs = int(self.cfg.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

    def append_sdxl_size_keys(self, prompts):
        # given list of prompts, convert into a batch
        # also append size and crop keys
        batch_size = len(prompts)
        batch = dict()
        batch["txt"] = prompts
        batch["captions"] = prompts
        batch["original_size_as_tuple"] = torch.tensor(
            [self.width, self.height], device=torch.cuda.current_device()
        ).repeat(batch_size, 1)
        batch["target_size_as_tuple"] = torch.tensor(
            [self.width, self.height], device=torch.cuda.current_device()
        ).repeat(batch_size, 1)
        batch["crop_coords_top_left"] = torch.tensor([0, 0], device=torch.cuda.current_device()).repeat(batch_size, 1)
        return batch

    @torch.no_grad()
    def generate_log_images(self, latents, batch, model):

        with torch.cuda.amp.autocast(
            enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
        ):
            # get sampler (contains discretizer, scaler, and guider)
            sampler = self.sampler

            batch_c = self.append_sdxl_size_keys(batch)
            batch_size = len(batch)

            force_uc_zero_embeddings = ["txt", "captions"]  # force zero embeddings for text and captions
            cond, u_cond = model.conditioner.get_unconditional_conditioning(
                batch_c, batch_uc=None, force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            additional_model_inputs = {}  # not necessary for now

            denoiser = lambda input, sigma, c: model.denoiser(model.model, input, sigma, c, **additional_model_inputs)
            samples_z = sampler(denoiser, latents, cond=cond, uc=u_cond)
            samples_x = model.decode_first_stage(samples_z)
            vae_decoder_output = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0) * 255.0

            log_reward = [
                self.reward_model.get_reward(
                    vae_decoder_output[i].unsqueeze(0).detach().permute(0, 2, 3, 1), batch  # (C, H, W) -> (1, H, W, C)
                ).item()
                for i in range(batch_size)
            ]
            log_img = [
                vae_decoder_output[i].float().detach().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                for i in range(batch_size)
            ]
            return log_img, log_reward, vae_decoder_output

    @torch.no_grad()
    def log_visualization(self, prompts):
        batch_size = len(prompts)
        # Get different seeds for different dp rank
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            latents = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    self.height // self.downsampling_factor,
                    self.width // self.downsampling_factor,
                ],
                generator=None,
            ).to(torch.cuda.current_device())

        image_draft_p, reward_draft_p, vae_decoder_output_draft_p = self.generate_log_images(
            latents + 0, prompts, self.model
        )
        # run visualization with base model too
        base_model = self.init_model or self.model
        with adapter_control(base_model):
            image_init, reward_init, _ = self.generate_log_images(latents + 0, prompts, base_model)

        images = []
        captions = []
        for i in range(len(image_draft_p)):
            images.append(image_draft_p[i])
            images.append(image_init[i])
            captions.append("DRaFT+: " + prompts[i] + ", Reward = " + str(reward_draft_p[i]))
            captions.append("SDXL: " + prompts[i] + ", Reward = " + str(reward_init[i]))

        return vae_decoder_output_draft_p, images, captions

    def generate(
        self, batch, x_T,
    ):
        with torch.cuda.amp.autocast(
            enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
        ):
            batch_c = self.append_sdxl_size_keys(batch)

            truncation_steps = self.cfg.truncation_steps
            force_uc_zero_embeddings = ["txt", "captions"]
            
            sampler = self.sampler

            cond, uc = self.model.conditioner.get_unconditional_conditioning(
                batch_c, batch_uc=None, force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            additional_model_inputs = {}

            list_eps_draft = []
            list_eps_init = []

            # get denoisers
            denoiser_draft = lambda input, sigma, c: self.model.denoiser(
                self.model.model, input, sigma, c, **additional_model_inputs
            )
            base_model = self.init_model or self.model
            denoiser_base = lambda input, sigma, c: base_model.denoiser(
                base_model.model, input, sigma, c, **additional_model_inputs
            )

            # prep initial sampler config
            x = x_T + 0
            num_steps = sampler.num_steps
            x, s_in, sigmas, num_sigmas, cond, uc = sampler.prepare_sampling_loop(x, cond, uc, num_steps)
            # last step doesnt count since there is no additional sigma
            total_steps = num_sigmas - 1

            iterator = tqdm(range(num_sigmas - 1), desc=f"{sampler.__class__.__name__} Sampler", total=total_steps)
            for i in iterator:
                gamma = sampler.get_gamma(sigmas, num_sigmas, i)
                if i < total_steps - truncation_steps:
                    # just run the sampling without storing any grad
                    with torch.no_grad():
                        x_next_draft, eps_draft = sampler.sampler_step(
                            s_in * sigmas[i],
                            s_in * sigmas[i + 1],
                            denoiser_draft,
                            x,
                            cond,
                            uc,
                            gamma,
                            return_noise=True,
                        )
                        x = x_next_draft
                else:
                    # store computation graph of draft eps
                    x_next_draft, eps_draft = sampler.sampler_step(
                        s_in * sigmas[i], s_in * sigmas[i + 1], denoiser_draft, x, cond, uc, gamma, return_noise=True
                    )
                    list_eps_draft.append(eps_draft)
                    with torch.no_grad():
                        base_model = self.init_model or self.model
                        with adapter_control(base_model):
                            _, eps_init = sampler.sampler_step(
                                s_in * sigmas[i],
                                s_in * sigmas[i + 1],
                                denoiser_base,
                                x,
                                cond,
                                uc,
                                gamma,
                                return_noise=True,
                            )
                        list_eps_init.append(eps_init)
                    # set next \bar{x}
                    x = x_next_draft

            # compile list of eps
            t_eps_draft_p = torch.stack(list_eps_draft).to(torch.device("cuda"))
            t_eps_init = torch.stack(list_eps_init).to(torch.device("cuda"))

            # generate image from last sample
            image = self.model.differentiable_decode_first_stage(x)
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0) * 255.0

            return image, t_eps_draft_p, t_eps_init

    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )

    def finish_training(self):
        """no need to offload adam states here
        """

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(data_iterator, model):

            batch = next(data_iterator)
            batch_size = len(batch)
            # Get different seeds for different dp rank
            with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                latents = torch.randn(
                    [
                        batch_size,
                        self.in_channels,
                        self.height // self.downsampling_factor,
                        self.width // self.downsampling_factor,
                    ],
                    generator=None,
                ).to(torch.cuda.current_device())

            if validation_step:
                output_tensor_draft_p, images, captions = self.log_visualization(batch)
            else:
                output_tensor_draft_p, epsilons_draft_p, epsilons_init = self.generate(batch, latents)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor_draft_p = output_tensor_draft_p.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor_draft_p):
                # Loss per micro batch (ub).
                if self.cfg.kl_coeff == 0.0 or validation_step:
                    kl_penalty = torch.tensor([0.0]).to(torch.cuda.current_device())
                else:
                    kl_penalty = calculate_gaussian_kl_penalty_shared_var(epsilons_draft_p, epsilons_init).mean()
                rewards = self.reward_model.get_reward(
                    output_tensor_draft_p.permute(0, 2, 3, 1), batch
                )  # (ub, H, W, C) -> (ub, C, H, W)
                loss = -rewards.mean() + kl_penalty * self.cfg.kl_coeff

                reduced_loss = average_losses_across_data_parallel_group([loss])
                reduced_kl_penalty = average_losses_across_data_parallel_group([kl_penalty])

                metrics = {"loss": reduced_loss, "kl_penalty": reduced_kl_penalty}

                if validation_step:
                    metrics["images_and_captions"] = images, captions

                return (
                    loss,
                    metrics,
                )

            return output_tensor_draft_p, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only=False):
        data_iter = get_iterator_k_split_list(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=get_micro_batch_size(),
        )

        metrics = losses_reduced_per_micro_batch[0]
        for key in ["loss", "kl_penalty"]:
            if losses_reduced_per_micro_batch:
                metric_mean = torch.stack(
                    [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                ).mean()
            else:
                metric_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            torch.distributed.broadcast(metric_mean, get_last_rank())
            metrics[key] = metric_mean.cpu().item()

        return metrics["loss"], metrics
