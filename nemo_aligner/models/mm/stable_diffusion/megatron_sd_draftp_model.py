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

from typing import Mapping

import numpy as np
import torch
import wandb
from omegaconf.dictconfig import DictConfig
from PIL import Image
from tqdm import tqdm
from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches
import nemo.collections.multimodal.parts.stable_diffusion.pipeline as sampling_utils
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    LatentDiffusion,
    MegatronLatentDiffusion,
)
from megatron.core.tensor_parallel.random import get_data_parallel_rng_tracker_name, get_cuda_rng_tracker
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils.train_utils import grad_reductions, prepare_for_training_step
from nemo_aligner.utils.utils import configure_batch_sizes
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

HAVE_MEGATRON_CORE = True


BatchType = Mapping[str, torch.tensor]

def _get_autocast_dtype(precision: str):
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    if precision in [32, "32", "32-true"]:
        return torch.float
    if precision in [16, "16", "16-mixed"]:
        return torch.half
    raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')

def calculate_gaussian_kl_penalty_shared_var(curr_eps, init_eps):

        diff = curr_eps - init_eps
        kl = torch.sum(diff ** 2, dim=(1, 2, 3, 4), keepdim=True).flatten()
        dimensionality = torch.numel(curr_eps[0])
        kl /= dimensionality

        return kl

class MegatronSDDRaFTPModel(MegatronLatentDiffusion, SupervisedInterface):
    def __init__(self, cfg, trainer):

        super().__init__(cfg, trainer=trainer)

        self.init_model = LatentDiffusion(cfg, None).to(torch.cuda.current_device()).eval()
        self.cfg = cfg
        self.with_distributed_adam = self.with_distributed_adam
        self.model.first_stage_model.requires_grad_(False)
        self.distributed_adam_offload_manager = None
        self.vae_batch_size = self.cfg.infer.get("vae_batch_size", 8)
        self.height = self.cfg.infer.get("height", 512)
        self.width = self.cfg.infer.get("width", 512)
        self.downsampling_factor = self.cfg.infer.get("down_factor", 8)
        self.in_channels = self.model.model.diffusion_model.in_channels
        self.unconditional_guidance_scale = self.cfg.infer.get("unconditional_guidance_scale", 7.5)
        self.sampler_type = self.cfg.infer.get("sampler_type", "DDIM")
        self.inference_steps = self.cfg.infer.get("inference_steps", 50)
        self.eta = self.cfg.infer.get("eta", 0)
        self.autocast_dtype = _get_autocast_dtype(self.cfg.precision)
        # Required by nemo_aligner/utils/train_utils
        self.model.initialize_ub = False
        self.model.rampup_batch_size = False
        self.model.with_distributed_adam = False

    def finish_inference(self):
        return

    def finish_training_step(self):
        grad_reductions(self)

    def infer(self):
        return

    def prepare_for_inference(self):
        return

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self.model, zero_grad=False)

    @torch.no_grad()
    def generate_log_images(self, latents, batch, model):

        with torch.cuda.amp.autocast(
            enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
        ):

            sampler = sampling_utils.initialize_sampler(model, self.sampler_type.upper())

            batch_size = len(batch)
            cond, u_cond = sampling_utils.encode_prompt(
                model.cond_stage_model, batch, self.unconditional_guidance_scale
            )

            _, intermediates = sampler.sample(
                S=self.inference_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=list(latents.shape[:1] + latents.shape[2:]),
                verbose=False,
                unconditional_guidance_scale=self.unconditional_guidance_scale,
                unconditional_conditioning=u_cond,
                eta=self.eta,
                x_T=latents,
                log_every_t=1,
                truncation_steps=0,
                return_logprobs=False,
                return_mean_var=False,
            )

            # # transforms trajectores from a list of T+1 (b x image) tensors to a tensor of shape (b * (T+1), ...)
            trajectories_predx0 = (
                torch.stack(intermediates["pred_x0"], dim=0)
                .transpose(0, 1)
                .contiguous()
                .view(-1, *intermediates["pred_x0"][0].shape[1:])
            )

            vae_decoder_output = []
            idx_denoised_imgs = [self.inference_steps + (self.inference_steps + 1) * i for i in range(batch_size)]

            for i in range(0, batch_size, self.vae_batch_size):
                image = self.model.differentiable_decode_first_stage(
                    trajectories_predx0[idx_denoised_imgs[i : i + self.vae_batch_size]]
                )

                vae_decoder_output.append(image)

            vae_decoder_output = torch.cat(vae_decoder_output, dim=0)
            vae_decoder_output = torch.clip((vae_decoder_output + 1) / 2, 0, 1) * 255.0

            log_reward = [
                self.reward_model.get_reward(
                    vae_decoder_output[i].unsqueeze(0).detach().permute(0, 2, 3, 1), batch # (C, H, W) -> (1, H, W, C)
                ).item()
                for i in range(batch_size)
            ]
            log_img = [
                np.transpose(vae_decoder_output[i].float().detach().cpu().numpy(), (1, 2, 0)) # (C, H, W) -> (H, W, C)
                for i in range(batch_size)
            ]
            return log_img, log_reward

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

        image_draft_p, reward_draft_p = self.generate_log_images(latents, prompts, self.model)
        image_init, reward_init = self.generate_log_images(latents, prompts, self.init_model)

        images = []
        captions = []
        for i in range(len(image_draft_p)):
            images.append(image_draft_p[i])
            images.append(image_init[i])
            captions.append("DRaFT+: " + prompts[i] + ", Reward = " + str(reward_draft_p[i]))
            captions.append("SD: " + prompts[i] + ", Reward = " + str(reward_init[i]))

        self.wandb_logger.loggers[1].log_image(
            key="Inference Images",
            images=[wandb.Image(Image.fromarray(img.round().astype("uint8"))) for img in images],
            caption=captions,
        )

    def generate(
        self, batch, x_T,
    ):

        with torch.cuda.amp.autocast(
            enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
        ):

            batch_size = len(batch)
            prev_img_draft_p = x_T

            device_draft_p = self.model.betas.device

            sampler_draft_p = sampling_utils.initialize_sampler(self.model, self.sampler_type.upper())
            sampler_init = sampling_utils.initialize_sampler(self.init_model, self.sampler_type.upper())

            cond, u_cond = sampling_utils.encode_prompt(
                self.model.cond_stage_model, batch, self.unconditional_guidance_scale
            )

            sampler_draft_p.make_schedule(ddim_num_steps=self.inference_steps, ddim_eta=self.eta, verbose=False)
            sampler_init.make_schedule(ddim_num_steps=self.inference_steps, ddim_eta=self.eta, verbose=False)

            timesteps = sampler_draft_p.ddim_timesteps

            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]

            print(f"Running {sampler_draft_p.sampler.name} Sampling with {total_steps} timesteps")
            iterator = tqdm(time_range, desc=f"{sampler_draft_p.sampler.name} Sampler", total=total_steps)

            list_eps_draft_p = []
            list_eps_init = []
            truncation_steps = self.cfg.truncation_steps

            denoise_step_kwargs = {
                "unconditional_guidance_scale": self.unconditional_guidance_scale,
                "unconditional_conditioning": u_cond,
            }
            for i, step in enumerate(iterator):

                denoise_step_args = [total_steps, i, batch_size, device_draft_p, step, cond]

                if i < total_steps - truncation_steps:
                    with torch.no_grad():
                        img_draft_p, pred_x0_draft_p, eps_t_draft_p = sampler_draft_p.single_ddim_denoise_step(
                            prev_img_draft_p, *denoise_step_args, **denoise_step_kwargs
                        )

                        prev_img_draft_p = img_draft_p
                else:

                    img_draft_p, pred_x0_draft_p, eps_t_draft_p = sampler_draft_p.single_ddim_denoise_step(
                        prev_img_draft_p, *denoise_step_args, **denoise_step_kwargs
                    )
                    list_eps_draft_p.append(eps_t_draft_p)

                    with torch.no_grad():
                        _, _, eps_t_init = sampler_init.single_ddim_denoise_step(
                            prev_img_draft_p, *denoise_step_args, **denoise_step_kwargs
                        )
                        list_eps_init.append(eps_t_init)

                    prev_img_draft_p = img_draft_p

            last_states = [pred_x0_draft_p]

            trajectories_predx0 = (
                torch.stack(last_states, dim=0).transpose(0, 1).contiguous().view(-1, *last_states[0].shape[1:])
            )
            t_eps_draft_p = torch.stack(list_eps_draft_p).to(torch.device("cuda"))
            t_eps_init = torch.stack(list_eps_init).to(torch.device("cuda"))

            vae_decoder_output = []
            for i in range(0, batch_size, self.vae_batch_size):
                image = self.model.differentiable_decode_first_stage(trajectories_predx0[i : i + self.vae_batch_size])
                vae_decoder_output.append(image)

            vae_decoder_output = torch.cat(vae_decoder_output, dim=0)
            vae_decoder_output = torch.clip((vae_decoder_output + 1) / 2, 0, 1) * 255.0

            return vae_decoder_output, t_eps_draft_p, t_eps_init

    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()

    def onload_adam_states(self):
        if self.distributed_adam_offload_manager is not None:
            # load back onto GPU
            self.distributed_adam_offload_manager.__exit__(None, None, None)

        self.distributed_adam_offload_manager = None

    def finish_training(self):
        """no need to offload adam states here
        """

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(batch, model):

            batch_size = len(batch)
            torch.cuda.manual_seed(torch.distributed.get_rank())
            torch.manual_seed(torch.distributed.get_rank())
            latents = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    self.height // self.downsampling_factor,
                    self.width // self.downsampling_factor,
                ],
                generator=None,
            ).to(torch.cuda.current_device())

            output_tensor_draft_p, epsilons_draft_p, epsilons_init = self.generate(batch, latents)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor_draft_p = output_tensor_draft_p.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor_draft_p):
                # Loss per micro batch (ub).
                if self.cfg.kl_coeff == 0.0:
                    kl_penalty = torch.tensor([0.0]).to(torch.cuda.current_device())
                else:
                    kl_penalty = calculate_gaussian_kl_penalty_shared_var(epsilons_draft_p, epsilons_init).mean()
                rewards = self.reward_model.get_reward(output_tensor_draft_p.permute(0, 2, 3, 1), batch) #(ub, H, W, C) -> (ub, C, H, W)
                loss = -rewards.mean() + kl_penalty * self.cfg.kl_coeff

                reduced_loss = average_losses_across_data_parallel_group([loss])
                reduced_kl_penalty = average_losses_across_data_parallel_group([kl_penalty])

                return (
                    loss,
                    {"loss": reduced_loss, "kl_penalty": reduced_kl_penalty},
                )

            return output_tensor_draft_p, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only=False):

        fwd_bwd_function = get_forward_backward_func()
        losses_reduced_per_micro_batch = fwd_bwd_function(
                    forward_step_func=self.get_forward_output_and_loss_func(),
                    data_iterator=[batch], 
                    model=self.model,
                    num_microbatches=get_num_microbatches(), 
                    forward_only=forward_only,
                    seq_length=None,
                    micro_batch_size=get_micro_batch_size(), 
        )

        if torch.distributed.get_rank() == 0 and len(self.wandb_logger.loggers) > 1:
            self.log_visualization(batch[0:1])

        metrics = {}

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
