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

from typing import Mapping

import numpy as np
import torch
import wandb
from omegaconf.dictconfig import DictConfig
from PIL import Image
from tqdm import tqdm

import nemo.collections.multimodal.parts.stable_diffusion.pipeline as sampling_utils
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import LatentDiffusion
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import AlignableGenerativeInterface
from nemo_aligner.utils.train_utils import prepare_for_training_step
from nemo_aligner.utils.utils import configure_batch_sizes

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

BatchType = Mapping[str, torch.tensor]


class MegatronSDDRaFTPModel(AlignableGenerativeInterface):
    def __init__(
        self, model, reward_model, tokenizer, optimizer, config: DictConfig, logger,
    ):
        self.model = model
        self.init_model = LatentDiffusion(config, None).to(torch.cuda.current_device()).eval()
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.cfg = config
        self.logger = logger
        self.with_distributed_adam = self.model.with_distributed_adam
        self.megatron_amp_O2 = self.cfg.get("megatron_amp_O2", False)
        self.model.megatron_amp_O2 = self.cfg.get("megatron_amp_O2", False)
        self.model.model.first_stage_model.requires_grad_(False)

        self.distributed_adam_offload_manager = None

        self.vae_batch_size = self.cfg.infer.get("vae_batch_size", 8)
        self.height = self.cfg.infer.get("height", 512)
        self.width = self.cfg.infer.get("width", 512)
        self.downsampling_factor = self.cfg.infer.get("down_factor", 8)
        self.in_channels = self.model.model.model.diffusion_model.in_channels
        self.unconditional_guidance_scale = self.cfg.infer.get("unconditional_guidance_scale", 7.5)
        self.sampler_type = self.cfg.infer.get("sampler_type", "DDIM")
        self.inference_steps = self.cfg.infer.get("inference_steps", 50)
        self.eta = self.cfg.infer.get("eta", 0)

        # TODO @geshen: In train_utils.py, the prepare_for_training_step funtions requires the following which are not available for Stable Diffusion
        self.model.model.initialize_ub = False
        self.model.model.rampup_batch_size = False
        self.model.model.with_distributed_adam = False

    def get_parameters_with_grad(self):
        return self.model.get_parameters_with_grad()

    def finish_inference(self):
        return

    # TODO @geshen @sahilj: allreduce_gradients is not defined for Stable Diffusion for grad_reductions
    def finish_training_step(self):
        self.grad_reductions()
        # grad_reductions(self)

    def infer(self):
        return

    def prepare_for_inference(self):
        return

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self.model.model, zero_grad=False)

    def generate_rollout_batch(self):
        return

    def prepare_for_generation(self):
        # self.model.eval()
        self.model._reset_activation_checkpointing_args()
        self.model._reset_sequence_parallelism_args()
        return

    def finished_generation(self):
        self.model._restore_activation_checkpointing_args()
        self.model._restore_sequence_parallelism_args()
        return

    @torch.no_grad()
    def generate_log_images(self, latents, batch, model):

        # setup default values for inference configs
        # get autocast_dtype
        if self.cfg.precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif int(self.cfg.precision) == 32:
            autocast_dtype = torch.float
        elif int(self.cfg.precision) == 16:
            autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
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
                image = self.model.model.differentiable_decode_first_stage(
                    trajectories_predx0[idx_denoised_imgs[i : i + self.vae_batch_size]]
                )

                vae_decoder_output.append(image)

            vae_decoder_output = torch.cat(vae_decoder_output, dim=0)
            vae_decoder_output = torch.clip((vae_decoder_output + 1) / 2, 0, 1) * 255.0

            log_reward = [
                self.reward_model.get_reward(
                    vae_decoder_output[i].unsqueeze(0).detach().permute(0, 2, 3, 1), batch
                ).item()
                for i in range(batch_size)
            ]
            log_img = [
                np.transpose(vae_decoder_output[i].float().detach().cpu().numpy(), (1, 2, 0))
                for i in range(batch_size)
            ]
            return log_img, log_reward

    @torch.no_grad()
    def log_visualization(self, prompts):

        batch_size = len(prompts)
        latents = torch.randn(
            [
                batch_size,
                self.in_channels,
                self.height // self.downsampling_factor,
                self.width // self.downsampling_factor,
            ],
            generator=None,
        ).to(torch.cuda.current_device())

        image_draft_p, reward_draft_p = self.generate_log_images(latents, prompts, self.model.model)
        image_init, reward_init = self.generate_log_images(latents, prompts, self.init_model)

        images = []
        captions = []
        for i in range(len(image_draft_p)):
            images.append(image_draft_p[i])
            images.append(image_init[i])
            captions.append("draft_p: " + prompts[i] + ", Reward = " + str(reward_draft_p[i]))
            captions.append("SD: " + prompts[i] + ", Reward = " + str(reward_init[i]))

        self.logger.loggers[1].log_image(
            key="Inference Images",
            images=[wandb.Image(Image.fromarray(img.round().astype("uint8"))) for img in images],
            caption=captions,
        )

    def generate(
        self, batch, x_T,
    ):

        # get autocast_dtype
        if self.cfg.precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif int(self.cfg.precision) == 32:
            autocast_dtype = torch.float
        elif int(self.cfg.precision) == 16:
            autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        with torch.cuda.amp.autocast(
            enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
        ):

            batch_size = len(batch)
            prev_img_draft_p = x_T

            device_draft_p = self.model.model.betas.device

            sampler_draft_p = sampling_utils.initialize_sampler(self.model.model, self.sampler_type.upper())
            sampler_init = sampling_utils.initialize_sampler(self.init_model, self.sampler_type.upper())

            cond, u_cond = sampling_utils.encode_prompt(
                self.model.model.cond_stage_model, batch, self.unconditional_guidance_scale
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

            for i, step in enumerate(iterator):

                if i < total_steps - truncation_steps:
                    with torch.no_grad():
                        img_draft_p, pred_x0_draft_p, eps_t_draft_p = sampler_draft_p.single_ddim_denoise_step(
                            prev_img_draft_p,
                            total_steps,
                            i,
                            batch_size,
                            device_draft_p,
                            step,
                            cond,
                            unconditional_guidance_scale=self.unconditional_guidance_scale,
                            unconditional_conditioning=u_cond,
                        )

                        prev_img_draft_p = img_draft_p
                else:

                    img_draft_p, pred_x0_draft_p, eps_t_draft_p = sampler_draft_p.single_ddim_denoise_step(
                        prev_img_draft_p,
                        total_steps,
                        i,
                        batch_size,
                        device_draft_p,
                        step,
                        cond,
                        unconditional_guidance_scale=self.unconditional_guidance_scale,
                        unconditional_conditioning=u_cond,
                    )
                    list_eps_draft_p.append(eps_t_draft_p)
                    with torch.no_grad():
                        _, _, eps_t_init = sampler_init.single_ddim_denoise_step(
                            prev_img_draft_p,
                            total_steps,
                            i,
                            batch_size,
                            device_draft_p,
                            step,
                            cond,
                            unconditional_guidance_scale=self.unconditional_guidance_scale,
                            unconditional_conditioning=u_cond,
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
                image = self.model.model.differentiable_decode_first_stage(
                    trajectories_predx0[i : i + self.vae_batch_size]
                )
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
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            batch_size = len(dataloader_iter)

            latents = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    self.height // self.downsampling_factor,
                    self.width // self.downsampling_factor,
                ],
                generator=None,
            ).to(torch.cuda.current_device())

            output_tensor_draft_p, epsilons_draft_p, epsilons_init = self.generate(dataloader_iter, latents)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor_draft_p = output_tensor_draft_p.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor_draft_p):
                # Loss per micro batch (ub).
                if self.cfg.kl_coeff == 0.0:
                    kl_penalty = torch.tensor([0.0]).to(torch.cuda.current_device())
                else:
                    kl_penalty = self.calculate_gaussian_kl_penalty_shared_var(epsilons_draft_p, epsilons_init).mean()
                rewards = self.reward_model.get_reward(output_tensor_draft_p.permute(0, 2, 3, 1), dataloader_iter)
                loss = -rewards.mean() + kl_penalty * self.cfg.kl_coeff

                reduced_loss = average_losses_across_data_parallel_group([loss])

                return (
                    loss,
                    {"loss": reduced_loss, "kl_penalty": kl_penalty},
                )

            return output_tensor_draft_p, loss_func

        return fwd_output_and_loss_func

    def calculate_gaussian_kl_penalty_shared_var(self, curr_eps, init_eps):

        diff = curr_eps - init_eps
        kl = torch.sum(diff ** 2, dim=(1, 2, 3, 4), keepdim=True).flatten()
        dimensionality = torch.numel(curr_eps[0])
        kl /= dimensionality

        return kl

    def get_loss_and_metrics(self, batch, forward_only=False):

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = []

        # TODO @ataghibakhsh @geshen @sahilj: For grad accumulation, we currently have to do the for loop and fwd_bwd_function doesn't take care of that
        # TODO @ataghibakhsh: input num_microbatch manually
        for i in range(0, len(batch), self.cfg.micro_batch_size):
            data_item = batch[i : i + self.cfg.micro_batch_size]

            losses_reduced_per_micro_batch.append(
                fwd_bwd_function(
                    forward_step_func=self.get_forward_output_and_loss_func(),
                    data_iterator=[data_item],  # 4
                    model=self.model.model,
                    num_microbatches=self.cfg.micro_batch_size,  # 4
                    forward_only=forward_only,
                    seq_length=None,
                    micro_batch_size=self.cfg.micro_batch_size,  # 1
                )[0]
            )

        if torch.distributed.get_rank() == 0 and len(self.logger.loggers) > 1:
            self.log_visualization(batch[0:1])

        if self.model.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.model.allreduce_gradients()

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

    def grad_reductions(self):
        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.cfg.get("sequence_parallel", False):
            self.model.allreduce_sequence_parallel_gradients()

        if self.model.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self.optimizer._finish_bucket_grad_sync()
        elif self.model.megatron_amp_O2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if parallel_state.get_pipeline_model_parallel_world_size() > 1 or self.cfg.get("sequence_parallel", False):
                # main grads are stored in the MainParamsOptimizer wrapper
                self.optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.model.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.model.allreduce_first_last_embeddings()
