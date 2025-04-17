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

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Callable
import hashlib
import json

import pandas as pd
import torch
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.experimental.grpo.models.nlp.gpt.megatron_gpt_grpo_actor import MegatronGPTActorModel
from ..utils import parallel_state
from nemo_aligner.utils.distributed import (
    ScopedTimer,
    all_reduce_dict,
    masked_global_mean_var,
)
from nemo_aligner.utils.utils import log_memory, print_memory_info
from ..utils.parallel_state import is_inference_reshard, inference_reshard_region
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.experimental.grpo.utils.rl_utils import (
    calculate_kl_penalty_joschu2020,
    calculate_baseline_and_std_per_prompt,
)
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import clear_memory, cpu_dict, masked_mean
from nemo_aligner.experimental.grpo.experience.interfaces import RolloutGeneratorInterface
from nemo_aligner.experimental.grpo.experience.rollout_batch import GPTRolloutBatch
from nemo_aligner.utils.utils import log_memory

def compute_num_rollout_microbatches_per_dp_group(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )

class GRPOTrainer:
    """Trainer to coordinate GRPO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: MegatronGPTActorModel,
        optimizer,
        scheduler,
        train_dataloader_builder,
        val_dataloader_builder,
        collate_fn,
        rollout_generator: RolloutGeneratorInterface,
        batch_iterator_cls,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader_builder = train_dataloader_builder
        self.val_dataloader_builder = val_dataloader_builder
        self.collate_fn = collate_fn
        self.rollout_generator = rollout_generator
        self.batch_iterator_cls = batch_iterator_cls
        self.logger = logger
        self.ckpt_callback = ckpt_callback
        self.overall_iter = 0

        print_memory_info("In the GRPOTrainer constructor")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.reshard_weights_for_generation = cfg.inference_backend.enable and cfg.inference_backend.reshard

        print(f"model_gbs in initialization: {cfg.model_gbs}")
        

        # Tracked by state dict and checkpointed
        self.consumed_samples = 0       # consumed prompts (for the dataloader to keep track)
        self.step = 0                   # GRPO step (num sampling rounds)
        self.grpo_optimization_step = 0 # number of times we optimized the actor (num calls to optimizer step)

        # compute `max_steps`
        train_dataloader = self.train_dataloader_builder(consumed_samples=0)
        if (not isinstance(train_dataloader.batch_sampler, MegatronPretrainingRandomSampler)) and (
            self.cfg.max_epochs is not None and self.cfg.max_epochs > 1
        ):
            # if you use MegatronPretrainingBatchSampler as the batch_sampler passed to your train dataloader (in builders.py)
            # then each epoch will repeat all your samples in the same order as the previous epoch, there is no shuffling
            # to fix this, you should use MegatronPretrainingRandomSampler instead, which alleviates this issue and allows
            # random shuffling for each epoch.
            raise ValueError(
                "max_epochs > 1 is not supported unless using `MegatronPretrainingRandomSampler` as the batch_sampler for your train dataloader"
            )

        self.num_steps_per_epoch = compute_num_steps_per_epoch(train_dataloader.batch_sampler)
        self.set_max_steps()

        # size to pad our rollout batch to. If configured to None, we'll just use max-sequence padding
        self.rollout_batch_seq_length = self.cfg.rollout_batch_seq_length

        # TODO @sahilj move this to rollout generator or environments
        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        self.timer = ScopedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

    def create_grpo_training_data_from_inferences(self, rollout_batch: GPTRolloutBatch):
        """
        Generate grpo specific loss terms for training and metrics. 
        Runs on every rank.
        
        rollout_batch: GPTRolloutBatch.  A **global** rollout batch
        
        Returns:
        **sharded** output training data dictionary and global metrics.
        The output training data dictionary should be sharded among DP groups.
        """
        grpo_train_data = GPTRolloutBatch() # using this class for easy chunking/sharding
        grpo_rollout_metrics = {}

        rewards = rollout_batch["rewards"]
        assert not (self.cfg.length_penalty_enabled and self.cfg.get("dapo", {}).get("length_penalty", False)), "length_penalty_enabled (alex's) and dapo.length_penalty cannot both be True"
        # alex's linear length penalty
        if self.cfg.length_penalty_enabled:
            rewards = -1 * (1 - rewards) + -1 *rewards * ((rollout_batch["response_lengths"] - rollout_batch["prompt_lengths"]) / self.cfg.max_sequence_length) + 1
        
        # dapo's linear length penalty
        elif self.cfg.get("dapo", {}).get("length_penalty", False):
            # Get acceptable length from config or default to 75% of max sequence length
            acceptable_length = self.cfg.get("dapo", {}).get("acceptable_length", int(0.75 * self.cfg.dpo.length_penalty_params.max_length))
            
            # Calculate response lengths (excluding prompt)
            generation_lengths = rollout_batch["response_lengths"] - rollout_batch["prompt_lengths"]
            
            # Initialize penalty tensor with zeros (same shape as rewards)
            length_penalty = torch.zeros_like(rewards)
            
            # Find responses longer than acceptable length
            long_responses_mask = generation_lengths > acceptable_length
            
            # For long responses, apply linear penalty from 0 to 0.5 as length increases, if the reward scale changes (now 0-1), this needs to be changed
            if long_responses_mask.any():
                # Calculate how far into the penalty zone we are (0 to 1)
                penalty_zone_length = self.cfg.dpo.length_penalty_params.max_length - acceptable_length
                position_in_penalty_zone = (generation_lengths[long_responses_mask] - acceptable_length) / penalty_zone_length
                # Clamp to [0, 1] range in case of any numerical issues
                position_in_penalty_zone = torch.clamp(position_in_penalty_zone, 0, 1)
                # Scale to [0, 0.5] range for penalty
                length_penalty[long_responses_mask] = 0.5 * position_in_penalty_zone
            
            # Apply penalty to rewards
            rewards = rewards - length_penalty

        baseline, reward_std = calculate_baseline_and_std_per_prompt(
                prompts=rollout_batch["text"],
                rewards=rewards,
                valid_mask=rollout_batch["is_end"],
                leave_one_out_baseline=self.cfg.use_leave_one_out_baseline
            )
        advantages = rewards.unsqueeze(-1) - baseline.unsqueeze(-1)
        pre_norm_advantages = advantages.clone()

        if self.cfg.normalize_rewards:
            # don't sharpen the ones with no variation
            zero_std_mask = reward_std > 0
            advantages[zero_std_mask] = advantages[zero_std_mask] / reward_std.unsqueeze(-1)[zero_std_mask]

        mask = create_mask(
            values=rollout_batch["logprobs"],
            prompt_lengths=rollout_batch["prompt_lengths"],
            response_lengths=rollout_batch["response_lengths"],
        )

        # collect everything we need to train GRPO
        grpo_train_data["response_tokens"] = rollout_batch["response_tokens"]
        grpo_train_data["logprobs"] = rollout_batch["logprobs"]
        grpo_train_data["inference_logprobs"] = rollout_batch["response_trt_lps"]
        grpo_train_data["valid_mask"] = rollout_batch["is_end"]
        grpo_train_data["mask"] = mask
        grpo_train_data["init_logprobs"] = rollout_batch["init_logprobs"]
        grpo_train_data["advantages"] = advantages

        # compute metrics
        if self.cfg.get("importance_sample_correct", False):
            # logprobs is left rolled by 1 and padded to the rollout max_seqlen. So apply the same to "response_trt_lps"
            trt_lps = rollout_batch["response_trt_lps"][:, 1:]
            rollout_batch_size = rollout_batch['logprobs'].size()
            trt_lps_size = trt_lps.size()
            if trt_lps_size[1] == rollout_batch_size[1] - 1:
                trt_lps = torch.nn.functional.pad(trt_lps, (0, 1), value=0)
            print(f"rollout_batch shape = {rollout_batch['logprobs'].shape}, trt_lps shape = {trt_lps.shape}", flush=True)
            assert trt_lps.shape == rollout_batch["logprobs"].shape

            importance_correction = torch.exp(rollout_batch["logprobs"] - trt_lps)
            importance_correction = torch.nan_to_num(importance_correction, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"importance_correction max = {torch.max(importance_correction).item()}, min = {torch.min(importance_correction).item()}")
        else:
            importance_correction = torch.ones_like(rollout_batch["logprobs"])
        grpo_train_data["importance_correction"] = importance_correction

        init_policy_kl = calculate_kl_penalty_joschu2020(
                log_probs_policy=rollout_batch["logprobs"],
                log_probs_reference=rollout_batch["init_logprobs"],
            ) * importance_correction
        grpo_rollout_metrics["init_policy_kl"] = (
            masked_mean(init_policy_kl, mask).item()
        )
        grpo_rollout_metrics["nonzero_advantages"] = (advantages != 0).float().mean()
        grpo_rollout_metrics["logprobs_mean"] = masked_mean(rollout_batch["logprobs"], mask).item()
        grpo_rollout_metrics["init_logprobs_mean"] = masked_mean(rollout_batch["init_logprobs"], mask).item()
        grpo_rollout_metrics["importance_correction_mean"] = masked_mean(importance_correction, mask).item()
        
        local_grpo_train_data = grpo_train_data.chunk(
            rank=parallel_state.get_training_data_parallel_rank(),
            split_size=parallel_state.get_training_data_parallel_world_size()
        )

        return local_grpo_train_data.get_dict(), cpu_dict(grpo_rollout_metrics)

    def _run_inference(self, dataloader_builder: Callable[[int], Any], consumed_samples: int, is_validation: bool=False):
        """
        Helper function that runs rollout generation for both training and validation. Runs on all ranks.

        dataloader_builder: Callable    constructor for a dataloader that accepts a 'consumed_samples' argument 
                                        to start the dataloader in the right place.
        consumed_samples:   int         number of prompts consumed in training so far. Used to initialize the dataloader
        is_validation:      bool        is sampling for validation (vs training when False)
        
        Returns:
        rollout_data:    GPTRolloutBatch   A **global** rollout batch
        rollout_metrics: dict              **global** rollout metrics from each environment + some general ones
        rollout_timing:  dict              **dp local** timing metrics (will be reduced later)
        """
        # initialize prompt dataloader
        generation_reshard_context = inference_reshard_region if self.reshard_weights_for_generation else nullcontext
        with generation_reshard_context():
            # dataloader must be built within the generation context because it uses DP rank and size
            dataloader = dataloader_builder(consumed_samples=consumed_samples)
            sampler_iter = iter(dataloader.batch_sampler)

            # must compute the number of microbatches in the generation context for correct DP groups
            # num_microbatches = compute_num_rollout_microbatches_per_dp_group(dataloader)
            if is_validation:
                num_prompts_per_grpo_step = self.cfg.val_num_prompts_per_grpo_step
                samples_per_prompt = self.cfg.val_samples_per_prompt
            else:
                num_prompts_per_grpo_step = self.cfg.num_prompts_per_grpo_step
                samples_per_prompt = self.cfg.samples_per_prompt

            with self.timer("batch_iterator_init"):
                batch_iterator = self.batch_iterator_cls(
                    sampler_iter=sampler_iter, 
                    micro_batch_size=self.cfg.generation_rollout_mbs, 
                    num_prompts_per_grpo_step=num_prompts_per_grpo_step,
                    samples_per_prompt=samples_per_prompt,
                    dataset=dataloader.dataset, 
                    collate_fn=self.collate_fn
                )

        # the rollout_generator handles experience generation and returns a global batch
        rollout_data, rollout_metrics, rollout_timing = self.rollout_generator.generate_rollouts(
            batch_iterator, self.model, is_validation=is_validation, greedy=is_validation and self.cfg.greedy_on_validation
        )
        return rollout_data, rollout_metrics, rollout_timing

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        _, rollout_metrics, rollout_timing = self._run_inference(self.val_dataloader_builder, consumed_samples=0, is_validation=True)

        self.model.finish_inference()
        return rollout_metrics

    @torch.no_grad()
    def generate_rollouts(self):
        """
        Generates rollouts for training. Also collects and returns metrics and timing. Runs on every rank.
        
        Returns:
        train_data: dict    A dictionary of all data required to train the model
        metrics: dict       Collected metrics from environments and GRPO stats
        timing: dict
        """
        with self.timer("prepare_for_inference"):
            # Timing includes build if first step and refit if step > 1
            self.model.prepare_for_inference()

        rollout_data, rollout_metrics, rollout_timing = self._run_inference(self.train_dataloader_builder, consumed_samples=self.consumed_samples, is_validation=False)       
        
        train_data, grpo_metrics = self.create_grpo_training_data_from_inferences(rollout_data)

        self.consumed_samples += self.cfg.num_prompts_per_grpo_step

        with self.timer("finish_inference"):
            # Timing includes engine unloading if enabled
            self.model.finish_inference()

        return (
            train_data,
            rollout_metrics | grpo_metrics | {"consumed_samples": self.consumed_samples},
            self.timer.consume_durations() | rollout_timing,
        )

    def run_training(self, dataloader_iter):
        iteration = 0
        self.model.prepare_for_training()

        for batch in dataloader_iter:
            print(f"Iteration {iteration}, overall_iter = {self.overall_iter}")
            iteration += 1
            self.optimizer.zero_grad()

            self.model.prepare_for_training_step()
            loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=False)
            self.model.finish_training_step()

            grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
            grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            if torch.isnan(torch.tensor(grad_norm)):
                logging.warning("Skipping training step due to NaN gradient norm.")
                self.optimizer.zero_grad()  # Zero out gradients
                metrics["grad_norm"] = float('nan')
                metrics.update({"loss": loss_mean, "optim_step": self.grpo_optimization_step})
                self.logger.log_metrics(metrics, step=self.step, prefix="train_optim/")
                continue

            lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.step()
            self.scheduler.step()

            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            if lr is not None:
                # Some optimizers like adafactor do not require a LR in their initializer
                metrics["lr"] = lr

            metrics.update({"loss": loss_mean, "optim_step": self.grpo_optimization_step})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.grpo_optimization_step += 1
        print("grpo optimization step", self.grpo_optimization_step)
        self.overall_iter += 1

        self.model.finish_training()

        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()
        return loss_mean, metrics

    def fit(self):
        epoch_iter = range(self.epoch, self.cfg.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        for _ in epoch_iter:
            num_grpo_steps_in_epoch = min(
                self.max_steps - self.step, self.num_steps_per_epoch - self.step % self.num_steps_per_epoch
            )
            loop_iter = range(num_grpo_steps_in_epoch)
            if not loop_iter:
                return  # training ended

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="GRPO Global Step")

            dp_size = parallel_state.get_training_data_parallel_world_size()
            num_samples_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)
           
            print(f"dp = {dp_size}, num_samples_to_load_on_each_dp = {num_samples_to_load_on_each_dp}, model_gbs = {self.cfg.model_gbs}", flush=True)

            self.run_timer.start_time()
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                clear_memory()
                with self.timer("rollout_time"):
                    grpo_rollout_data, metrics, rollout_timer_metrics = self.generate_rollouts()
                # Consume rollout_time
                timing_metrics.update(self.timer.consume_durations())

                clear_memory()

                rollout_timer_metrics = all_reduce_dict(rollout_timer_metrics, op=torch.distributed.ReduceOp.MAX)
                timing_metrics.update(rollout_timer_metrics)

                # TODO @sahilj move table logging to rollout generator
                # logging
                # table_metrics = metrics.pop("table")
                # self.train_df.loc[len(self.train_df)] = [
                    # self.step,
                    # table_metrics["prompt"],
                    # table_metrics["response"],
                    # table_metrics["reward"],
                # ]
                metrics["epoch"] = self.epoch + 1
                self.logger.log_metrics(metrics, step=self.step, prefix="train_rollouts/")
                # self.logger.log_table(
                    # key="table/train_rollouts", dataframe=self.train_df, step=self.step,
                # )

                if self.step == 0 and self.cfg.run_validation_step_0:
                    # TODO @sahilj flag doing a validation run at step 0
                    with self.timer("validation_time"):
                        val_metrics = self.run_validation()
                    timing_metrics.update(self.timer.consume_durations())
                    # TODO @sahilj replace table here with something in rollout generator
                    # val_table_metrics = val_metrics.pop("table")

                    # self.val_df.loc[len(self.val_df)] = [
                        # self.step,
                        # val_table_metrics["prompt"],
                        # val_table_metrics["response"],
                        # val_table_metrics["reward"],
                    # ]
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    # self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                    clear_memory()

                rollout_size = grpo_rollout_data["response_tokens"].size(0)
                rollout_dataloader_iter = get_iterator_k_split(
                    grpo_rollout_data, divide(rollout_size, num_samples_to_load_on_each_dp)
                )
                print(f"grpo_rollout_data size = {grpo_rollout_data['response_tokens'].size()}, rollout_size = {rollout_size}, num_samples_to_load_on_each_dp = {num_samples_to_load_on_each_dp}")
                # start training
                clear_memory()
                with self.timer("train_time"):
                    log_memory("grpo.py self.run_training")
                    self.run_training(rollout_dataloader_iter)

                self.logger.log_metrics(
                    timing_metrics | self.timer.consume_durations(), step=self.step, prefix="timers/"
                )

                self.step += 1

                run_time_exceeded = self.run_timer.is_finished()
                run_val, save_model, is_train_end = check_progress(
                    self.step,
                    self.max_steps,
                    self.cfg.val_check_interval,
                    self.cfg.save_interval,
                    1.0,  # TODO:(geshen): allow for limit val batches
                    run_time_exceeded=run_time_exceeded,
                )

                print(f"run_val = {run_val}, save_model = {save_model}, is_train_end = {is_train_end}, step = {self.step}, cfg.val_check_interval = {self.cfg.val_check_interval}, cfg.save_interval = {self.cfg.save_interval}")
                
                if run_val:
                    with self.timer("validation_time"):
                        val_metrics = self.run_validation()
                    # Note: validation_time is logged one step behind (val step 5 means we've completed step 4)
                    timing_metrics.update(self.timer.consume_durations())

                    # val_table_metrics = val_metrics.pop("table")

                    # self.val_df.loc[len(self.val_df)] = [
                        # self.step,
                        # val_table_metrics["prompt"],
                        # val_table_metrics["response"],
                        # val_table_metrics["reward"],
                    # ]
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    # self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                step_metrics.update(timing_metrics)
                step_metrics.update({f"train_{k}": v for k, v in metrics.items()})
                global_pbar.set_postfix(step_metrics)

                if save_model:
                    step_metrics = {k: torch.as_tensor(v) for k, v in filter(lambda i: not isinstance(i[1], dict), step_metrics.items())}
                    self.save(step_metrics, is_train_end=is_train_end, run_val=run_val)

                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

        self.logger.finalize()

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "grpo_optimization_step": self.grpo_optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.grpo_optimization_step = state_dict["grpo_optimization_step"]

        loaded_values = [self.step, self.consumed_samples, self.grpo_optimization_step]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def save(self, extra_candidates=None, is_train_end=False, run_val=False):
        self.model.prepare_for_training()
        # load back in the adam states if needed
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None or run_val is False:
            # Add an arbitrary rewards to keep the checkpoint even when validation is not triggered.
            extra_candidates = {
                'val_rewards': torch.tensor(0.1),
            }

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        log_memory("Before checkpoint save")
        print_memory_info("Before checkpoint save")
        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)
        log_memory("After checkpoint save")
        print_memory_info("After checkpoint save")

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.step // self.num_steps_per_epoch
