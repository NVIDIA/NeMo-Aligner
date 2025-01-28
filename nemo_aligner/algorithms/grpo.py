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
from typing import Dict, List, Optional

import pandas as pd
import torch
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.megatron_gpt_ppo_actor import MegatronGPTActorModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    ScopedTimer,
    all_reduce_dict,
    masked_global_mean_var,
)
from nemo_aligner.utils.parallel_state import is_trt_llm_reshard, trt_llm_reshard_region
from nemo_aligner.utils.ppo_utils import (
    calculate_kl_penalty,
    create_mask,
)
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import clear_memory, cpu_dict, masked_mean
from nemo_aligner.experimental.experience.interfaces import RolloutGeneratorInterface

def compute_num_rollout_microbatches_per_dp_group(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )

# TODO @sahilj add unit test
def calculate_baseline_per_prompt(prompts, rewards, valid_mask, leave_one_out_baseline=True):
    """
    Function to compute a baseline for each (prompt, response) pair in the batch. 
    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask' 
    are not included in the baseline calculation. This function assumes the user has
    _already_ normalized the rewards (if they desire to do so).

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute baseline by leaving out the sample that the baseline is for (from RLOO)
    
    returns
    tensor (b,) of baselines on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = "cpu"

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(reward_device)

        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            prompt_baseline = torch.matmul(baseline_mask_matrix, 
                                           rewards[prompt_idx] * valid_mask[prompt_idx]) / (valid_mask[prompt_idx].sum() - 1)
            baseline[prompt_idx] = prompt_baseline

    return baseline


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

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.trtllm_reshard = "trt_llm" in cfg and cfg.trt_llm.enable and cfg.trt_llm.reshard

        self.consumed_samples = 0
        # the step here is GRPO step (num sampling rounds)
        self.step = 0
        # keep track of how many times we optimized the actor (num calls to optimizer step)
        self.grpo_optimization_step = 0

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

        self.num_grpo_steps_per_epoch = compute_num_steps_per_epoch(train_dataloader.batch_sampler)
        self.set_max_steps()

        # TODO @sahilj remove all references to this and force True
        self.compute_init_policy_kl = True #self.cfg.initial_policy_kl_penalty > 0
        # size to pad our rollout batch to
        self.rollout_batch_seq_length = self.cfg.rollout_batch_seq_length

        # TODO @sahilj move this to rollout generator or environments
        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        self.timer = ScopedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

    # TODO @sahilj convert to new rollout_generator formulation
    def generate_grpo_data(self, rollout_batch):
        """generate ppo specific data for training
        """
        grpo_rollout_data = {}
        grpo_rollout_metrics = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]
        logprobs = rollout_batch["logprobs"]
        is_end = rollout_batch["is_end"]

        num_responses = self.cfg.num_responses_per_prompt

        # TODO @sahilj map to gerald's version, but with prompt checking from alex
        baseline, reward_std = calculate_baseline_per_prompt(
                prompts=rollout_batch["prompts"], reward=rollout_batch["rewards"]
            )
        advantages = rewards - baseline
        pre_norm_advantages = advantages.clone()

        if self.cfg.normalize_rewards:
            # don't sharpen the ones with no variation
            # advantages = advantages / grouped_reward_std
            mask = reward_std > 0
            advantages[mask] = advantages[mask] / reward_std[mask]
        num_advantages_not_zero = (advantages != 0).any(-1).sum()

        prenorm_avg_adv_max = pre_norm_advantages.max(-1).values.sum()
        prenorm_avg_adv_min = pre_norm_advantages.min(-1).values.sum()

        # TODO: consider normalizing the advantages
        advantages = advantages.flatten()
        pre_norm_advantages = pre_norm_advantages.flatten().cuda()

        greater_than_0_adv_mask = advantages > 0
        num_advantages_greater_than_0 = greater_than_0_adv_mask.sum()

        normalized_response_length = (normalized_response_length * greater_than_0_adv_mask).sum()
        response_length_z = (response_length_z * greater_than_0_adv_mask).sum()

        # advantage estimation
        mask = create_mask(values=logprobs, prompt_lengths=prompt_lengths, response_lengths=response_lengths)

        # collect everything we need to train GRPO
        grpo_rollout_data["mask"] = mask
        grpo_rollout_data["response_tokens"] = response_tokens
        grpo_rollout_data["prev_logprobs"] = logprobs
        grpo_rollout_data["init_logprobs"] = rollout_batch["init_logprobs"]
        grpo_rollout_data["response_tokens"] = response_tokens
        grpo_rollout_data["is_end"] = is_end

        # compute metrics
        init_policy_kl = calculate_kl_penalty(
                log_probs_policy=rollout_batch["logprobs"],
                log_probs_reference=rollout_batch["init_logprobs"],
            )

        ppo_rollout_metrics["init_policy_kl"] = (
            masked_mean(init_policy_kl, mask, dim=-1).sum().item() if self.compute_init_policy_kl else 0
        )
        ppo_rollout_metrics["num_samples"] = prompt_lengths.size(0)

        # now the metrics are global
        ppo_rollout_metrics = all_reduce_dict(
            ppo_rollout_metrics, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM
        )
        num_samples = grpo_rollout_metrics.pop("num_samples")
        grpo_rollout_metrics = {k: v / num_samples for k, v in grpo_rollout_metrics.items()}

        mask = grpo_rollout_data["mask"]
        for key in ["advantages", "returns", "values"]:
            tensor = grpo_rollout_data[key]

            global_mean, global_var = masked_global_mean_var(
                tensor, mask, group=parallel_state.get_data_parallel_group(),
            )
            grpo_rollout_metrics[f"{key}_mean"] = global_mean.item()
            grpo_rollout_metrics[f"{key}_std"] = global_var.sqrt().item()


        return grpo_rollout_data, cpu_dict(grpo_rollout_metrics)

    # TODO @sahilj shift to rollout generator's responsibility
    def compute_rollout_metrics(self, rollout_batch):
        table = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]
        is_end = rollout_batch["is_end"]

        # take the first sample for logging
        reward = rewards[0]
        prompt_length = prompt_lengths[0]
        response_length = response_lengths[0]
        response_token = response_tokens[0]

        table["reward"] = reward.item()
        table["prompt"] = self.model.tokenizer.ids_to_text(response_token[:prompt_length].tolist())
        table["response"] = self.model.tokenizer.ids_to_text(response_token[prompt_length:response_length].tolist())

        metrics = {
            "table": table,
            "rollout_size": prompt_lengths.size(0),
            "response_lengths": response_lengths.float().mean().item(),
            "prompt_lengths": prompt_lengths.float().mean().item(),
            "generation_length": (response_lengths - prompt_lengths).float().mean().item(),
            "rewards": rewards.mean().item(),
            "fraction_of_samples_properly_ended": is_end.float().mean().item(),
        }

        return metrics

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        _, rollout_metrics = self._run_inference(self.val_dataloader_builder, consumed_samples=0, is_validation=True)

        self.model.finish_inference()
        return rollout_metrics

    @torch.no_grad()
    def generate_rollouts(self):
        with self.timer("prepare_for_inference"):
            # Timing includes build if first step and refit if step > 1
            self.model.prepare_for_inference()

        # initialize prompt dataloader
        inference_reshard_context = trt_llm_reshard_region if self.trtllm_reshard else nullcontext
        with inference_reshard_context():
            # dataloader must be built within the inference context because it uses DP rank and size
            dataloader = self.train_dataloader_builder(consumed_samples=self.consumed_samples)
            sampler_iter = iter(dataloader.batch_sampler)

            # must compute the number of microbatches in the inference context for correct DP groups
            num_microbatches = compute_num_rollout_microbatches_per_dp_group(dataloader)

            with self.timer("batch_iterator_init"):
                batch_iterator = self.batch_iterator_cls(
                    sampler_iter, num_microbatches, dataloader.dataset, self.collate_fn
                )

        # let the rollout_generator handle experience generation
        rollout_data, rollout_metrics = self.rollout_generator.generate(
            batch_iterator, self.model, self.timer, is_validation=False
        )
        
        rollout_data["baseline"] = calculate_baseline_per_prompt(
            rollout_data["prompts"], rollout_data["rewards"], 
            rollout_data["is_valid"], leave_one_out_baseline=self.cfg.use_leave_one_out_baseline
        )

        self.consumed_samples += rollout_metrics["rollout_size"]

        with self.timer("finish_inference"):
            # Timing includes engine unloading if enabled
            self.model.finish_inference()

        return (
            rollout_data,
            rollout_metrics | {"consumed_samples": self.consumed_samples},
            self.timer.consume_durations(),
        )

    def run_training(self, dataloader_iter):
        self.model.prepare_for_training()

        for batch in dataloader_iter:
            self.optimizer.zero_grad()

            self.model.prepare_for_training_step()
            loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=False)
            self.model.finish_training_step()

            grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
            grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.step()
            self.scheduler.step()

            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            if lr is not None:
                # Some optimizers like adafactor do not require a LR in their initializer
                metrics["lr"] = lr

            metrics.update({"loss": loss_mean, "optim_step": self.ppo_optimization_step})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.grpo_optimization_step += 1

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

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            dp_size = parallel_state.get_training_data_parallel_world_size()

            num_samples_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            self.run_timer.start_time()
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                clear_memory()
                with self.timer("rollout_time"):
                    ppo_rollout_data, metrics, rollout_timer_metrics = self.generate_rollouts()
                # Consume rollout_time
                timing_metrics.update(self.timer.consume_durations())

                clear_memory()

                rollout_timer_metrics = all_reduce_dict(rollout_timer_metrics, op=torch.distributed.ReduceOp.MAX)
                timing_metrics.update(rollout_timer_metrics)

                # TODO @sahilj move table logging to rollout generator
                # logging
                table_metrics = metrics.pop("table")
                self.train_df.loc[len(self.train_df)] = [
                    self.step,
                    table_metrics["prompt"],
                    table_metrics["response"],
                    table_metrics["reward"],
                ]
                metrics["epoch"] = self.epoch + 1
                self.logger.log_metrics(
                    metrics, step=self.step, prefix="train_rollouts/",
                )
                self.logger.log_table(
                    key="table/train_rollouts", dataframe=self.train_df, step=self.step,
                )

                if self.step == 0:
                    self.consumed_samples = metrics["rollout_size"]

                    # TODO @sahilj flag doing a validation run at step 0

                    with self.timer("validation_time"):
                        val_metrics = self.run_validation()
                    timing_metrics.update(self.timer.consume_durations())
                    # TODO @sahilj replace table here with something in rollout generator
                    val_table_metrics = val_metrics.pop("table")

                    self.val_df.loc[len(self.val_df)] = [
                        self.step,
                        val_table_metrics["prompt"],
                        val_table_metrics["response"],
                        val_table_metrics["reward"],
                    ]
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                    clear_memory()

                rollout_size = ppo_rollout_data["response_tokens"].size(0)
                rollout_dataloader_iter = get_iterator_k_split(
                    ppo_rollout_data, divide(rollout_size, num_samples_to_load_on_each_dp)
                )
                # start training
                clear_memory()
                with self.timer("train_time"):
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

                if run_val:
                    with self.timer("validation_time"):
                        val_metrics = self.run_validation()
                    # Note: validation_time is logged one step behind (val step 5 means we've completed step 4)
                    timing_metrics.update(self.timer.consume_durations())

                    val_table_metrics = val_metrics.pop("table")

                    self.val_df.loc[len(self.val_df)] = [
                        self.step,
                        val_table_metrics["prompt"],
                        val_table_metrics["response"],
                        val_table_metrics["reward"],
                    ]
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                step_metrics.update(timing_metrics)
                step_metrics.update({f"train_{k}": v for k, v in metrics.items()})
                global_pbar.set_postfix(step_metrics)

                if save_model:
                    step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                    self.save(step_metrics, is_train_end=is_train_end)

                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

        self.logger.finalize()

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "ppo_optimization_step": self.ppo_optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.ppo_optimization_step = state_dict["ppo_optimization_step"]

        loaded_values = [self.step, self.consumed_samples, self.ppo_optimization_step]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def save(self, extra_candidates=None, is_train_end=False):
        self.model.prepare_for_training()
        # load back in the adam states if needed
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.step // self.num_steps_per_epoch
