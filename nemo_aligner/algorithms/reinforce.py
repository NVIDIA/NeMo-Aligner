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

import itertools
from collections import UserDict
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from megatron.core import parallel_state as mcore_parallel_state
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from typing_extensions import Self

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.megatron_gpt_reinforce_actor import MegatronGPTReinforceActorModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    ScopedTimer,
    all_reduce_dict,
    masked_global_mean_var,
    normalize_tensor,
    rebalance_nd_tensor,
)
from nemo_aligner.utils.parallel_state import is_trt_llm_reshard, trt_llm_reshard_region
from nemo_aligner.utils.ppo_utils import calculate_kl_penalty, calculate_rloo_baseline, create_mask
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import clear_memory, cpu_dict, masked_mean


class ReinforceRolloutBatch(UserDict):
    @classmethod
    def from_rollout_batches(
        cls: Self, rollout_batches: List[Dict], eos_id: int, rollout_batch_seq_length: Optional[int]
    ) -> Self:
        """Given a list of rollout batches, stack the tensors within and put them in a single dictionary
        """
        stacked_dict = cls()

        for k in sorted(rollout_batches[0]):

            list_of_tensors = [item[k] for item in rollout_batches]

            if all(x.ndim == 1 for x in list_of_tensors):
                tensor = torch.cat(list_of_tensors)
            else:
                pad_value = eos_id if k == "response_tokens" else 0

                list_of_tensors = [row.flatten() for tensor in list_of_tensors for row in tensor]
                # TODO: can we avoid padding locally then padding globally?
                tensor = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

                # find the max sequence length globally
                max_seqlen = torch.tensor([tensor.size(-1)], dtype=torch.long, device=torch.cuda.current_device())
                torch.distributed.all_reduce(max_seqlen, op=torch.distributed.ReduceOp.MAX)

                if rollout_batch_seq_length is None or max_seqlen >= rollout_batch_seq_length:
                    pad_seq_len = max_seqlen.item()
                else:
                    # response tokens must be B x S because computing log probs requires us to offset by 1
                    pad_seq_len = rollout_batch_seq_length if k == "response_tokens" else rollout_batch_seq_length - 1

                tensor = torch.nn.functional.pad(tensor, (0, pad_seq_len - tensor.size(-1)), value=pad_value)

            stacked_dict[k] = tensor

        return stacked_dict

    def gather_and_balance_globally(self):
        global_rollout_batch = type(self)()

        for k, tensor in self.data.items():
            # with reshard enabled, PP groups turn into DP groups. So need to balance them first and then
            # balance by all the original DP groups
            # NOTE: this logic needs to use the pure parallel state, that is one without sharding but needs
            # to ping the is_trt_llm_reshard variable
            if is_trt_llm_reshard():
                tensor = rebalance_nd_tensor(tensor, group=mcore_parallel_state.get_pipeline_model_parallel_group())

            tensor = rebalance_nd_tensor(tensor, group=mcore_parallel_state.get_data_parallel_group())
            global_rollout_batch[k] = tensor

        return global_rollout_batch

    def chunk(self, rank, split_size, seed):
        chunked_rollout_batch = type(self)()

        batch_set = set(tensor.size(0) for tensor in self.data.values())
        assert len(batch_set) == 1, "batch sizes are not the same across the rollout batch"
        B = batch_set.pop()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        indices = torch.arange(B)

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][indices].clone()

        return chunked_rollout_batch


def compute_num_rollout_microbatches(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )


class ReinforceTrainer:
    """Trainer to coordinate REINFORCE training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: MegatronGPTReinforceActorModel,
        optimizer,
        scheduler,
        train_dataloader_builder,
        val_dataloader_builder,
        collate_fn,
        rm,
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
        self.rm = rm
        self.batch_iterator_cls = batch_iterator_cls
        self.logger = logger
        self.ckpt_callback = ckpt_callback

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.trtllm_reshard = "trt_llm" in cfg and cfg.trt_llm.enable and cfg.trt_llm.reshard

        self.consumed_samples = 0
        # the step here is REINFORCE step
        self.step = 0
        # keep track of how many times we optimized the actor
        self.reinforce_optimization_step = 0

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

        self.compute_init_policy_kl = self.cfg.initial_policy_kl_penalty > 0
        # size to pad our rollout batch to
        self.rollout_batch_seq_length = self.cfg.rollout_batch_seq_length

        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        self.timer = ScopedTimer(reduction="mean", sync_cuda=True, buffer_size=1)

    def generate_reinforce_data(self, rollout_batch):
        """generate reinforce specific data for training
        """
        reinforce_rollout_data = {}
        reinforce_rollout_metrics = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        prompt_tokens = rollout_batch["prompt_tokens"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]
        logprobs = rollout_batch["logprobs"]
        is_end = rollout_batch["is_end"]

        if self.compute_init_policy_kl:
            init_policy_kl = calculate_kl_penalty(
                log_probs_a=rollout_batch["logprobs"],
                log_probs_b=rollout_batch["init_logprobs"],
                use_absolute_kl=self.cfg.use_absolute_kl,
            )
        else:
            init_policy_kl = torch.tensor(0, dtype=logprobs.dtype, device=logprobs.device)

        mask = create_mask(values=logprobs, prompt_lengths=prompt_lengths, response_lengths=response_lengths)

        init_policy_kl = masked_mean(init_policy_kl, mask, dim=-1)
        rewards_with_kl = rewards - self.cfg.initial_policy_kl_penalty * init_policy_kl

        baseline = calculate_rloo_baseline(prompts=prompt_tokens, reward=rewards_with_kl, mask=is_end.float())

        # collect everything we need to train REINFORCE
        reinforce_rollout_data["mask"] = mask
        reinforce_rollout_data["rewards_with_kl"] = rewards_with_kl
        reinforce_rollout_data["baseline"] = baseline
        reinforce_rollout_data["response_tokens"] = response_tokens
        reinforce_rollout_data["is_end"] = is_end

        # compute metrics
        # these are not global yet
        reinforce_rollout_metrics["init_policy_kl"] = init_policy_kl.sum().item() if self.compute_init_policy_kl else 0
        reinforce_rollout_metrics["rewards_with_kl"] = rewards_with_kl.sum().item()
        reinforce_rollout_metrics["num_samples"] = prompt_lengths.size(0)

        # now the metrics are global
        reinforce_rollout_metrics = all_reduce_dict(
            reinforce_rollout_metrics,
            group=parallel_state.get_data_parallel_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
        num_samples = reinforce_rollout_metrics.pop("num_samples")
        reinforce_rollout_metrics = {k: v / num_samples for k, v in reinforce_rollout_metrics.items()}

        return reinforce_rollout_data, cpu_dict(reinforce_rollout_metrics)

    def _run_inference(self, dataloader_builder, consumed_samples, is_validation):
        """this function is run per DP so the metrics need to be computed globally
        assumes that the dataloader is built with the proper consumed samples value
        """
        reshard_context = trt_llm_reshard_region if self.trtllm_reshard else nullcontext

        rollout_batches, futures = [], []

        with reshard_context():
            # dataloader must be built within the reshard context because it uses DP rank and size
            dataloader = dataloader_builder(consumed_samples=consumed_samples)
            sampler_iter = iter(dataloader.batch_sampler)

            # must compute the number of microbatches in the reshard context
            # so the DP groups are correct
            num_microbatches = compute_num_rollout_microbatches(dataloader)

            with self.timer("batch_iterator_init"):
                batch_iterator = self.batch_iterator_cls(
                    sampler_iter, num_microbatches, dataloader.dataset, self.collate_fn
                )

            with self.timer("generate"):
                for batch in batch_iterator:
                    if not is_validation:
                        for _ in range(self.cfg.num_rollouts_per_prompt):
                            rollout_batch = self.model.infer(batch)
                            rollout_batch["prompt_tokens"] = batch["text"]
                            rollout_batches.append(rollout_batch)
                            futures.append(self.rm.infer_rm(rollout_batch))
                    else:
                        rollout_batch = self.model.infer(batch)
                        rollout_batches.append(rollout_batch)
                        futures.append(self.rm.infer_rm(rollout_batch))

            unbalanced_local_batch = ReinforceRolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=self.cfg.rollout_batch_seq_length,
            )
            global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()

        padded_rollout_sequence_length = global_rollout_batch["response_tokens"].size(-1)

        # the chunking must be outside of the TRT-LLM context because we do logprob calculation in nemo
        balanced_local_batch = global_rollout_batch.chunk(
            rank=parallel_state.get_data_parallel_rank(),
            split_size=parallel_state.get_data_parallel_world_size(),
            seed=self.step,
        )
        # since we compute the logprobs in nemo we need to disable the resharding
        batched_response_tokens = balanced_local_batch["response_tokens"]

        with self.timer("logprobs"):
            rollout_logprobs = self.model.get_inference_log_probs(batched_response_tokens)
            balanced_local_batch["logprobs"] = rollout_logprobs

        compute_init_policy_kl = not is_validation and self.compute_init_policy_kl
        if compute_init_policy_kl:
            with self.timer("init_logprobs"):
                rollout_init_logprobs = self.model.get_init_policy_logprobs(batched_response_tokens)
                balanced_local_batch["init_logprobs"] = rollout_init_logprobs

        # we send the request in sharded context, so we need to keep this sharding and then undo it
        with reshard_context():
            with self.timer("critic_wait"):
                rm_rollout_batches = []
                for future in futures:
                    rewards = future.result().squeeze(1)
                    rm_rollout_batches.append({"rewards": rewards})

            unbalanced_rm_batch = ReinforceRolloutBatch.from_rollout_batches(
                rm_rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=padded_rollout_sequence_length,
            )
            global_rm_batch = unbalanced_rm_batch.gather_and_balance_globally()

        # chunking needs to be outside of reshard region
        # NOTE: the seed here must be the same as the chunk before since we need to shuffle
        # these values the same way as the other values
        balanced_rm_batch = global_rm_batch.chunk(
            rank=parallel_state.get_data_parallel_rank(),
            split_size=parallel_state.get_data_parallel_world_size(),
            seed=self.step,
        )
        balanced_local_batch.update(balanced_rm_batch)

        global_rollout_batch.update(global_rm_batch)

        return balanced_local_batch, cpu_dict(self.compute_rollout_metrics(global_rollout_batch))

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

        rollout_batch, rollout_metrics = self._run_inference(
            self.train_dataloader_builder, consumed_samples=self.consumed_samples, is_validation=False
        )

        self.consumed_samples += rollout_metrics["rollout_size"]

        reinforce_rollout_data, reinforce_rollout_metrics = self.generate_reinforce_data(rollout_batch)

        with self.timer("finish_inference"):
            # Timing includes engine unloading if enabled
            self.model.finish_inference()

        return (
            reinforce_rollout_data,
            rollout_metrics | reinforce_rollout_metrics | {"consumed_samples": self.consumed_samples},
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

            metrics.update({"loss": loss_mean, "optim_step": self.reinforce_optimization_step})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.reinforce_optimization_step += 1

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
            num_steps_in_epoch = min(
                self.max_steps - self.step, self.num_steps_per_epoch - self.step % self.num_steps_per_epoch
            )
            loop_iter = range(num_steps_in_epoch)

            if not loop_iter:
                return  # training ended

            global_pbar = tqdm(
                loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="REINFORCE Global Step"
            )

            dp_size = parallel_state.get_data_parallel_world_size()

            num_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            self.run_timer.start_time()
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                clear_memory()
                with self.timer("rollout_time"):
                    reinforce_rollout_data, metrics, rollout_timer_metrics = self.generate_rollouts()
                # Consume rollout_time
                timing_metrics.update(self.timer.consume_durations())

                rollout_timer_metrics = all_reduce_dict(rollout_timer_metrics, op=torch.distributed.ReduceOp.MAX)
                timing_metrics.update(rollout_timer_metrics)

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

                rollout_size = reinforce_rollout_data["response_tokens"].size(0)
                rollout_dataloader_iter = get_iterator_k_split(
                    reinforce_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
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
            "reinforce_optimization_step": self.reinforce_optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.reinforce_optimization_step = state_dict["reinforce_optimization_step"]

        loaded_values = [self.step, self.consumed_samples, self.reinforce_optimization_step]

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
