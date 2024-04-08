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
import json
import time
from collections import UserDict, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable

import pandas as pd
import requests
import torch
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from nemo.utils import logging
from nemo_aligner.data.nlp.builders import collate_with_pad_to_max_batch
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    SyncTimer,
    all_reduce_dict,
    broadcast_2d_tensor,
    broadcast_2d_tensor_within_mp,
    masked_global_mean_var,
    normalize_tensor,
    rebalance_nd_tensor,
    run_if_model_parallel_src,
)
from nemo_aligner.utils.parallel_state import is_trt_llm_reshard, trt_llm_reshard_region
from nemo_aligner.utils.ppo_utils import (
    calculate_advantages_and_returns,
    calculate_kl_penalty,
    calculate_ppo_rewards,
    create_mask,
)
from nemo_aligner.utils.server_utils import FutureResult, get_idx, set_idx
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.utils import clear_memory, cpu_dict, masked_mean


@dataclass
class DefaultBatchIterator:
    sampler_iter: Iterator[int]
    num_microbatches: int
    dataset: Mapping
    collate_fn: Callable

    def __iter__(self):
        for _, ids in zip(range(self.num_microbatches), self.sampler_iter):
            batch = self.collate_fn([self.dataset[index] for index in ids])
            yield batch


@dataclass
class HTTPBatchIterator:
    host: str
    port: int
    sampler_iter: Iterator[int]
    num_microbatches: int
    dataset: Mapping
    collate_fn: Callable

    def __post_init__(self):
        local_ids = [ids for _, ids in zip(range(self.num_microbatches), self.sampler_iter)]
        self.desired_batch_size = len(local_ids[0]) if len(local_ids) > 0 else 1

        local_ids = set(itertools.chain.from_iterable(local_ids))
        global_ids = get_global_set(local_ids)

        if torch.distributed.get_rank() == 0:
            set_idx(global_ids)

        torch.distributed.barrier()

    def __iter__(self):
        ids = send_request(host=self.host, port=self.port, batch_size=self.desired_batch_size)

        while len(ids) > 0:
            batch = self.collate_fn([self.dataset[idx] for idx in ids])
            yield batch
            ids = send_request(host=self.host, port=self.port, batch_size=self.desired_batch_size)


class PPORolloutBatch(UserDict):
    @classmethod
    def from_rollout_batches(cls, rollout_batches, eos_id, rollout_batch_seq_length):
        stacked_dict = cls()

        if len(rollout_batches) == 0:
            return stacked_dict

        keys = rollout_batches[0].keys()

        for k in sorted(keys):

            list_of_tensors = [item[k] for item in rollout_batches]

            if all(map(lambda x: x.ndim == 1, list_of_tensors)):
                tensor = torch.cat(list_of_tensors)
            else:
                pad_seq_length = rollout_batch_seq_length
                pad_value = eos_id

                list_of_tensors = list(
                    itertools.chain(*(map(lambda x: x.flatten(), item.split(1, dim=0)) for item in list_of_tensors))
                )

                tensor = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

                # find the max sequence length
                max_seqlen = torch.tensor([tensor.size(-1)], dtype=torch.long, device=torch.cuda.current_device())
                torch.distributed.all_reduce(max_seqlen, op=torch.distributed.ReduceOp.MAX)

                if pad_seq_length is None:
                    # pad to the max sequence length if not specified`
                    pad_seq_length = max_seqlen.item()

                if max_seqlen > rollout_batch_seq_length:
                    logging.warning(
                        "specified rollout sequence length to be padded to {} but actual rollout sequence length is {}".format(
                            rollout_batch_seq_length, max_seqlen
                        )
                    )
                    pad_seq_length = max_seqlen

                if k != "response_tokens":
                    pad_value = 0
                    pad_seq_length -= 1

                tensor = torch.nn.functional.pad(tensor, (0, pad_seq_length - tensor.size(-1)), value=pad_value)

            stacked_dict[k] = tensor

        return stacked_dict

    def gather_and_balance_globally(self):
        global_rollout_batch = type(self)()

        for k, tensor in self.data.items():
            if is_trt_llm_reshard():
                tensor = rebalance_nd_tensor(tensor, group=parallel_state.get_pipeline_model_parallel_group())

            tensor = rebalance_nd_tensor(tensor, group=parallel_state.get_data_parallel_group())
            global_rollout_batch[k] = tensor

        return global_rollout_batch

    def chunk(self, rank, split_size, seed):
        chunked_rollout_batch = type(self)()

        batch_set = set(tensor.size(0) for tensor in self.data.values())
        assert len(batch_set) == 1, "batch sizes are not the same across rollout batch"
        B = batch_set.pop()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        indices = torch.randperm(B, generator=g_cpu).tensor_split(split_size)[rank]

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][indices].clone()

        return chunked_rollout_batch


def send_request(host, port, endpoint="/get_idx", batch_size=1):
    output = run_if_model_parallel_src(
        requests.put,
        url=f"http://{host}:{port}/{endpoint}",
        data=json.dumps({"batch_size": batch_size}),
        headers={"Content-Type": "application/json"},
    )

    if output is not None:
        output = output.json()
        output = torch.as_tensor(output).view(1, -1)

    output = broadcast_2d_tensor_within_mp(output)
    return output.flatten().tolist()


def get_global_set(local_data_ids):
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, local_data_ids)
    global_set = set().union(*output)

    return global_set


def compute_num_rollout_microbatches(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )


class PPOTrainer:
    """Trainer to coordinate PPO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader_builder,
        val_dataloader_builder,
        collate_fn,
        rm_critic,
        batch_iterator_cls,
        logger,
        ckpt_callback,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader_builder = train_dataloader_builder
        self.val_dataloader_builder = val_dataloader_builder
        self.collate_fn = collate_fn
        self.rm_critic = rm_critic
        self.batch_iterator_cls = batch_iterator_cls
        self.logger = logger
        self.ckpt_callback = ckpt_callback

        # TODO: need to knob it so we can actually disable it on nemo export side
        self.trtllm_reshard = cfg.trt_llm.enable and cfg.trt_llm.reshard

        self.consumed_samples = 0
        self.epoch = 0
        # the step here is PPO step
        self.step = 0
        # keep track of how many times we optimized the actor
        self.ppo_optimization_step = 0

        self.set_max_steps()

        self.compute_init_policy_kl = self.cfg.initial_policy_kl_penalty > 0
        # size to pad our rollout batch to
        self.rollout_batch_seq_length = self.cfg.rollout_batch_seq_length

        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        assert (
            self.cfg.save_interval % self.cfg.val_check_interval == 0
        ), f"{self.cfg.save_interval=} must be divisible by {self.cfg.val_check_interval=}"

    def generate_ppo_data(self, rollout_batch):
        """generate ppo specific data for training
        """
        ppo_rollout_data = {}
        ppo_rollout_metrics = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        values = rollout_batch["values"]
        rewards = rollout_batch["rewards"]
        logprobs = rollout_batch["logprobs"]

        if self.compute_init_policy_kl:
            init_policy_kl = calculate_kl_penalty(
                log_probs_a=rollout_batch["logprobs"],
                log_probs_b=rollout_batch["init_logprobs"],
                use_absolute_kl=self.cfg.use_absolute_kl,
            )
        else:
            init_policy_kl = torch.tensor(0, dtype=logprobs.dtype, device=logprobs.device)

        rewards_with_kl = calculate_ppo_rewards(
            values, rewards, response_lengths, init_policy_kl, self.cfg.initial_policy_kl_penalty
        )
        mask = create_mask(values=values, prompt_lengths=prompt_lengths, response_lengths=response_lengths)

        # TODO(geshen): we may not need this mask
        advantages, returns = calculate_advantages_and_returns(
            values=values,
            rewards=rewards_with_kl,
            discount_factor=self.cfg.discount_factor,
            gae_lambda=self.cfg.gae_lambda,
            mask=mask,
        )

        # collect everything we need to train PPO
        ppo_rollout_data["mask"] = mask
        ppo_rollout_data["advantages"] = advantages
        ppo_rollout_data["prev_logprobs"] = logprobs
        ppo_rollout_data["response_tokens"] = response_tokens
        # for the critic
        ppo_rollout_data["values"] = values
        ppo_rollout_data["returns"] = returns

        # compute metrics
        # these are not global yet
        ppo_rollout_metrics["global_init_policy_kl"] = (
            masked_mean(init_policy_kl, mask, dim=-1).sum().item() if self.compute_init_policy_kl else 0
        )
        ppo_rollout_metrics["global_rewards_with_kl"] = masked_mean(rewards_with_kl, mask, dim=-1).sum().item()
        ppo_rollout_metrics["num_samples"] = prompt_lengths.size(0)

        # now the metrics are global
        ppo_rollout_metrics = all_reduce_dict(
            ppo_rollout_metrics, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM
        )
        num_samples = ppo_rollout_metrics.pop("num_samples")
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        mask = ppo_rollout_data["mask"]
        for key in ["advantages", "returns", "values"]:
            tensor = ppo_rollout_data[key]

            global_mean, global_var = masked_global_mean_var(
                tensor, mask, group=parallel_state.get_data_parallel_group(),
            )
            ppo_rollout_metrics[f"global_{key}_mean"] = global_mean.item()
            ppo_rollout_metrics[f"global_{key}_std"] = global_var.sqrt().item()

        if self.cfg.normalize_advantages:
            ppo_rollout_data["advantages"] = normalize_tensor(
                ppo_rollout_data["advantages"],
                ppo_rollout_data["mask"],
                group=parallel_state.get_data_parallel_group(),
            )

        return ppo_rollout_data, cpu_dict(ppo_rollout_metrics)

    def _run_inference(self, dataloader_builder, consumed_samples, is_validation):
        """this function is run per DP so the metrics need to be computed globally
        assumes that the dataloader is built with the proper consumed samples value
        """
        reshard_context = trt_llm_reshard_region if self.trtllm_reshard else nullcontext

        rollout_batches, futures = [], []
        timer_metrics = {}

        with reshard_context():
            # dataloader must be built within the reshard context because it uses DP rank and size
            dataloader = dataloader_builder(consumed_samples=consumed_samples)
            sampler_iter = iter(dataloader.batch_sampler)

            # must compute the number of microbatches in the reshard context
            # so the DP groups are correct
            num_microbatches = compute_num_rollout_microbatches(dataloader)

            self.timer.start("batch_iterator_init")
            batch_iterator = self.batch_iterator_cls(
                sampler_iter, num_microbatches, dataloader.dataset, self.collate_fn
            )
            timer_metrics["batch_iterator_init"] = self.timer.stop_and_get_time("batch_iterator_init")

            self.timer.start("generate")
            for batch in batch_iterator:
                rollout_batch = self.model.infer(batch)
                rollout_batches.append(rollout_batch)

                futures.append(self.rm_critic.infer_rm_critic(rollout_batch))

            timer_metrics["generate"] = self.timer.stop_and_get_time("generate")

            unbalanced_local_batch = PPORolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=self.cfg.rollout_batch_seq_length,
            )
            global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()
            balanced_local_batch = global_rollout_batch.chunk(
                parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size(), self.step
            )

        # since we compute the logprobs in nemo we need to disable the resharding
        batched_response_tokens = balanced_local_batch["response_tokens"]

        self.timer.start("logprobs")
        rollout_logprobs = self.model.get_logprobs(batched_response_tokens)
        balanced_local_batch["logprobs"] = rollout_logprobs
        timer_metrics["logprobs"] = self.timer.stop_and_get_time("logprobs")

        compute_init_policy_kl = not is_validation and self.compute_init_policy_kl
        if compute_init_policy_kl:
            self.timer.start("init_logprobs")
            rollout_init_logprobs = self.model.get_init_policy_logprobs(batched_response_tokens)
            balanced_local_batch["init_logprobs"] = rollout_init_logprobs
            timer_metrics["init_logprobs"] = self.timer.stop_and_get_time("init_logprobs")

        # we send the request in sharded context, so we need to keep this sharding and then undo it
        with reshard_context():
            self.timer.start("critic_wait")
            rm_value_rollout_batches = []
            for future in futures:
                rewards, values = future.result() if isinstance(future, FutureResult) else future
                rm_value_rollout_batches.append({"rewards": rewards, "values": values})
            timer_metrics["critic_wait"] = self.timer.stop_and_get_time("critic_wait")

            unbalanced_rm_value_batch = PPORolloutBatch.from_rollout_batches(
                rm_value_rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=self.cfg.rollout_batch_seq_length,
            )
            global_rm_value_batch = unbalanced_rm_value_batch.gather_and_balance_globally()
            balanced_rm_value_batch = global_rm_value_batch.chunk(
                parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size(), self.step
            )
            balanced_local_batch.update(balanced_rm_value_batch)

        global_rollout_batch.update(global_rm_value_batch)

        return balanced_local_batch, cpu_dict(self.compute_global_rollout_metrics(global_rollout_batch)), timer_metrics

    def compute_rollout_metrics(self, rollout_batch):
        table = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]

        reward = rewards[0]
        prompt_length = prompt_lengths[0]
        response_length = response_lengths[0]
        response_token = response_tokens[0]

        table["reward"] = reward.item()
        table["prompt"] = self.model.tokenizer.ids_to_text(response_token[:prompt_length].tolist())
        table["response"] = self.model.tokenizer.ids_to_text(response_token[prompt_length:response_length].tolist())

        metrics = {
            "table": table,
            "consumed_samples": prompt_lengths.size(0).item(),
            "global_response_lengths_mean": response_lengths.mean(),
            "global_prompt_lengths": prompt_lengths.mean(),
            "global_rewards": rewards.mean(),
        }

        return metrics

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        _, rollout_metrics, _ = self._run_inference(
            self.val_dataloader_builder, consumed_samples=0, is_validation=True
        )

        self.model.finish_inference()
        return rollout_metrics

    @torch.no_grad()
    def generate_rollouts(self):
        timing_metrics = {}

        self.timer.start("prepare_for_inference")
        self.model.prepare_for_inference()
        timing_metrics["prepare_for_inference"] = self.timer.stop_and_get_time("prepare_for_inference")

        rollout_batch, rollout_metrics, timer_metrics = self._run_inference(
            self.train_dataloader_builder, consumed_samples=self.consumed_samples, is_validation=False
        )
        self.consumed_samples += rollout_metrics["consumed_samples"]

        ppo_rollout_data, ppo_rollout_metrics = map(cpu_dict, self.generate_ppo_data(rollout_batch))

        self.timer.start("finish_inference")
        self.model.finish_inference()
        timing_metrics["finish_inference"] = self.timer.stop_and_get_time("finish_inference")

        timing_metrics.update(timer_metrics)

        return (
            ppo_rollout_data,
            rollout_metrics | ppo_rollout_metrics | {"consumed_samples": self.consumed_samples},
            timing_metrics,
        )

    def run_training(self, dataloader_iter):
        self.model.prepare_for_training()

        for batch in dataloader_iter:
            self.timer.start("train_step_time")
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

            metrics.update({"lr": lr, "loss": loss_mean, "optim_step": self.ppo_optimization_step})

            self.timer.stop("train_step_time")
            metrics["train_step_time"] = self.timer.get("train_step_time")

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.ppo_optimization_step += 1

        self.model.finish_training()

        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()
        return loss_mean, metrics

    def fit(self):
        if self.cfg.max_epochs is not None and self.cfg.max_epochs > 1:
            # because we need to figure out a nice way to reset the shuffling on our dataset
            # otherwise epoch > 1 will loop over the dataset in the same order
            raise ValueError("epoch > 1 is not supported")

        epoch_iter = range(self.epoch, self.cfg.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        for _ in epoch_iter:
            loop_iter = range(self.step, self.max_steps)

            # TODO(geshen): to change for when we support > 1 epoch
            if len(loop_iter) <= 0:
                return  # training ended

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            dp_size = parallel_state.get_data_parallel_world_size()

            num_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            for _ in global_pbar:
                print(f"***STEP {self.step}")
                step_metrics = {}
                timing_metrics = {}

                self.timer.start("rollout_time")
                ppo_rollout_data, metrics, timer_metrics = self.generate_rollouts()

                self.timer.stop("rollout_time")
                timing_metrics["rollout_time"] = self.timer.get("rollout_time")

                # send critic train
                start_time = time.time()
                self.rm_critic.train(ppo_rollout_data)
                end_time = time.time()
                print("### CRITIC TRAIN TIME", end_time - start_time)

                timer_metrics = all_reduce_dict(timer_metrics, op=torch.distributed.ReduceOp.MAX)
                timing_metrics.update(timer_metrics)

                # logging
                table_metrics = metrics.pop("table")
                self.train_df.loc[len(self.train_df)] = [
                    self.step,
                    table_metrics["prompt"],
                    table_metrics["response"],
                    table_metrics["reward"],
                ]
                self.logger.log_metrics(
                    metrics, step=self.step, prefix="train_rollouts/",
                )
                self.logger.log_table(
                    key="table/train_rollouts", dataframe=self.train_df, step=self.step,
                )

                rollout_size = ppo_rollout_data["response_tokens"].size(0)
                rollout_dataloader_iter = get_iterator_k_split(
                    ppo_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
                )
                # start training
                clear_memory()
                self.timer.start("train_time")
                self.run_training(rollout_dataloader_iter)
                self.timer.stop("train_time")
                timing_metrics["train_time"] = self.timer.get("train_time")

                self.logger.log_metrics(timing_metrics, step=self.step, prefix="timers/")

                self.step += 1

                is_train_end = self.step == self.max_steps
                run_val = (self.step % self.cfg.val_check_interval == 0) or is_train_end
                if run_val:
                    print(f"***VAL {self.step}")
                    self.timer.start("validation_time")
                    val_metrics = self.run_validation()
                    self.timer.stop("validation_time")
                    timing_metrics["validation_time"] = self.timer.get("validation_time")

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

                step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                if run_val and (self.step % self.cfg.save_interval == 0 or is_train_end):
                    self.save(step_metrics, is_train_end=is_train_end)

            self.epoch += 1

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
        self.epoch = state_dict["epoch"]
        self.ppo_optimization_step = state_dict["ppo_optimization_step"]

        loaded_values = [self.step, self.consumed_samples, self.epoch, self.ppo_optimization_step]

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

        future = self.rm_critic.save()

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        future.result()

        self.model.finish_training()

    def set_max_steps(self):
        max_steps = self.cfg.get("max_steps", -1)

        train_dataloader_len = len(self.train_dataloader_builder(consumed_samples=self.consumed_samples))

        if max_steps == -1:
            # the dataloader already knows how much longer
            # because consumed samples is resumed
            max_steps = train_dataloader_len
        else:
            # user specified the max step, figure out how much longer
            # we need to run for
            max_steps = max_steps - self.step

        self.max_steps = min(max_steps, train_dataloader_len) + self.step
