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

import operator
import re
from collections import UserDict
from contextlib import nullcontext
from itertools import permutations, product
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.distributed
from megatron.core import parallel_state as mcore_parallel_state
from megatron.core.utils import divide
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import get_sandbox
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from typing_extensions import Self

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.megatron_gpt_ppo_actor import MegatronGPTActorModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    SyncTimer,
    all_reduce_dict,
    broadcast_2d_tensor_within_mp,
    masked_global_mean_var,
    rebalance_nd_tensor,
    run_if_model_parallel_src,
)
from nemo_aligner.utils.parallel_state import is_trt_llm_reshard, trt_llm_reshard_region
from nemo_aligner.utils.ppo_utils import calculate_ppo_rewards, create_mask
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import clear_memory, cpu_dict


def solve_24(numbers):
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

    for nums in permutations(numbers):
        for op1, op2, op3 in product(ops, repeat=3):
            # ((a op1 b) op2 c) op3 d
            try:
                if abs(ops[op3](ops[op2](ops[op1](nums[0], nums[1]), nums[2]), nums[3]) - 24) < 1e-10:
                    return f"(({nums[0]} {op1} {nums[1]}) {op2} {nums[2]}) {op3} {nums[3]}"
            except ZeroDivisionError:
                continue

            # (a op1 b) op2 (c op3 d)
            try:
                if abs(ops[op2](ops[op1](nums[0], nums[1]), ops[op3](nums[2], nums[3])) - 24) < 1e-10:
                    return f"({nums[0]} {op1} {nums[1]}) {op2} ({nums[2]} {op3} {nums[3]})"
            except ZeroDivisionError:
                continue

    return "Impossible"


def extract_box_content(text):
    # Pattern to match \box{} and capture its content
    pattern = r"\\boxed\{([^}]*)\}"
    # Find all matches
    matches = re.findall(pattern, text)
    return matches


def evaluate_all(answers):
    return [judget_game24(*item)[0] for item in answers]


def judget_game24(answer: str, input: str) -> bool:
    if answer == "" or answer is None:
        return False, "No answer provided"
    # strip away the content after the '=' sign
    answer = answer.split("=")[0].strip()

    # convert input to list of integers
    input_list = list(map(int, input.split(",")))
    solver_ans = solve_24(input_list)
    if solver_ans == "Impossible":
        # check if 'impossible' is a substring of the answer
        if "impossible" in answer.lower():
            return True, "there is no solution and the answer says impossible"
        else:
            return False, "there is no solution, but the answer is not impossible"
    # there must be a solution
    if answer == "Impossible":
        return False, "there is a solution, but the answer is impossible"
    try:
        # extract out the numbers from the answer
        numbers = re.findall(r"\d+", answer)
        if len(numbers) != 4:
            return False, "there are not exactly 4 numbers in the answer"
        # check if the numbers in the answer are the same as the input
        if set(map(int, numbers)) != set(input_list):
            return False, "the numbers in the answer are not the same as the input"

        # check if the answer evaluates to 24
        if eval(answer) != 24:
            return False, "the answer does not evaluate to 24"
        return True, "the answer is correct"
    except:
        return False, "the answer is not a valid arithmetic"


def sandbox_call(sandbox, answers):
    return [sandbox.is_output_correct(*item) for item in answers]


class PPORolloutBatch(UserDict):
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

    def chunk(self, rank, split_size):
        chunked_rollout_batch = type(self)()

        batch_set = set(tensor.size(0) for tensor in self.data.values())
        assert len(batch_set) == 1, "batch sizes are not the same across the rollout batch"
        B = batch_set.pop()

        # g_cpu = torch.Generator()
        # g_cpu.manual_seed(seed)
        # indices = torch.randperm(B, generator=g_cpu).tensor_split(split_size)[rank]
        # g_cpu = torch.Generator()
        # g_cpu.manual_seed(seed)
        indices = torch.arange(B).tensor_split(split_size)[rank]

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][indices].clone()

        return chunked_rollout_batch


def compute_num_rollout_microbatches(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )


def get_rloo_mean_std(grouped_rewards):
    """reward tensor should be B x num responses
    """
    num_responses = grouped_rewards.size(1)

    grouped_reward_mean = (
        grouped_rewards @ (1 - torch.eye(num_responses, dtype=grouped_rewards.dtype, device=grouped_rewards.device))
    ) / (num_responses - 1)

    grouped_square_mean = (
        grouped_rewards.square()
        @ (1 - torch.eye(num_responses, dtype=grouped_rewards.dtype, device=grouped_rewards.device))
    ) / (num_responses - 1)
    grouped_reward_std = (grouped_square_mean - grouped_reward_mean.square()).sqrt()

    return grouped_reward_mean, grouped_reward_std


class GRPOTrainer:
    """Trainer to coordinate PPO training
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
        rm_critic,
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
        self.rm_critic = rm_critic
        self.batch_iterator_cls = batch_iterator_cls
        self.logger = logger
        self.ckpt_callback = ckpt_callback

        # TODO: we need to send it once per TP rank, but do that later i guess...
        # self.sandbox = get_sandbox()

        # sanity check
        # assert self.sandbox.is_output_correct("123", "123.0"), "sandbox messed up"

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.trtllm_reshard = "trt_llm" in cfg and cfg.trt_llm.enable and cfg.trt_llm.reshard
        self.critic_warmup_steps = cfg.get("critic_warmup_steps", 0)

        self.consumed_samples = 0
        # the step here is PPO step
        self.step = 0
        # keep track of how many times we optimized the actor
        self.ppo_optimization_step = 0

        # compute `max_steps`
        train_dataloader = self.train_dataloader_builder(consumed_samples=0)

        self.num_steps_per_epoch = compute_num_steps_per_epoch(train_dataloader.batch_sampler)
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

    def generate_ppo_data(self, rollout_batch):
        """generate ppo specific data for training
        """
        ppo_rollout_data = {}
        ppo_rollout_metrics = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]

        logprobs = rollout_batch["logprobs"]
        is_end = rollout_batch["is_end"]

        num_responses = self.cfg.num_responses_per_prompt

        # TODO: switch this to RLOO like! this adds bias
        # compute advantages
        grouped_rewards = rewards.view(-1, num_responses)

        # compute median
        mask = 1 - torch.eye(num_responses, dtype=torch.long, device=grouped_rewards.device)
        gathered_medians = (
            (
                grouped_rewards.view(-1, 1, num_responses)
                .broadcast_to((grouped_rewards.size(0), *mask.shape))[:, mask.bool()]
                .view(grouped_rewards.size(0), num_responses, num_responses - 1)
            )
            .float()
            .quantile(q=0.5, dim=-1)
        )

        grouped_reward_mean, grouped_reward_std = get_rloo_mean_std(grouped_rewards)
        num_reward_non_zero = (grouped_rewards > 0).sum(-1).count_nonzero()
        num_std_zero = (grouped_reward_std == 0).count_nonzero()

        if self.cfg.use_raw_rewards:
            advantages = rewards
        elif self.cfg.use_median:
            advantages = grouped_rewards - gathered_medians

            if self.cfg.normalize_group_advantages:
                mean, std = get_rloo_mean_std(advantages)
                std_above_0 = std > 0

                if self.cfg.use_mean_for_group_adv:
                    advantages = advantages - mean

                advantages[std_above_0] /= std[std_above_0]

        else:
            advantages = grouped_rewards - grouped_reward_mean

            if self.cfg.normalize_rewards:
                # don't sharpen the ones with no variation
                # advantages = advantages / grouped_reward_std
                mask = grouped_reward_std > 0
                advantages[mask] = advantages[mask] / grouped_reward_std[mask]

        # TODO: consider normalizing the advantages
        advantages = advantages.flatten()
        num_advantages_not_zero = (advantages != 0).sum()

        # advantage estimation
        mask = create_mask(values=logprobs, prompt_lengths=prompt_lengths, response_lengths=response_lengths)

        # collect everything we need to train PPO
        ppo_rollout_data["mask"] = mask
        ppo_rollout_data["prev_logprobs"] = logprobs
        ppo_rollout_data["response_tokens"] = response_tokens
        ppo_rollout_data["is_end"] = is_end

        if "init_logprobs" in rollout_batch:
            ppo_rollout_data["init_logprobs"] = rollout_batch["init_logprobs"]

        ppo_rollout_metrics["num_samples"] = prompt_lengths.size(0)

        # group statistics
        ppo_rollout_metrics["num_prompt_with_positive_reward"] = (
            num_reward_non_zero * self.cfg.num_responses_per_prompt
        )
        ppo_rollout_metrics["num_prompt_with_zero_std"] = num_std_zero * self.cfg.num_responses_per_prompt
        ppo_rollout_metrics["grouped_reward_mean"] = (
            grouped_reward_mean.mean(-1).sum() * self.cfg.num_responses_per_prompt
        )
        ppo_rollout_metrics["percent_advantages_not_zero"] = num_advantages_not_zero

        ppo_rollout_metrics["grouped_reward_std"] = (
            grouped_reward_std.mean(-1).sum() * self.cfg.num_responses_per_prompt
        )

        # now the metrics are global
        ppo_rollout_metrics = all_reduce_dict(
            ppo_rollout_metrics, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM
        )

        num_samples = ppo_rollout_metrics.pop("num_samples")
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        mask = ppo_rollout_data["mask"]

        for key in ["advantages"]:
            tensor = advantages

            global_mean, global_var = masked_global_mean_var(
                tensor, torch.ones_like(tensor), group=parallel_state.get_data_parallel_group(),
            )

            if self.cfg.normalize_advantages:
                advantages = (advantages - global_mean) / global_var.sqrt()

            ppo_rollout_metrics[f"{key}_mean"] = global_mean.item()
            ppo_rollout_metrics[f"{key}_std"] = global_var.sqrt().item()

            max_tensor = tensor.max().item()
            min_tensor = tensor.min().item()
            reduce_tensor = torch.as_tensor(
                [-min_tensor, max_tensor], device=torch.cuda.current_device(), dtype=torch.float32
            )
            torch.distributed.all_reduce(
                reduce_tensor, torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group()
            )
            min_tensor, max_tensor = reduce_tensor.tolist()
            min_tensor = -min_tensor
            ppo_rollout_metrics[f"{key}_min"] = min_tensor
            ppo_rollout_metrics[f"{key}_max"] = max_tensor

        advantages = (torch.zeros_like(logprobs) + advantages.view(-1, 1)) * mask
        ppo_rollout_data["advantages"] = advantages

        return ppo_rollout_data, cpu_dict(ppo_rollout_metrics)

    def _run_inference(self, dataloader_builder, consumed_samples, is_validation):
        """this function is run per DP so the metrics need to be computed globally
        assumes that the dataloader is built with the proper consumed samples value
        """
        reshard_context = trt_llm_reshard_region if self.trtllm_reshard else nullcontext

        rollout_batches, all_rewards = [], []
        timer_metrics = {}

        num_responses = 1 if is_validation else self.cfg.num_responses_per_prompt

        with reshard_context():
            # dataloader must be built within the reshard context because it uses DP rank and size
            dataloader = dataloader_builder(consumed_samples=consumed_samples)
            sampler_iter = iter(dataloader.batch_sampler)

            self.timer.start("batch_iterator_init")
            batch_iterator = self.batch_iterator_cls(
                sampler_iter, dataloader.dataset, self.collate_fn, num_responses, self.cfg.rollout_micro_batch_size,
            )
            timer_metrics["batch_iterator_init"] = self.timer.stop_and_get_time("batch_iterator_init")

            self.timer.start("generate")

            for batch in batch_iterator:
                # during val we want to use greedy sampling
                rollout_batch = self.model.infer(batch, use_greedy=is_validation)
                rollout_batches.append(rollout_batch)

                # futures.append(self.rm_critic.infer_rm_critic(rollout_batch))
                texts = [
                    self.model.tokenizer.ids_to_text(item[length:].tolist())
                    for item, length in zip(rollout_batch["response_tokens"], batch["length"])
                ]
                answers = [(extract_answer(t), a) for t, a in zip(texts, batch["answers"])]
                # TODO: need to make this async for real
                output = run_if_model_parallel_src(evaluate_all, answers)
                if output is not None:
                    reward, _ = output
                    all_rewards.extend(reward)

            timer_metrics["generate"] = self.timer.stop_and_get_time("generate")

            unbalanced_local_batch = PPORolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=self.cfg.rollout_batch_seq_length,
            )
            # global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()
            balanced_local_batch = unbalanced_local_batch

        # since we compute the logprobs in nemo we need to disable the resharding
        batched_response_tokens = balanced_local_batch["response_tokens"]

        self.timer.start("logprobs")
        rollout_logprobs = self.model.get_inference_log_probs(batched_response_tokens)
        balanced_local_batch["logprobs"] = rollout_logprobs
        timer_metrics["logprobs"] = self.timer.stop_and_get_time("logprobs")

        compute_init_policy_kl = not is_validation and self.compute_init_policy_kl
        if compute_init_policy_kl:
            self.timer.start("init_logprobs")
            rollout_init_logprobs = self.model.get_init_policy_logprobs(batched_response_tokens)
            balanced_local_batch["init_logprobs"] = rollout_init_logprobs
            timer_metrics["init_logprobs"] = self.timer.stop_and_get_time("init_logprobs")

        if len(all_rewards) > 0:
            new_all_rewards = []

            for item in all_rewards:
                if item is None:
                    item = 0.0
                new_all_rewards.append(item)

            all_rewards = torch.as_tensor(
                new_all_rewards, dtype=torch.float32, device=torch.cuda.current_device()
            ).view(-1, 1)

        all_rewards = broadcast_2d_tensor_within_mp(all_rewards).flatten()

        balanced_local_batch.update({"rewards": all_rewards})
        # gather the global rollout batch
        global_rollout_batch = balanced_local_batch.gather_and_balance_globally()
        return (
            balanced_local_batch,
            cpu_dict(self.compute_rollout_metrics(global_rollout_batch, num_responses)),
            timer_metrics,
        )

    def compute_rollout_metrics(self, rollout_batch, num_responses):
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
            "rollout_size": prompt_lengths.size(0) // num_responses,
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

        self.consumed_samples += rollout_metrics["rollout_size"]

        ppo_rollout_data, ppo_rollout_metrics = self.generate_ppo_data(rollout_batch)

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
            if lr is not None:
                # Some optimizers like adafactor do not require a LR in their initializer
                metrics["lr"] = lr

            metrics.update({"loss": loss_mean, "optim_step": self.ppo_optimization_step})
            metrics["train_step_time"] = self.timer.stop_and_get_time("train_step_time")

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            metrics.pop("entropies", None)

            self.ppo_optimization_step += 1

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

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            dp_size = parallel_state.get_data_parallel_world_size()

            num_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            self.run_timer.start_time()
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                self.timer.start("rollout_time")
                clear_memory()
                ppo_rollout_data, metrics, timer_metrics = self.generate_rollouts()
                timing_metrics["rollout_time"] = self.timer.stop_and_get_time("rollout_time")

                # send critic train
                clear_memory()
                # self.rm_critic.train(ppo_rollout_data)

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
                metrics["epoch"] = self.epoch + 1
                self.logger.log_metrics(
                    metrics, step=self.step, prefix="train_rollouts/",
                )
                self.logger.log_table(
                    key="table/train_rollouts", dataframe=self.train_df, step=self.step,
                )

                if self.step == 0:
                    # at step 0 we do critic warmup which consumed samples
                    # we don't want to waste these samples and instead reset
                    # the consumed samples to as if we were to do no critic warmup
                    self.consumed_samples = metrics["rollout_size"]

                    self.timer.start("validation_time")
                    val_metrics = self.run_validation()
                    timing_metrics["validation_time"] = self.timer.stop_and_get_time("validation_time")

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
                    ppo_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
                )
                # start training
                clear_memory()
                self.timer.start("train_time")
                self.run_training(rollout_dataloader_iter)
                timing_metrics["train_time"] = self.timer.stop_and_get_time("train_time")

                self.logger.log_metrics(timing_metrics, step=self.step, prefix="timers/")

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
                    self.timer.start("validation_time")
                    val_metrics = self.run_validation()
                    timing_metrics["validation_time"] = self.timer.stop_and_get_time("validation_time")

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

        # _ = self.rm_critic.save()

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        # future.result()

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.step // self.num_steps_per_epoch
