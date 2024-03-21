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
from collections import defaultdict
from functools import partial

import pandas as pd
import requests
import torch
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from nemo_aligner.data.nlp.builders import collate_with_pad_to_max_batch
from nemo_aligner.utils.distributed import (
    SyncTimer,
    broadcast_2d_tensor,
    masked_global_mean_var,
    normalize_tensor,
    pad_tensors_to_max_global_seq_len,
)
from nemo_aligner.utils.ppo_utils import (
    calculate_advantages_and_returns,
    calculate_kl_penalty,
    calculate_ppo_rewards,
    create_mask,
)
from nemo_aligner.utils.server_utils import FutureResult, get_idx, set_idx
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.utils import clear_memory, cpu_dict, masked_mean


def pad_to_seq_length(list_of_tensors, seq_length, pad_value=0):
    tensors_padded = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

    return torch.nn.functional.pad(tensors_padded, (0, seq_length - tensors_padded.size(-1)), value=pad_value)


def rebalance_nd_tensor(tensor, group):
    """
    """
    num_samples = torch.as_tensor(tensor.size(0), dtype=torch.int64, device=torch.cuda.current_device())
    print("### NUM SAMPLES", num_samples)
    batch_num_per_rank = torch.zeros(
        torch.distributed.get_world_size(group), dtype=torch.int64, device=torch.cuda.current_device()
    )
    print("### BATCH NUM PER RANK", batch_num_per_rank, batch_num_per_rank.shape)
    torch.distributed.all_gather_into_tensor(batch_num_per_rank, num_samples, group=group)

    B = batch_num_per_rank.sum()
    other_dims = tensor.shape[1:]

    indices = batch_num_per_rank.cumsum(dim=0)
    output_tensor = torch.zeros(B, *other_dims, dtype=tensor.dtype, device=torch.cuda.current_device())
    print("### OUTPUT TENSOR", output_tensor.shape, output_tensor.dtype, tensor.dtype)

    # tensor_split is a view we can copy into
    output_tensor.tensor_split(indices.cpu())[torch.distributed.get_rank(group=group)].copy_(tensor)
    print("## BEFORE REDUCE")
    torch.distributed.all_reduce(output_tensor, group=group)
    print("## AFTER REDUCE")
    return output_tensor


def rebalance_dp(rollout_batch, shuffle_seed, use_trtllm_reshard=False):
    # assumes it's all padded
    rebalanced_rollout_batch = dict()

    for k, tensor in rollout_batch.items():
        print("### RANK BEFORE REBALANCE SIZE", k, torch.distributed.get_rank(), tensor.size())
        if use_trtllm_reshard:
            tensor = rebalance_nd_tensor(tensor, group=parallel_state.get_pipeline_model_parallel_group())

        tensor = rebalance_nd_tensor(tensor, group=parallel_state.get_data_parallel_group())
        rebalanced_rollout_batch[k] = tensor
        B = tensor.size(0)
        print("### RANK GLOBAL SIZE", torch.distributed.get_rank(), tensor.size())

    g_cpu = torch.Generator()
    g_cpu.manual_seed(shuffle_seed)
    indices = torch.randperm(B, generator=g_cpu).tensor_split(parallel_state.get_data_parallel_world_size())[
        parallel_state.get_data_parallel_rank()
    ]

    for k in rebalanced_rollout_batch:
        # anti alias the underlying tensor
        rebalanced_rollout_batch[k] = rebalanced_rollout_batch[k][indices].clone()
        print("### RANK LOCAL SIZE", torch.distributed.get_rank(), rebalanced_rollout_batch[k].size())

    return rebalanced_rollout_batch


def send_request(host, port, endpoint="/get_idx", batch_size=1, use_trtllm_reshard=False):
    src_rank = (
        parallel_state.get_tensor_model_parallel_src_rank() if use_trtllm_reshard else get_model_parallel_src_rank()
    )
    group = (
        parallel_state.get_tensor_model_parallel_group()
        if use_trtllm_reshard
        else parallel_state.get_model_parallel_group()
    )

    output = None

    if torch.distributed.get_rank() == src_rank:
        output = requests.put(
            url=f"http://{host}:{port}/{endpoint}",
            data=json.dumps({"batch_size": batch_size}),
            headers={"Content-Type": "application/json"},
        ).json()

        output = torch.as_tensor(output).view(1, -1)

    output = broadcast_2d_tensor(output, src_rank, group, dtype=torch.int64)
    return output.flatten().tolist()


def get_global_set(local_data_ids):
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, local_data_ids)
    global_set = set().union(*output)

    return global_set


def get_custom_data_parallel_world_size(trtllm_reshard=False):
    data_parallel_size = parallel_state.get_data_parallel_world_size()
    if trtllm_reshard:
        data_parallel_size = divide(
            torch.distributed.get_world_size(), parallel_state.get_tensor_model_parallel_world_size()
        )

    return data_parallel_size


def compute_num_rollout_microbatches(dataloader, trtllm_reshard=False):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        get_custom_data_parallel_world_size(trtllm_reshard=trtllm_reshard),
    )


def shard_rollout_batch_from_dp_to_pp(rollout_batches, pad_id):
    if len(rollout_batches) < 1:
        return rollout_batches

    pp_group_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_rollout_batches = len(rollout_batches) * pp_group_size

    group = parallel_state.get_pipeline_model_parallel_group()
    new_rollout_batch = {}

    # save the response lengths to unpad
    resp_lengths = [rb["response_tokens"].shape[-1] for rb in rollout_batches]
    resp_lengths = torch.tensor(resp_lengths, dtype=torch.int).cuda()
    global_resp_lengths = torch.empty(
        len(rollout_batches) * pp_group_size, dtype=torch.int, device=torch.cuda.current_device()
    )
    torch.distributed.all_gather_into_tensor(global_resp_lengths, resp_lengths, group=group)

    for key in rollout_batches[0].keys():
        if key == "response_tokens":
            list_of_things = []
            for item in rollout_batches:
                list_of_things.extend(item[key])

            local_output = pad_tensors_to_max_global_seq_len(list_of_things, pad_id, group).cuda()
            assert local_output.ndim == 2
            output_tensor = torch.empty(
                local_output.size(0) * pp_group_size,
                local_output.size(-1),
                dtype=local_output.dtype,
                device=torch.cuda.current_device(),
            )

        else:
            list_of_things = [r[key] for r in rollout_batches]
            local_output = torch.cat(list_of_things).cuda()
            output_tensor = torch.empty(
                local_output.size(0) * parallel_state.get_pipeline_model_parallel_world_size(),
                dtype=local_output.dtype,
                device=torch.cuda.current_device(),
            )

        torch.distributed.all_gather_into_tensor(output_tensor, local_output, group=group)
        new_rollout_batch[key] = output_tensor
    rollout_batches = list(get_iterator_k_split(new_rollout_batch, num_rollout_batches))

    # unpad
    for idx, rb in enumerate(rollout_batches):
        rb["response_tokens"] = rb["response_tokens"][..., : global_resp_lengths[idx]]
    return rollout_batches


class PPOTrainer:
    """Trainer to coordinate PPO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        rm_critic,
        logger,
        ckpt_callback,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.rm_critic = rm_critic
        self.logger = logger
        self.ckpt_callback = ckpt_callback
        self.use_trtllm_reshard = self.cfg.use_trtllm and parallel_state.get_pipeline_model_parallel_world_size() > 1

        self.consumed_samples = 0
        self.epoch = 0
        # the step here is PPO step
        self.step = 0
        # keep track of how many times we optimized the actor
        self.ppo_optimization_step = 0

        # used to compute the max step
        self._train_dataloader_len = len(train_dataloader)
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

        self.host = cfg.host
        self.port = cfg.port

        assert (
            self.cfg.save_interval % self.cfg.val_check_interval == 0
        ), f"{self.cfg.save_interval=} must be divisible by {self.cfg.val_check_interval=}"

    def generate_ppo_data(self, rollout_batches):
        """generate ppo specific data for training
        """
        ppo_rollout_data = defaultdict(list)
        ppo_rollout_metrics = defaultdict(lambda: 0)
        num_samples = 0

        def post_process_tensor(max_response_length, tensor):
            return map(lambda x: x.flatten(), tensor[..., :max_response_length].cpu().split(1, dim=0))

        assert len(rollout_batches) == 1

        max_response_length = rollout_batches[0]["response_lengths"].max().cuda()
        torch.distributed.all_reduce(
            max_response_length, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group()
        )

        for rollout_batch in rollout_batches:
            # NOTE: all items in rollout batch or out of this computation
            # must have a leading B dimension
            prompt_lengths = rollout_batch["prompt_lengths"]
            response_lengths = rollout_batch["response_lengths"]
            response_tokens = rollout_batch["response_tokens"]
            values = rollout_batch["values"]
            rewards = rollout_batch["rewards"]
            logprobs = rollout_batch["logprobs"]

            num_samples += prompt_lengths.size(0)

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

            func = partial(post_process_tensor, max_response_length.item() - 1)
            func2 = partial(post_process_tensor, max_response_length.item())

            # collect everything we need to train PPO
            ppo_rollout_data["mask"].extend(func(mask))
            ppo_rollout_data["advantages"].extend(func(advantages))
            ppo_rollout_data["prev_logprobs"].extend(func(logprobs))
            ppo_rollout_data["response_tokens"].extend(func2(response_tokens))
            # for the critic
            ppo_rollout_data["values"].extend(func(values))
            ppo_rollout_data["returns"].extend(func(returns))

            # compute metrics
            # NOTE: this metric is not accumulated globally so it will differ between DP ranks
            ppo_rollout_metrics["init_policy_kl"] += (
                masked_mean(init_policy_kl, mask, dim=-1).sum().item() if self.compute_init_policy_kl else 0
            )

        # average across the samples for the non global metrics
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        for k in ppo_rollout_data:
            rollout_batch_seq_length = 2048  # TODO: fix
            pad_value = self.model.tokenizer.eos_id

            # all other tensors in the rollout batch
            # will be B x S -1 (because we don't predict anything for the last token)
            if k != "response_tokens":
                pad_value = 0
                if rollout_batch_seq_length is not None:
                    rollout_batch_seq_length -= 1

            ppo_rollout_data[k] = pad_tensors_to_max_global_seq_len(
                ppo_rollout_data[k],
                pad_value=pad_value,
                group=parallel_state.get_data_parallel_group(),
                sequence_length_to_pad_to=rollout_batch_seq_length,
            )

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

    def stack_rollout_batches(self, rollout_batches):
        stacked_dict = dict()

        if len(rollout_batches) == 0:
            return stacked_dict

        keys = rollout_batches[0].keys()

        for k in keys:
            pad_value = 0
            rollout_batch_seq_length = self.rollout_batch_seq_length

            if k == "response_tokens":
                pad_value = self.model.tokenizer.eos_id

            if k == "values":
                if rollout_batch_seq_length is not None:
                    rollout_batch_seq_length -= 1

            list_of_tensors = [item[k] for item in rollout_batches]

            if all(map(lambda x: x.ndim == 1, list_of_tensors)):
                tensor = torch.cat(list_of_tensors)
            else:
                list_of_tensors = list(
                    itertools.chain(*(map(lambda x: x.flatten(), item.split(1, dim=0)) for item in list_of_tensors))
                )

                tensor = pad_to_seq_length(list_of_tensors, rollout_batch_seq_length, pad_value)

            stacked_dict[k] = tensor

        return stacked_dict

    def _run_inference(self, dataloader_iter, num_microbatches, is_validation):
        """this function is run per DP so the metrics need to be computed globally
        """
        rollout_batches = []
        futures = []

        print(f"num_microbatches {num_microbatches}")
        ids = [item[1]["idx"] for item in zip(range(num_microbatches), dataloader_iter)]
        local_ids = set(itertools.chain.from_iterable(ids))
        global_ids = get_global_set(local_ids)

        if torch.distributed.get_rank() == 0:
            set_idx(global_ids)
            print("### ALL IDS", global_ids)

        torch.distributed.barrier()

        start = time.time()

        request_time = time.time()
        idx = send_request(host=self.host, port=self.port, use_trtllm_reshard=self.use_trtllm_reshard)
        request_end_time = time.time()
        print("### REQUEST TOOK", request_end_time - request_time)

        print("## idx pre RANK IDX", torch.distributed.get_rank(), idx)
        while len(idx) > 0:
            idx = idx[0]
            batch = dataloader_iter._dataset[idx]
            batch = collate_with_pad_to_max_batch(
                self.model.cfg.ppo.length_params.max_length, self.model.tokenizer.eos_id, self.model.cfg
            )([batch])

            rollout_batch = self.model.infer(batch)
            rollout_batches.append(rollout_batch)

            futures.append(self.rm_critic.infer_rm_critic(rollout_batch, use_trtllm_reshard=self.use_trtllm_reshard))

            request_time = time.time()
            idx = send_request(host=self.host, port=self.port, use_trtllm_reshard=self.use_trtllm_reshard)
            request_end_time = time.time()

            print("### REQUEST TOOK", request_end_time - request_time)
            print("## idx pre RANK IDX", torch.distributed.get_rank(), idx)

        end = time.time()
        print("### GENERATE ONLY", end - start)

        start = time.time()
        torch.distributed.barrier()
        end = time.time()
        print("### DP SYNC TOOK", end - start)

        stacked_rollout_batch = self.stack_rollout_batches(rollout_batches)

        start = time.time()
        local_rollout_batch = rebalance_dp(stacked_rollout_batch, self.step, self.use_trtllm_reshard)
        del stacked_rollout_batch
        end = time.time()
        print("### REBALANCE DP TOOK", end - start)

        batched_response_tokens = local_rollout_batch["response_tokens"]

        start_time = time.time()
        rollout_logprobs = self.model.get_logprobs(batched_response_tokens)
        local_rollout_batch["logprobs"] = rollout_logprobs
        torch.cuda.synchronize()
        end_time = time.time()
        print("#### TIME FOR JUST LOG PROB", end_time - start_time)

        compute_init_policy_kl = not is_validation and self.compute_init_policy_kl
        if compute_init_policy_kl:
            start_time = time.time()

            rollout_init_logprobs = self.model.get_init_policy_logprobs(batched_response_tokens)
            local_rollout_batch["init_logprobs"] = rollout_init_logprobs

            torch.cuda.synchronize()
            end_time = time.time()
            print("#### TIME FOR INIT LOG PROB", end_time - start_time)

        start_time = time.time()

        rm_value_rollout_batches = []
        for future in futures:
            rewards, values = future.result(self.use_trtllm_reshard) if isinstance(future, FutureResult) else future
            rm_value_rollout_batches.append({"rewards": rewards, "values": values})

        rm_value_rollout_batches = self.stack_rollout_batches(rm_value_rollout_batches)
        # TODO: does this reshard into the same stuff?
        rm_value_rollout_batches = rebalance_dp(rm_value_rollout_batches, self.step, self.use_trtllm_reshard)
        local_rollout_batch.update(rm_value_rollout_batches)

        end_time = time.time()
        print("#### TIME SPEND WAITING ON THE CRITIC", end_time - start_time)

        # TODO: can rewrite the compute metrics
        rollout_batches = [local_rollout_batch]

        return rollout_batches, cpu_dict(self.compute_global_rollout_metrics(rollout_batches))

    def compute_global_rollout_metrics(self, rollout_batches):
        metrics = defaultdict(lambda: 0)
        table = {}

        num_samples = 0
        for i, rollout_batch in enumerate(rollout_batches):
            prompt_lengths = rollout_batch["prompt_lengths"]
            response_lengths = rollout_batch["response_lengths"]
            response_tokens = rollout_batch["response_tokens"]
            rewards = rollout_batch["rewards"]

            # table logging
            if i == 0:
                reward = rewards[0]
                prompt_length = prompt_lengths[0]
                response_length = response_lengths[0]
                response_token = response_tokens[0]

                table["reward"] = reward.item()
                table["prompt"] = self.model.tokenizer.ids_to_text(response_token[:prompt_length].tolist())
                table["response"] = self.model.tokenizer.ids_to_text(
                    response_token[prompt_length:response_length].tolist()
                )

            metrics["response_lengths"] += (response_lengths - prompt_lengths).sum()
            metrics["prompt_lengths"] += prompt_lengths.sum()
            metrics["rewards"] += rewards.sum()
            num_samples += prompt_lengths.size(0)

        tensor_to_accumulate = torch.tensor(
            [metrics["response_lengths"], metrics["prompt_lengths"], metrics["rewards"], num_samples],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group())

        (
            global_response_lengths,
            global_prompt_lengths,
            global_rewards,
            global_num_samples,
        ) = tensor_to_accumulate.tolist()

        metrics = {
            "table": table,
            "global_response_lengths_mean": global_response_lengths / global_num_samples,
            "global_prompt_lengths": global_prompt_lengths / global_num_samples,
            "global_rewards": global_rewards / global_num_samples,
        }

        return metrics

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        num_val_micro_batches = compute_num_rollout_microbatches(self.val_dataloader, self.use_trtllm_reshard)
        val_dataloader = iter(self.val_dataloader)

        _, rollout_metrics = self._run_inference(val_dataloader, num_val_micro_batches, is_validation=True)
        self.model.finish_inference()
        return rollout_metrics

    @torch.no_grad()
    def generate_rollouts(self, dataloader_iter, num_microbatches):
        self.model.prepare_for_inference()

        rollout_batches, rollout_metrics = self._run_inference(dataloader_iter, num_microbatches, is_validation=False)
        ppo_rollout_data, ppo_rollout_metrics = map(cpu_dict, self.generate_ppo_data(rollout_batches))

        self.model.finish_inference()

        self.consumed_samples += (
            ppo_rollout_data["response_tokens"].size(0) * parallel_state.get_data_parallel_world_size()
        )
        return ppo_rollout_data, rollout_metrics | ppo_rollout_metrics | {"consumed_samples": self.consumed_samples}

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

            dataloader_iter = iter(self.train_dataloader)

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            num_rollout_micro_batches = compute_num_rollout_microbatches(
                self.train_dataloader, self.use_trtllm_reshard
            )
            dp_size = parallel_state.get_data_parallel_world_size()

            num_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            for _ in global_pbar:
                print(f"***STEP {self.step}")

                step_metrics = {}
                timing_metrics = {}

                self.timer.start("rollout_time")
                ppo_rollout_data, metrics = self.generate_rollouts(dataloader_iter, num_rollout_micro_batches)
                self.timer.stop("rollout_time")
                timing_metrics["rollout_time"] = self.timer.get("rollout_time")

                # send critic train
                start_time = time.time()
                self.rm_critic.train(ppo_rollout_data)
                end_time = time.time()
                print("### CRITIC TRAIN TIME", end_time - start_time)

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

                self.logger.log_metrics(timing_metrics, step=self.step, prefix="timers/")

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

        if max_steps == -1:
            # the dataloader already knows how much longer
            # because consumed samples is resumed
            max_steps = self._train_dataloader_len
        else:
            # user specified the max step, figure out how much longer
            # we need to run for
            max_steps = max_steps - self.step

        self.max_steps = min(max_steps, self._train_dataloader_len) + self.step
