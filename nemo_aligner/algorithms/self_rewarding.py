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

import copy
import itertools
import json
import math
import os
from collections import defaultdict
from functools import partial
from jinja2 import meta
from statistics import mean
from textwrap import dedent

import numpy as np
import pandas as pd
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import SyncTimer, broadcast_2d_tensor_within_pp
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches, compute_num_steps_per_epoch
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM
from nemo_aligner.utils.utils import (
    batch_pad_to_fixed_len,
    clear_memory,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)

"""
GPTSFTChatDataset output is dict with keys: ['input_ids', 'mask', 'context_ids', 'answer_ids', 'metadata']

input_ids: torch.LongTensor - the entire prompt + response, including the system preamble which is specified by "system" in the jsonl
mask: torch.BoolTensor with False for the preamble+prompt, and True for the response
context_ids: torch.LongTensor - the entire preamble + prompt
answer_ids: torch.LongTensor - the entire response only
metadata: dict - with keys "system" for the preamble, and "mask" which is "User" or "Assistant"
"""


def self_rewarding_custom_collate(batch, eos_id):
    input_ids = [item["input_ids"] for item in batch]
    masks = [item["mask"] for item in batch]
    context_ids = [item["context_ids"] for item in batch]
    answer_ids = [item["answer_ids"] for item in batch]
    context_lengths = torch.LongTensor([len(x) for x in context_ids])
    combined_lengths = torch.LongTensor([len(x) for x in input_ids])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=eos_id)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    context_ids = torch.nn.utils.rnn.pad_sequence(context_ids, batch_first=True, padding_value=eos_id)
    answer_ids = torch.nn.utils.rnn.pad_sequence(answer_ids, batch_first=True, padding_value=eos_id)

    output = {
        "prompts_and_answers": input_ids,
        "masks": masks,
        "prompts_only": context_ids,
        "answers_only": answer_ids,
        "prompt_lengths": context_lengths,
        "combined_lengths": combined_lengths,
    }

    return output


import re

import jinja2

jinja2_env = jinja2.Environment()


def db(msg):
    if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
        print(f"*** rank[{torch.distributed.get_rank()}]  {msg}", flush=True)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def find_variables_from_jinja_template(template: str):
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {
        "reward"
    }, 'reward template must include "reward" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward=r"([0-9\.]+)")

    def parse_reward_fn(llm_response: str) -> float:
        result = re.search(rf"{reward_regex_str}", llm_response)

        if not exists(result) or result.groups == 0:
            return None

        group_one = result.groups(1)[0] if isinstance(result.groups(1), tuple) else result.groups(1)

        try:
            ret = float(group_one)
        except:
            ret = None

        return ret

    return parse_reward_fn


def create_meta_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {
        "reward"
    }, 'reward template must include "reward" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward=r"([A-B\.]+)")

    # @always(lambda: randrange(0, 10))
    def parse_reward_fn(llm_response: str) -> float:
        result = re.search(rf"{reward_regex_str}", llm_response)

        if not exists(result) or result.groups == 0:
            return None

        group_one = result.groups(1)[0] if isinstance(result.groups(1), tuple) else result.groups(1)

        if group_one == "A" or group_one == "B":
            return group_one
        else:
            return None

    return parse_reward_fn


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


# Hyper-parameters of the Elo scores computation.
SCALE = 400
INIT_RATING = 1000


def ids_to_text(self, ids):
    tokens = self.ids_to_tokens(ids)
    text = self.tokens_to_text(tokens)
    return text


class SelfRewardingTrainer:
    """Trainer to coordinate Self-Rewarding training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.step = 0
        self.consumed_samples = 0

        self.ckpt_callback = ckpt_callback

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(
            self.train_dataloader.batch_sampler, self.cfg.get("limit_train_batches", 1.0)
        )

        if isinstance(self.cfg.get("limit_train_batches", 1.0), int):
            self.train_dataloader.batch_sampler.total_samples = min(
                self.train_dataloader.batch_sampler.total_samples,
                self.cfg.limit_train_batches * self.train_dataloader.batch_sampler.global_batch_size,
            )
            if hasattr(self.train_dataloader.batch_sampler, "last_batch_size"):
                self.train_dataloader.batch_sampler.last_batch_size = 0

        self.limit_val_batches = compute_limit_batches(len(val_dataloader), self.cfg.limit_val_batches)
        self.val_check_interval = (
            int(self.cfg.val_check_interval * self.num_steps_per_epoch)
            if isinstance(self.cfg.val_check_interval, float)
            else self.cfg.val_check_interval
        )
        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        self.spin_config = OmegaConf.to_container(self.model.cfg.spin, resolve=True)
        if isinstance(self.spin_config["length_control"], (float, int)):
            self.rho = self.spin_config["length_control"]
        elif isinstance(self.spin_config["length_control"], list):
            self.rho = 0.0
        else:
            raise TypeError(
                f"`length_control` must be a scalar or list, but got {type(self.spin_config['length_control'])}"
            )

        self.num_responses_to_gen = self.model.cfg.spin.num_responses_to_gen
        self.num_evals_to_average = self.model.cfg.spin.num_evals_to_average
        self.first_iteration_sft = self.model.cfg.spin.get("first_iteration_sft", False)
        self.use_meta_judge = self.model.cfg.spin.get("use_meta_judge", False)
        self.meta_judge_pcnt = self.model.cfg.spin.get("meta_judge_pcnt", -1.0)
        self.length_params = OmegaConf.to_container(self.model.cfg.spin.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.model.cfg.spin.sampling_params, resolve=True)
        self.max_gen_seq_len = self.length_params["max_length"]
        dp_batch_size = self.model.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
        assert (
            self.model.cfg.spin.rollout_micro_batch_size % dp_batch_size == 0
        ), f"rollout_micro_batch_size [{self.model.cfg.spin.rollout_micro_batch_size}] must be a multiple of GBS [{self.model.cfg.global_batch_size}] // DP [{parallel_state.get_data_parallel_world_size()}]"
        self.rollout_micro_batch_size = self.model.cfg.spin.rollout_micro_batch_size
        assert self.rollout_micro_batch_size > 0, "`rollout_micro_batch_size` must be > 0"

        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "chosen_response", "rejected_response"])

        # This is a hack to work around the fact that, by default, `AutoTokenizer` discards special tokens in `ids_to_text()`.
        if isinstance(self.model.tokenizer, AutoTokenizer):
            self.tokenizer = copy.copy(self.model.tokenizer)
            self.tokenizer.ids_to_text = partial(ids_to_text, self.tokenizer)
        else:
            self.tokenizer = self.model.tokenizer

        self.prompt_template = self.model.cfg.spin.get("llm_judge_prompt").strip()
        self.meta_judge_template = self.model.cfg.spin.get("llm_meta_judge_prompt").strip()
        self.reward_regex_template = self.model.cfg.spin.get("judge_reward_regex")
        self.meta_judge_reward_regex_template = self.model.cfg.spin.get("meta_judge_reward_regex")
        self.judge_score_low = self.model.cfg.spin.get("judge_score_low", 0)
        self.judge_score_high = self.model.cfg.spin.get("judge_score_high", 5)
        self.meta_max_relative_pcnt = self.model.cfg.spin.get("meta_max_relative_pcnt", 0.4)

        assert find_variables_from_jinja_template(self.prompt_template) == {
            "prompt",
            "response",
        }, "llm_judge_prompt must include `prompt` and `response` templating variables"
        assert find_variables_from_jinja_template(self.meta_judge_template) == {
            "prompt",
            "response",
            "judgement_a",
            "judgement_b",
        }, "llm_meta_judge_prompt must include `prompt`, `response`, `judgement_a`, and `judgement_b` templating variables"
        assert find_variables_from_jinja_template(self.reward_regex_template) == {
            "reward"
        }, "judge_reward_regex must include `reward` templating variable"
        assert find_variables_from_jinja_template(self.meta_judge_reward_regex_template) == {
            "reward"
        }, "meta_judge_reward_regex must include `reward` templating variable"

        self.template_fn = jinja2_env.from_string(self.prompt_template).render
        self.meta_judge_template_fn = jinja2_env.from_string(self.meta_judge_template).render
        self.parse_reward_fn = create_parse_reward_fn(self.reward_regex_template)
        self.meta_parse_reward_fn = create_meta_parse_reward_fn(self.meta_judge_reward_regex_template)

        self.use_trtllm_generation = self.cfg.trt_llm.get("enable", False) if "trt_llm" in self.cfg else False
        if self.use_trtllm_generation:
            # assert HAVE_TRTLLM, "TRTLLM generation was enabled but TRTLLM libraries could not be successfully imported"
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.model.cfg,
                end_strings=self.sampling_params["end_strings"],
                tokenizer=self.model.tokenizer,
                sample_temperature=self.sampling_params["temperature"],
                sample_top_k=self.sampling_params["top_k"],
                sample_top_p=self.sampling_params["top_p"],
                repetition_penalty=self.sampling_params["repetition_penalty"],
                max_generation_length=self.length_params["max_length"],
                max_input_len=self.cfg.trt_llm.get(
                    "max_input_len", self.model.cfg.encoder_seq_length - self.length_params["max_length"]
                ),
                generation_batch_size=self.model.cfg.spin.get("rollout_micro_batch_size", 4),
                use_greedy=self.sampling_params.get("use_greedy", False),
                trt_model_type=self.cfg.trt_llm.get("model_type", "gptnext"),
                seed=self.model.cfg.get("seed", None),
                unload_engine_train=self.cfg.trt_llm.get("unload_engine_train", False),
                reshard_model=False,
            )

    def validation_step(self, global_batch):
        # these things should go into a GPTModel wrapper
        self.model.prepare_for_validation_step()

        loss_mean, metrics = self.model.get_loss_and_metrics_vanilla_sft(batch=global_batch, forward_only=True)

        self.model.finish_validation_step()
        return loss_mean, metrics

    @torch.no_grad()
    def run_validation(self):
        loss_means = []
        val_metrics = defaultdict(list)

        val_pbar = tqdm(
            zip(range(self.limit_val_batches), self.val_dataloader),
            total=self.limit_val_batches,
            leave=True,
            desc="Validation steps",
        )

        for _, batch in val_pbar:
            # self.model.prepare_for_validation()

            self.timer.start("validation_step_time")
            loss_mean, metrics = self.validation_step(batch)
            self.timer.stop("validation_step_time")
            validation_step_time = self.timer.get("validation_step_time")

            metrics["validation_step_time"] = validation_step_time

            loss_means.append(loss_mean)
            for k, v in metrics.items():
                val_metrics[k].append(v)
            log_val_metrics = {f"val_{k}": v for k, v in metrics.items()}
            val_pbar.set_postfix(log_val_metrics)

            # self.model.finish_validation()

        val_metrics = {k: mean(v) for k, v in val_metrics.items()}
        return mean(loss_means), val_metrics

    def train_single_step_sft(self, global_batch):
        self.optimizer.zero_grad()

        self.model.prepare_for_training_step()

        # NOTE: assume backward is called on the loss already
        loss_mean, metrics = self.model.get_loss_and_metrics_vanilla_sft(batch=global_batch, forward_only=False)

        self.model.finish_training_step()

        grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()
        self.scheduler.step()

        trainer_metrics = {}
        if grad_norm is not None:
            trainer_metrics["grad_norm"] = grad_norm
        trainer_metrics.update({"lr": lr, "loss": loss_mean})

        return loss_mean, {**metrics, **trainer_metrics}

    def train_single_step_dpo(self, global_batch):
        self.optimizer.zero_grad()

        self.model.prepare_for_training_step()

        # NOTE: assume backward is called on the loss already
        loss_mean, metrics = self.model.get_loss_and_metrics(batch=global_batch, forward_only=False)

        self.model.finish_training_step()

        grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()
        self.scheduler.step()

        trainer_metrics = {}
        if grad_norm is not None:
            trainer_metrics["grad_norm"] = grad_norm
        trainer_metrics.update({"lr": lr, "loss": loss_mean})

        num_samples = global_batch["chosen"].shape[0]
        num_bad_samples = global_batch["bad_samples"].sum()
        num_bad_ends = global_batch["bad_ends"].sum()
        gen_lengths_chosen = (global_batch["chosen_gen_lens"] - global_batch["chosen_prompt_lens"]).sum()
        gen_lengths_reject = (global_batch["reject_gen_lens"] - global_batch["reject_prompt_lens"]).sum()
        sum_chosen_rewards = global_batch["chosen_rewards"][global_batch["chosen_rewards"] != -1].sum()
        sum_reject_rewards = global_batch["rejected_rewards"][global_batch["rejected_rewards"] != -1].sum()
        tensor_to_accumulate = torch.tensor(
            [
                gen_lengths_chosen,
                gen_lengths_reject,
                num_bad_samples,
                num_bad_ends,
                num_samples,
                sum_chosen_rewards,
                sum_reject_rewards,
            ],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group())

        (
            global_chosen_response_lengths,
            global_reject_response_lengths,
            GBS_sum_bad_samples,
            GBS_sum_bad_ends,
            GBS_num_samples,
            global_chosen_rewards,
            global_reject_rewards,
        ) = tensor_to_accumulate.tolist()
        metrics["chosen_lengths"] = global_chosen_response_lengths / GBS_num_samples
        metrics["reject_lengths"] = global_reject_response_lengths / GBS_num_samples
        metrics["bad_samples_per_GBS"] = GBS_sum_bad_samples / GBS_num_samples
        metrics["bad_ends_per_GBS"] = GBS_sum_bad_ends / (GBS_num_samples * self.num_responses_to_gen)
        metrics["chosen_generated_rewards"] = global_chosen_rewards / GBS_num_samples
        metrics["rejected_generated_rewards"] = global_reject_rewards / GBS_num_samples

        return loss_mean, {**metrics, **trainer_metrics}

    @torch.no_grad()
    def get_generations(self, list_of_batches):
        self.model.prepare_for_inference()
        if self.use_trtllm_generation:
            # at this point self.model is the reference policy from cpu_weight_swap
            self.trtllm_generate.refit(self.model)
            clear_memory()

        prompt_lengths = torch.cat([b["prompt_lengths"] for b in list_of_batches], dim=0)
        batch_max_length = prompt_lengths.max().item()
        max_possible_length = min(self.model.cfg.encoder_seq_length, batch_max_length + self.max_gen_seq_len)
        # in case the prompt length exceeds encoder_seq_length - max_gen_seq_len, we need to truncate how many
        # tokens we are allowed to generate such that we never exceed encoder_seq_length, otherwise you will get
        # errors inside model.generate()
        adj_generation_length = min(self.max_gen_seq_len, self.model.cfg.encoder_seq_length - batch_max_length)

        prompt_tokens = torch.cat(
            [
                batch_pad_to_fixed_len(b["prompts_only"], max_possible_length, pad_token=self.model.tokenizer.eos_id)
                for b in list_of_batches
            ],
            dim=0,
        )
        prompt_tokens = prompt_tokens.cuda(non_blocking=True)
        prompt_lengths = prompt_lengths.cuda(non_blocking=True)
        inputs = (prompt_tokens, prompt_lengths)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self.model, context_lengths=prompt_lengths, max_length=adj_generation_length
        )

        if self.use_trtllm_generation:
            generations = self.trtllm_generate.generate(inputs)
            response_tokens = generations["response_tokens"]
            response_lengths = generations["response_lengths"]
        else:
            generations = self.model.generate(
                inputs=inputs,
                length_params=self.length_params | {"max_length": adj_generation_length},
                sampling_params=self.sampling_params,
                strategy=strategy,
            )

            # this is a 1D LongTensor with the length of the responses where response is prompt+response
            response_tokens = torch.cuda.LongTensor(generations["token_ids"]) if generations else None
            response_tokens = broadcast_2d_tensor_within_pp(response_tokens, dtype=torch.long)
            response_lengths = strategy.get_lengths()

            max_response_length = response_lengths.max().item()

            # Sanity check to validate response length.
            if max_response_length != response_tokens.size(1):
                # This may actually happen because NeMo does not always stop generation after `max_length` in batch mode
                # => `response_tokens` may contain up to `max_length + max_context_length` tokens.
                # TODO once NeMo fixes this issue we should be able to always raise an exception when the check above fails,
                # and remove the `if` below.
                if (
                    max_response_length >= response_tokens.size(1)
                    or response_tokens.size(1) != batch_max_length + adj_generation_length
                ):
                    raise AssertionError(
                        f"max response length ({max_response_length}) does not match the size of "
                        f"`response_tokens` ({response_tokens.size(1)})"
                    )

        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.model.tokenizer, self.sampling_params["end_strings"]
        )

        self.model.finish_inference()
        if self.use_trtllm_generation:
            self.trtllm_generate.free()

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_valid.cpu()

    def get_rewards(self, list_of_batches):
        reward_scores = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in list_of_batches]))]
        judge_responses = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in list_of_batches]))]
        for _ in range(self.num_evals_to_average):
            reward_responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_batches)
            batch_responses_str = []
            for t, s, e in zip(reward_responses, prompt_lengths.tolist(), resp_lengths.tolist()):
                response = self.tokenizer.ids_to_text(t[s:e].tolist())
                batch_responses_str.append(response)
            rewards = [self.parse_reward_fn(resp_str) for resp_str in batch_responses_str]
            for idx, (r, t, s, e, end) in enumerate(
                zip(rewards, reward_responses, prompt_lengths.tolist(), resp_lengths.tolist(), is_end.tolist())
            ):
                # we can choose to invalidate scores where is_end==False, but there's really no need because so long as we get
                # a valid score, it's all good, we don't need correctness beyond that
                # reward_scores[idx].append(r if end else None)
                reward_scores[idx].append(
                    r if ((r is not None) and (r >= self.judge_score_low and r <= self.judge_score_high)) else None
                )
                # we may want to also check is_end here too, but we currently don't
                if self.use_meta_judge:
                    judge_responses[idx].append(
                        (t, s, e, end)
                        if ((r is not None) and (r >= self.judge_score_low and r <= self.judge_score_high))
                        else None
                    )

        assert all(
            [len(b) == self.num_evals_to_average for b in reward_scores]
        ), f"did not get generate the correct number of reward scores: {reward_scores}"
        reward_scores = [[*filter(exists, b)] for b in reward_scores]
        if self.use_meta_judge:
            assert all(
                [len(b) == self.num_evals_to_average for b in judge_responses]
            ), f"did not get generate the correct number of judge scores: {judge_responses}"
            judge_responses = [[*filter(exists, b)] for b in judge_responses]

        reward_means = [(np.mean(b) if len(b) > 0 else None) for b in reward_scores]
        reward_variance = [(np.var(b) if len(b) > 0 else None) for b in reward_scores]

        return reward_means, reward_variance, judge_responses

    def get_rewards_meta(self, list_of_batches):
        reward_scores = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in list_of_batches]))]
        reward_scores = []
        reward_responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_batches)
        batch_responses_str = []
        for t, s, e in zip(reward_responses, prompt_lengths.tolist(), resp_lengths.tolist()):
            response = self.tokenizer.ids_to_text(t[s:e].tolist())
            batch_responses_str.append(response)
        rewards = [self.meta_parse_reward_fn(resp_str) for resp_str in batch_responses_str]
        for idx, (r, end) in enumerate(zip(rewards, is_end.tolist())):
            # we can choose to invalidate scores where is_end==False, but there's really no need because so long as we get
            # a valid score, it's all good, we don't need correctness beyond that
            # reward_scores[idx].append(r if end else None)
            reward_scores.append(r if ((r is not None) and (r in ["A", "B"])) else None)

        return reward_scores

    def fit(self):
        if (not isinstance(self.train_dataloader.batch_sampler, MegatronPretrainingRandomBatchSampler)) and (
            self.cfg.max_epochs is not None and self.cfg.max_epochs > 1
        ):
            # if you use MegatronPretrainingBatchSampler as the batch_sampler passed to your train dataloader (in builders.py)
            # then each epoch will repeat all your samples in the same order as the previous epoch, there is no shuffling
            # to fix this, you should use MegatronPretrainingRandomBatchSampler instead, which alleviates this issue and allows
            # random shuffling for each epoch.
            raise ValueError(
                "max_epochs > 1 is not supported unless using `MegatronPretrainingRandomBatchSampler` as the batch_sampler for your train dataloader"
            )

        self.run_timer.start_time()

        iterations_iter = range(self.iteration, self.cfg.max_iterations)
        if len(iterations_iter) <= 0:
            # iteration done
            return

        for _ in iterations_iter:
            epoch_iter = range(self.epoch, self.cfg.max_epochs)
            if len(epoch_iter) <= 0:
                # epoch done
                return

            # call this in case the model is using a KL scheduler based on iteration number
            self.model.set_KL_penalty_by_iteration(self.iteration)
            # call this in case we are using a length_control scheduler based on iteration number
            self.set_rho_by_iteration(self.iteration)

            # print(f"*** Iteration [ {self.iteration} ]  RHO [ {self.rho} ] ***")

            for _ in epoch_iter:
                num_steps_in_epoch = min(
                    self.max_steps - self.step, self.num_steps_per_epoch - self.step % self.num_steps_per_epoch
                )
                loop_iter = range(num_steps_in_epoch)

                if not loop_iter:
                    return  # training ended

                global_pbar = tqdm(
                    self.augment_dataloader(self.train_dataloader),
                    initial=self.step,
                    total=self.max_steps,
                    leave=True,
                    desc="Training steps",
                )

                for _, global_batch in zip(loop_iter, global_pbar):
                    self.model.prepare_for_training()

                    self.timer.start("train_step_time")
                    if self.first_iteration_sft and self.iteration == 0:
                        loss, metrics = self.train_single_step_sft(global_batch)
                    else:
                        loss, metrics = self.train_single_step_dpo(global_batch)
                    self.timer.stop("train_step_time")
                    train_step_time = self.timer.get("train_step_time")
                    # to help avoid fragmentation
                    clear_memory()

                    # TODO(geshen): maybe use the dataloader instead
                    # bump up the consumed samples but not the step
                    self.consumed_samples += self.model.cfg.global_batch_size
                    metrics["consumed_samples"] = self.consumed_samples
                    metrics["step_time"] = train_step_time
                    metrics["epoch"] = self.epoch
                    metrics["iteration"] = self.iteration
                    self.logger.log_metrics(
                        metrics, step=self.step, prefix="train/",
                    )
                    metrics = {f"train_{k}": v for k, v in metrics.items()}

                    self.step += 1

                    run_time_exceeded = self.run_timer.is_finished()
                    run_val, save_model, is_train_end = check_progress(
                        self.step,
                        self.max_steps,
                        self.val_check_interval,
                        self.cfg.save_interval,
                        self.limit_val_batches,
                        run_time_exceeded=run_time_exceeded,
                    )

                    if run_val:
                        val_loss, val_metrics = self.run_validation()
                        # validation is done on the UPDATED weights
                        # so we use the incremented self.step
                        self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")
                        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                        metrics.update(val_metrics)

                        # we update the pandas table here only during validation to avoid blowing up wandb storage space
                        # we update only for rank 0 although this is redudant because .log_table() only works on rank 0
                        if (not (self.first_iteration_sft and self.iteration == 0)) and torch.distributed.get_rank() == 0:
                            for idx in range(len(global_batch["bad_samples"])):
                                if not global_batch["bad_samples"][idx]:
                                    self.train_df.loc[len(self.train_df)] = [
                                        self.step,
                                        self.tokenizer.ids_to_text(
                                            global_batch["chosen"][idx][
                                                : global_batch["chosen_prompt_lens"][idx].item()
                                            ].tolist()
                                        ),
                                        self.tokenizer.ids_to_text(
                                            global_batch["chosen"][idx][
                                                global_batch["chosen_prompt_lens"][idx]
                                                .item() : global_batch["chosen_gen_lens"][idx]
                                                .item()
                                            ].tolist()
                                        ),
                                        self.tokenizer.ids_to_text(
                                            global_batch["rejected"][idx][
                                                global_batch["reject_prompt_lens"][idx]
                                                .item() : global_batch["reject_gen_lens"][idx]
                                                .item()
                                            ].tolist()
                                        ),
                                    ]
                                    self.logger.log_table(
                                        key="table/train_generations", dataframe=self.train_df, step=self.step - 1,
                                    )
                                    break

                    global_pbar.set_postfix(metrics)

                    if save_model:
                        # PTL save wants tensors only
                        metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}
                        self.save(metrics, is_train_end=is_train_end)

                    if run_time_exceeded:
                        logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                        return

                    metrics.clear()
                    self.model.finish_training()

            # update the reference policy weights
            self.model.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(
                self.model, megatron_amp_O2=self.model.cfg.get("megatron_amp_O2", False)
            )

        self.logger.finalize()

        if self.use_trtllm_generation:
            self.trtllm_generate.free()

    def save(self, extra_candidates=None, is_train_end=False):
        # load back in the adam states if needed
        self.model.prepare_for_training()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        # we don't want to save the ref policy at the very end, although this prohibits continuation training from the .nemo file
        if is_train_end:
            self.model.ref_policy_state_dict = None

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs * self.cfg.max_iterations

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]

        loaded_values = [self.step, self.consumed_samples]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()
    
    def normalise_prompt(self, prompt, response):
        if self.cfg.trt_llm.get("model_type", "gptnext").lower() == "llama":
            p_list = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>user\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt)
            r_list = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt)
            resp_raw = response.replace(self.model.cfg.data.chat_prompt_tokens.end_of_turn, "").strip()
        else:
            p_list = re.findall(rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}User\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})", prompt)
            r_list = re.findall(rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}Assistant\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})", prompt)
            resp_raw = response.replace(f"\n{self.model.cfg.data.chat_prompt_tokens.turn_start}", "")
        if len(p_list) == 1 and len(r_list) == 0:
            return "User: " + p_list[0], resp_raw
        elif len(p_list) == len(r_list) + 1:
            comp = "User: " + p_list[0]
            for p, r in zip(p_list[1:], r_list):
                comp += "\n\nAssistant: " + r
                comp += "\n\nUser: " + p
            return comp, resp_raw
        else:
            raise RuntimeError(f"Received strange normalise payload PROMPT [ {prompt} ]  RESP [ {response} ]")

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations and ref policy log probs"""
        iter_dataloader = iter(dataloader)
        buffer = []
        meta_buffer_pending, meta_buffer_done = [], []
        done = False
        cnt_tracker = np.array([1 for _ in range(self.judge_score_high + 1)])
        while not done:
            try:
                batches = next(iter_dataloader)
                if self.first_iteration_sft and self.iteration == 0:
                    batch = self.train_dataloader.dataset.collate_fn(batches)
                else:
                    batch = self_rewarding_custom_collate(batches, eos_id=self.model.tokenizer.eos_id)
            except StopIteration:
                done = True
            else:
                buffer.append(batch)

            if self.first_iteration_sft and self.iteration == 0:
                for batch in buffer:
                    yield batch
                buffer.clear()
            elif (done and buffer) or sum(
                [len(b["prompts_and_answers"]) for b in buffer]
            ) == self.rollout_micro_batch_size:
                # generations use the reference model weights, as per the paper
                with cpu_weight_swap(
                    self.model, self.model.ref_policy_state_dict, megatron_amp_O2=self.model.megatron_amp_O2
                ):
                    candidate_responses_with_rewards = [
                        [] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))
                    ]
                    for _ in range(self.num_responses_to_gen):
                        # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                        gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(buffer)

                        # Transform into batch of LLM-as-judge template samples for reward scoring
                        reward_buffer = []
                        for t, s, e in zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist()):
                            '''
                            if self.cfg.trt_llm.get("model_type", "gptnext").lower() == "llama":
                                prompt = self.tokenizer.ids_to_text(t[:s].tolist()).replace(
                                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", ""
                                )
                                response = (
                                    self.tokenizer.ids_to_text(t[s:e].tolist()).replace("<|eot_id|>", "").strip()
                                )
                            else:
                                prompt = self.tokenizer.ids_to_text(t[:s].tolist()).replace(
                                    "<extra_id_0>System\n\n", ""
                                )
                                response = self.tokenizer.ids_to_text(t[s:e].tolist()).replace("\n<extra_id_1>", "")
                            '''
                            prompt, response = self.normalise_prompt(self.tokenizer.ids_to_text(t[:s].tolist()), self.tokenizer.ids_to_text(t[s:e].tolist()))
                            reward_prompt_str = self.template_fn(prompt=prompt, response=response)
                            reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                            if len(reward_prompt) > self.model.cfg.data.train_ds.max_seq_length:
                                prompt_and_response = self.tokenizer.ids_to_text(t[:e].tolist())
                                try:
                                    if self.cfg.trt_llm.get("model_type", "gptnext").lower() == "llama":
                                        prompt_ft = re.findall(
                                            r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>user\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)",
                                            prompt_and_response,
                                        )[0]
                                        response_ft = re.findall(
                                            r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)",
                                            prompt_and_response,
                                        )[0]
                                    else:
                                        prompt_ft = re.findall(
                                            rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}User\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})", prompt_and_response
                                        )[0]
                                        response_ft = re.findall(
                                            rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}Assistant\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})",
                                            prompt_and_response,
                                        )[0]
                                    # llama3
                                    # prompt_ft = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>user\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt_and_response)[0]
                                    # response_ft = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt_and_response)[0]
                                    reward_prompt_str = self.template_fn(prompt=prompt_ft, response=response_ft)
                                    reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)

                                    while len(reward_prompt) > (
                                        self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                    ):
                                        overage = len(reward_prompt) - self.model.cfg.data.train_ds.max_seq_length
                                        if overage > len(self.model.tokenizer.text_to_ids(response_ft)):
                                            print(f"*** OVERAGE_NOT_FIT_RESPONSE: {reward_prompt_str}")
                                            reward_prompt_str = self.template_fn(
                                                prompt="How does one make tea?", response="I have no answer at all."
                                            )
                                            reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                                            break
                                        response_ft = self.tokenizer.ids_to_text(
                                            self.model.tokenizer.text_to_ids(response_ft)[:-overage]
                                        )
                                        reward_prompt_str = self.template_fn(prompt=prompt_ft, response=response_ft)
                                        reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                                except:
                                    print(f"*** TOO_LONG: {prompt_and_response}")
                                    # overage = len(reward_prompt) - (self.model.cfg.encoder_seq_length - self.max_gen_seq_len)
                                    while len(reward_prompt) > (
                                        self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                    ):
                                        overage = len(reward_prompt) - self.model.cfg.data.train_ds.max_seq_length
                                        if len(self.model.tokenizer.text_to_ids(response)) >= overage:
                                            # truncate response only
                                            response = self.tokenizer.ids_to_text(
                                                self.model.tokenizer.text_to_ids(response)[:-overage]
                                            )
                                            reward_prompt_str = self.template_fn(prompt=prompt, response=response)
                                            reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                                        else:
                                            # truncate response and prompt *SHOULD NEVER HAPPEN*
                                            print("*** PROMPT_AND_RESPONSE_NEED_TRUNCATION")
                                            reward_prompt_str = self.template_fn(
                                                prompt="How does one make tea?", response="I have no answer at all."
                                            )
                                            reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                                            break
                                assert len(reward_prompt) <= (
                                    self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                ), f"truncation of response only failed [ {len(reward_prompt)} ]: {reward_prompt_str}"

                            reward_buffer.append(
                                {
                                    "prompt_lengths": torch.LongTensor([len(reward_prompt)]),
                                    "prompts_only": torch.LongTensor(reward_prompt).unsqueeze(0),
                                }
                            )

                        # list of floats, same length as gen_tokens_buf
                        reward_scores, reward_variances, judge_responses = self.get_rewards(reward_buffer)
                        for idx, (t, s, e, r, v, j, end) in enumerate(
                            zip(
                                gen_tokens_buf,
                                gen_prompt_lengths_buf.tolist(),
                                gen_lengths_buf.tolist(),
                                reward_scores,
                                reward_variances,
                                judge_responses,
                                is_end.tolist(),
                            )
                        ):
                            candidate_responses_with_rewards[idx].append((r, t, s, e, v, j, end))

                    final_buffer = []
                    # now we need to pick the chosen/rejected
                    for cand_list in candidate_responses_with_rewards:
                        scores = [b[0] for b in cand_list]
                        ends = [b[-1] for b in cand_list]
                        resp_lengths = [len(b[1][b[2] : b[3]]) for b in cand_list]
                        variances = [b[-3] for b in cand_list]
                        j_responses = [b[-2] for b in cand_list]
                        filtered_scores = [
                            (s, r, v, idx)
                            for idx, (s, r, v, e) in enumerate(zip(scores, resp_lengths, variances, ends))
                            if (s is not None) and e
                        ]
                        filtered_variances = [
                            (v, j, idx)
                            for idx, (v, j, e) in enumerate(zip(variances, j_responses, ends))
                            if (v is not None) and (v > 0) and (len(j) > 1) and e
                        ]
                        bad_sample = False

                        # if all scores are identical (even all None) we just randomly choose
                        if len(filtered_scores) <= 1 or all([filtered_scores[0][0] == s[0] for s in filtered_scores]):
                            idx_chosen, idx_reject = np.random.choice(len(scores), size=2, replace=False)
                            bad_sample = True
                            # if len(filtered_scores) <= 1:
                            #    print("BAD_SAMPLE_1")
                            # elif all([filtered_scores[0][0] == s[0] for s in filtered_scores]):
                            #    print("BAD_SAMPLE_2")
                        elif len(filtered_scores) > 1:
                            # idx_chosen = filtered_scores[np.argmax([s[0] for s in filtered_scores])][-1]
                            # idx_reject = filtered_scores[np.argmin([s[0] for s in filtered_scores])][-1]
                            s_min = np.min([s[0] for s in filtered_scores])
                            s_max = np.max([s[0] for s in filtered_scores])
                            rng_chosen = [((1.0 - self.rho) * s_max) + (self.rho * s_min), s_max]
                            rng_reject = [s_min, ((1.0 - self.rho) * s_min) + (self.rho * s_max)]
                            chosen_cands = [
                                s for s in filtered_scores if s[0] >= rng_chosen[0] and s[0] <= rng_chosen[1]
                            ]
                            reject_cands = [
                                s for s in filtered_scores if s[0] >= rng_reject[0] and s[0] <= rng_reject[1]
                            ]
                            if self.rho > 0:
                                # choose based on shortest/longest response length
                                idx_chosen = chosen_cands[np.argmin([s[1] for s in chosen_cands])][-1]
                                idx_reject = reject_cands[np.argmax([s[1] for s in reject_cands])][-1]
                            else:
                                assert self.rho == 0
                                # choose based on lowest variance of judgements
                                idx_chosen = chosen_cands[np.argmin([s[2] for s in chosen_cands])][-1]
                                idx_reject = reject_cands[np.argmin([s[2] for s in reject_cands])][-1]
                            #if self.rho == 0:
                            #    assert all([s_max == s[0] for s in chosen_cands]), "chosen_cands violation"
                            #    assert all([s_min == s[0] for s in reject_cands]), "reject_cands violation"
                        else:
                            logging.error(f"*** final_scores [ {scores} ]  final_filtered_scores [ {filtered_scores} ]")
                            raise RuntimeError("hit strange score selection state, please investigate")

                        # 1 x max_len tensor
                        chosen_prompt_len = cand_list[idx_chosen][2]
                        chosen_gen_len = cand_list[idx_chosen][3]
                        chosen_tokens = cand_list[idx_chosen][1][:chosen_gen_len]
                        chosen_score = scores[idx_chosen]
                        reject_prompt_len = cand_list[idx_reject][2]
                        reject_gen_len = cand_list[idx_reject][3]
                        reject_tokens = cand_list[idx_reject][1][:reject_gen_len]
                        reject_score = scores[idx_reject]
                        bad_ends = sum(~np.array([cand_list[idx_chosen][-1], cand_list[idx_reject][-1]]))

                        if torch.equal(chosen_tokens, reject_tokens):
                            bad_sample = True
                            # print("BAD_SAMPLE_3")

                        # meta-judge logic goes here
                        if self.use_meta_judge and len(filtered_variances) > 0:
                            highest_variance_idx = np.argmax([s[0] for s in filtered_variances])
                            reward_tokens_raw = filtered_variances[highest_variance_idx][1]
                            idx_for_cand = filtered_variances[highest_variance_idx][-1]
                            cand_for_meta = cand_list[idx_for_cand]
                            orig_prompt_str = self.tokenizer.ids_to_text(cand_for_meta[1][: cand_for_meta[2]].tolist())
                            orig_response_str = self.tokenizer.ids_to_text(
                                cand_for_meta[1][cand_for_meta[2] : cand_for_meta[3]].tolist()
                            )
                            meta_batch = []
                            for a, b in itertools.combinations(
                                [self.tokenizer.ids_to_text(s[0][s[1] : s[2]].tolist()) for s in reward_tokens_raw], 2
                            ):
                                score_a = self.parse_reward_fn(a)
                                score_b = self.parse_reward_fn(b)
                                if score_a is None or score_b is None or a == b:
                                    continue
                                # we remove the actual scores here because we want the judge to judge purely on the
                                # CoT/explanation and not based on the numerical scores
                                a = re.sub("(?i)(?:Score|Points): ([0-9\.]+)", "", a)
                                b = re.sub("(?i)(?:Score|Points): ([0-9\.]+)", "", b)
                                meta_str_ab = self.meta_judge_template_fn(
                                    prompt=orig_prompt_str, response=orig_response_str, judgement_a=a, judgement_b=b
                                )
                                meta_str_ba = self.meta_judge_template_fn(
                                    prompt=orig_prompt_str, response=orig_response_str, judgement_a=b, judgement_b=a
                                )
                                meta_tokens_ab = self.model.tokenizer.text_to_ids(meta_str_ab)
                                meta_tokens_ba = self.model.tokenizer.text_to_ids(meta_str_ba)
                                # check for seq len violation
                                if (
                                    len(meta_tokens_ab) > self.model.cfg.data.train_ds.max_seq_length
                                    or len(meta_tokens_ba) > self.model.cfg.data.train_ds.max_seq_length
                                ):
                                    continue
                                meta_batch.append(
                                    {
                                        "prompt_lengths": torch.LongTensor([len(meta_tokens_ab)]),
                                        "prompts_only": torch.LongTensor(meta_tokens_ab).unsqueeze(0),
                                    }
                                )
                                meta_batch.append(
                                    {
                                        "prompt_lengths": torch.LongTensor([len(meta_tokens_ba)]),
                                        "prompts_only": torch.LongTensor(meta_tokens_ba).unsqueeze(0),
                                    }
                                )
                            # we keep the meta_buffer_done at no more than GBS * 3 to avoid using too much memory
                            # GBS * 3 should be more than enough of a buffer size to ensure we have sufficient samples to draw from
                            if meta_batch and len(meta_buffer_done) < self.model.cfg.global_batch_size * 3:
                                meta_buffer_pending.append((reward_tokens_raw, meta_batch))

                        # due to DP sync issues, we cannot dynamically increase/decrease samples in the local DP batch
                        # so the only thing we can do is replace/modify existing samples. Hence, at the moment, we only
                        # replace the bad samples in each DP batch with meta-judge samples. This means that the true amount
                        # of meta juddge samples will be between 0 and up to meta_judge_pcnt, so the meta_judge_pcnt param
                        # is really an upper bound, not the exact replacement %. This can be easily altered though.
                        if (
                            self.use_meta_judge
                            and ((bad_ends > 0 or bad_sample) and (torch.rand((1,)) <= self.meta_judge_pcnt))
                            and len(meta_buffer_done) > 0
                        ):
                            # if self.use_meta_judge and (bad_ends > 0 or bad_sample) and len(meta_buffer_done) > 0:
                            final_buffer.append(meta_buffer_done.pop(0))
                            # if you want to pop a random element instead, uncomment the below
                            # final_buffer.append(meta_buffer_done.pop(torch.randint(0, len(meta_buffer_done), (1,)).item()))
                        else:
                            final_buffer.append(
                                {
                                    "chosen_tokens": chosen_tokens,
                                    "chosen_prompt_len": chosen_prompt_len,
                                    "chosen_gen_len": chosen_gen_len,
                                    "chosen_score": chosen_score,
                                    "reject_tokens": reject_tokens,
                                    "reject_prompt_len": reject_prompt_len,
                                    "reject_gen_len": reject_gen_len,
                                    "reject_score": reject_score,
                                    "bad_sample": bad_sample,
                                    "bad_ends": bad_ends,
                                }
                            )

                original_gbs_size = len(buffer[0]["prompt_lengths"])
                for batch in divide_chunks(final_buffer, original_gbs_size):
                    chosen_prompt_lens = torch.LongTensor([b["chosen_prompt_len"] for b in batch])
                    chosen_gen_lens = torch.LongTensor([b["chosen_gen_len"] for b in batch])
                    chosen_scores = torch.FloatTensor(
                        [(0 if b["chosen_score"] is None else b["chosen_score"]) for b in batch]
                    )
                    reject_prompt_lens = torch.LongTensor([b["reject_prompt_len"] for b in batch])
                    reject_gen_lens = torch.LongTensor([b["reject_gen_len"] for b in batch])
                    reject_scores = torch.FloatTensor(
                        [(0 if b["reject_score"] is None else b["reject_score"]) for b in batch]
                    )
                    bad_samples = torch.BoolTensor([b["bad_sample"] for b in batch])

                    max_batch_len = max(
                        [len(b["chosen_tokens"]) for b in batch] + [len(b["reject_tokens"]) for b in batch]
                    )

                    """
                    chosen_tokens_pad = torch.cat(
                        [
                            batch_pad_to_fixed_len(b["chosen_tokens"].unsqueeze(0), max_batch_len, pad_token=self.model.tokenizer.eos_id)
                            for b in batch
                        ],
                        dim=0,
                    )
                    reject_tokens_pad = torch.cat(
                        [
                            batch_pad_to_fixed_len(b["reject_tokens"].unsqueeze(0), max_batch_len, pad_token=self.model.tokenizer.eos_id)
                            for b in batch
                        ],
                        dim=0,
                    )
                    """
                    # only works without the outer wrapping because it's a 1D tensor instead of 2D
                    chosen_tokens_pad = batch_pad_to_fixed_len(
                        [b["chosen_tokens"] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id
                    )
                    reject_tokens_pad = batch_pad_to_fixed_len(
                        [b["reject_tokens"] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id
                    )

                    chosen_mask = create_mask(
                        chosen_tokens_pad, chosen_prompt_lens, chosen_gen_lens
                    ) * ~bad_samples.unsqueeze(-1)
                    reject_mask = create_mask(
                        reject_tokens_pad, reject_prompt_lens, reject_gen_lens
                    ) * ~bad_samples.unsqueeze(-1)

                    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                        chosen_tokens_pad,
                        self.model.tokenizer.eos_id,
                        self.model.cfg.data.reset_position_ids,
                        self.model.cfg.data.reset_attention_mask,
                        self.model.cfg.data.eod_mask_loss,
                    )
                    assert attention_mask.ndim == 4, "attention_mask is incorrect shape"
                    if attention_mask.shape[0] == 1:
                        # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
                        # attention_mask = attention_mask.expand(len(act_tokens_pad), *((-1,) * (len(attention_mask.shape) - 1)))
                        attention_mask = attention_mask.repeat(
                            len(chosen_tokens_pad), *((1,) * (len(attention_mask.shape) - 1))
                        )

                    new_batch = {
                        "chosen": chosen_tokens_pad,
                        "rejected": reject_tokens_pad,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "chosen_mask": chosen_mask,
                        "rejected_mask": reject_mask,
                        "chosen_rewards": chosen_scores,
                        "rejected_rewards": reject_scores,
                        "chosen_prompt_lens": chosen_prompt_lens,
                        "reject_prompt_lens": reject_prompt_lens,
                        "chosen_gen_lens": chosen_gen_lens,
                        "reject_gen_lens": reject_gen_lens,
                        "bad_samples": bad_samples,
                        "bad_ends": torch.IntTensor([b["bad_ends"] for b in batch]),
                    }

                    assert (
                        chosen_gen_lens - chosen_prompt_lens >= 0
                    ).all(), "negative generated length encountered in chosen"
                    assert (
                        reject_gen_lens - reject_prompt_lens >= 0
                    ).all(), "negative generated length encountered in rejected"

                    # NB: this could be optimized by computing log probs earlier while the reference policy was still loaded.
                    logprobs = self.model.get_ref_policy_logprobs(new_batch).cpu()
                    chosen_logps, reject_logps = torch.split(logprobs, len(logprobs) // 2, dim=0)

                    new_batch["ref_policy_log_probs_chosen"] = chosen_logps
                    new_batch["ref_policy_log_probs_rejected"] = reject_logps

                    yield new_batch
                    del logprobs, chosen_logps, reject_logps, new_batch

                buffer.clear()

                # print(f"*** Rank [ {torch.distributed.get_rank()} ] Iteration [ {self.iteration} ] Step [ {self.step} ] META_BATCH_PENDING [ {len(meta_buffer_pending)} ] META_BATCH_ROLLOUT [ {sum([len(x[-1]) for x in meta_buffer_pending])} ] META_BATCH_DONE [ {len(meta_buffer_done)} ]")
                # print(f"*** Rank [ {torch.distributed.get_rank()} ] Iteration [ {self.iteration} ] Step [ {self.step} ] META_CNTR  {cnt_tracker}  META_CNTR_PCNT  {cnt_tracker / sum(cnt_tracker).clip(min=1.0)}")
                if done:
                    meta_buffer_pending.clear()
                    # meta_buffer_done.clear()
                if (
                    self.use_meta_judge
                    and (not done)
                    and (rollout_len := sum([len(x[-1]) for x in meta_buffer_pending]))
                    >= self.rollout_micro_batch_size
                ):
                    num_rollouts = rollout_len // self.rollout_micro_batch_size
                    meta_buffer_unroll_grp = [(idx, y) for idx, x in enumerate(meta_buffer_pending) for y in x[-1]]
                    for _ in range(num_rollouts):
                        meta_buffer_unroll = [
                            meta_buffer_unroll_grp.pop(0) for _ in range(self.rollout_micro_batch_size)
                        ]
                        meta_reward_scores = self.get_rewards_meta([x[-1] for x in meta_buffer_unroll])
                        meta_pairs = []
                        reroll = [
                            (grp, [y[-1] for y in x])
                            for grp, x in itertools.groupby(meta_buffer_unroll, lambda kk: kk[0])
                        ]
                        for tup in reroll:
                            N = len(tup[-1])
                            bad_meta_sample = False
                            # list of tuples of (t,s,e,end), one tuple per self.num_evals_to_average
                            # we need to find a chosen and reject index in this list
                            reward_tokens_raw = meta_buffer_pending[tup[0]][0]
                            p = len(reward_tokens_raw)

                            elo_scores = self.get_elo_scores(p, N, meta_reward_scores)
                            if len(np.unique(elo_scores)) < p:
                                bad_meta_sample = True
                                # print("BAD_META_SAMPLE_1")

                            meta_chosen_idx = np.argmax(elo_scores)
                            meta_reject_idx = np.argmin(elo_scores)

                            chosen_prompt_len = reward_tokens_raw[meta_chosen_idx][1]
                            chosen_gen_len = reward_tokens_raw[meta_chosen_idx][2]
                            chosen_tokens = reward_tokens_raw[meta_chosen_idx][0][:chosen_gen_len]
                            reject_prompt_len = reward_tokens_raw[meta_reject_idx][1]
                            reject_gen_len = reward_tokens_raw[meta_reject_idx][2]
                            reject_tokens = reward_tokens_raw[meta_reject_idx][0][:reject_gen_len]
                            meta_bad_ends = sum(
                                ~np.array(
                                    [reward_tokens_raw[meta_chosen_idx][-1], reward_tokens_raw[meta_reject_idx][-1]]
                                )
                            )

                            if torch.equal(chosen_tokens, reject_tokens):
                                bad_meta_sample = True
                                # print("BAD_META_SAMPLE_2")

                            chosen_score = self.parse_reward_fn(
                                self.tokenizer.ids_to_text(chosen_tokens[chosen_prompt_len:chosen_gen_len].tolist())
                            )
                            reject_score = self.parse_reward_fn(
                                self.tokenizer.ids_to_text(reject_tokens[reject_prompt_len:reject_gen_len].tolist())
                            )
                            # print(f"*** Iteration [ {self.iteration} ] Step [ {self.step} ] META_ACTUAL_REWARDS  CHOSEN[ {chosen_score} ]  REJECT[ {reject_score} ]")
                            if chosen_score is None or reject_score is None or chosen_score == reject_score:
                                bad_meta_sample = True
                                # print("BAD_META_SAMPLE_3")

                            if (
                                meta_bad_ends == 0
                                and not bad_meta_sample
                                and (
                                    (cnt_tracker / sum(cnt_tracker).clip(min=1.0))[int(chosen_score)]
                                    < self.meta_max_relative_pcnt
                                )
                                and (
                                    cnt_tracker[int(chosen_score)]
                                    < int(
                                        self.num_steps_per_epoch
                                        * original_gbs_size
                                        / ((self.judge_score_high - self.judge_score_low) ** 2)
                                    )
                                )
                            ):
                                meta_pairs.append(
                                    {
                                        "chosen_tokens": chosen_tokens,
                                        "chosen_prompt_len": chosen_prompt_len,
                                        "chosen_gen_len": chosen_gen_len,
                                        "chosen_score": chosen_score,
                                        "reject_tokens": reject_tokens,
                                        "reject_prompt_len": reject_prompt_len,
                                        "reject_gen_len": reject_gen_len,
                                        "reject_score": reject_score,
                                        "bad_sample": bad_meta_sample,
                                        "bad_ends": meta_bad_ends,
                                    }
                                )
                                cnt_tracker[int(chosen_score)] += 1

                            if N <= len(meta_buffer_pending[tup[0]][-1]):
                                [meta_buffer_pending[tup[0]][-1].pop(0) for _ in range(N)]
                            else:
                                raise RuntimeError(
                                    f"{N=} should never be greater than buffer [ {meta_buffer_pending[tup[0]]} ]"
                                )

                        meta_buffer_done.extend(meta_pairs)

                    del meta_buffer_unroll_grp
                    meta_buffer_pending = [x for x in meta_buffer_pending if len(x[-1]) > 0]

    def set_rho_by_iteration(self, iteration):
        if isinstance(self.spin_config["length_control"], (float, int)):
            return
        elif isinstance(self.spin_config["length_control"], list):
            assert iteration < len(
                self.spin_config["length_control"]
            ), f"iteration [ {iteration} ] is out of bounds for length_control schedule {self.spin_config['length_control']}"

            self.rho = self.spin_config["length_control"][iteration]

    def get_elo_scores(self, p, N, meta_reward_scores):
        players = list(range(p))
        Bm = itertools.combinations(players, 2)
        alloc = []
        for _ in range(N):
            alloc.append(meta_reward_scores.pop(0))
        assert len(alloc) % 2 == 0, "alloc should always be divisible by 2"
        ptbl_a_win = np.zeros([p, p])
        ptbl_b_win = np.zeros([p, p])
        ptbl_tie = np.zeros([p, p])
        for (m_a, m_b), (ab, ba) in zip(Bm, divide_chunks(alloc, 2)):
            if ab is not None and ba is not None:
                ptbl_a_win[m_a, m_b] += int(ab == "A" and ba == "B")
                ptbl_b_win[m_a, m_b] += int(ab == "B" and ba == "A")
                ptbl_tie[m_a, m_b] += int(ab == ba)

        ptbl_win = ptbl_a_win * 1 + ptbl_b_win.T * 1 + (ptbl_tie + ptbl_tie.T)

        X = np.zeros([p * (p - 1) * 2, p])
        Y = np.zeros(p * (p - 1) * 2)
        # w1 = ptbl_b_win.sum() / (ptbl_a_win.sum() + ptbl_b_win.sum())
        # w2 = ptbl_a_win.sum() / (ptbl_a_win.sum() + ptbl_b_win.sum())
        cur_row = 0
        sample_weights = []
        for m_a in players:
            for m_b in players:
                if m_a == m_b:
                    continue
                # if nan skip
                if math.isnan(ptbl_win[m_a, m_b]) or math.isnan(ptbl_win[m_b, m_a]):
                    continue
                X[cur_row, players[m_a]] = 1.0
                X[cur_row, players[m_b]] = -1.0
                Y[cur_row] = 1.0
                sample_weights.append(ptbl_win[m_a, m_b])
                # sample_weights.append(w1 * (1 if ptbl_a_win[m_a, m_b] >= 1 else 0) + w2 * (1 if ptbl_b_win[m_a, m_b] >= 1 else 0))

                X[cur_row + 1, players[m_a]] = 1.0
                X[cur_row + 1, players[m_b]] = -1.0
                Y[cur_row + 1] = 0.0
                sample_weights.append(ptbl_win[m_b, m_a])
                # sample_weights.append(w1 * (1 if ptbl_a_win[m_b, m_a] >= 1 else 0) + w2 * (1 if ptbl_b_win[m_b, m_a] >= 1 else 0))
                cur_row += 2
        X = X[:cur_row]
        Y = Y[:cur_row]

        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
        lr.fit(X, Y, sample_weight=sample_weights)

        elo_scores = SCALE * lr.coef_[0] + INIT_RATING

        return elo_scores

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs

    @property
    def iteration(self):
        return (self.step // self.num_steps_per_epoch) // self.cfg.max_epochs
