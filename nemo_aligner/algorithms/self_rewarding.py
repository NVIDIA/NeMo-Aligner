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

import os, json
from functools import partial
import numpy as np
from collections import defaultdict
from statistics import mean
import pandas as pd
from textwrap import dedent

import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import SyncTimer, gather_tensor
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.text_generation_utils import TrackLengthGPTModelTextGenerationStrategy
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import (
    batch_pad_to_fixed_len,
    clear_memory,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)


try:
    from tensorrt_llm.bindings import GptSession

    from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM

    GptSession.refit_engine  # check if TRTLLM Cpp runtime was compiled with engine refitting
    HAVE_TRTLLM = True
except (ImportError, ModuleNotFoundError):
    HAVE_TRTLLM = False


"""
GPTSFTChatDataset output is dict with keys: ['input_ids', 'mask', 'context_ids', 'answer_ids', 'metadata']

input_ids: torch.LongTensor - the entire prompt + response, including the system preamble which is specified by "system" in the jsonl
mask: torch.BoolTensor with False for the preamble+prompt, and True for the response
context_ids: torch.LongTensor - the entire preamble + prompt
answer_ids: torch.LongTensor - the entire response only
metadata: dict - with keys "system" for the preamble, and "mask" which is "User" or "Assistant"
"""
def self_rewarding_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
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

def eye(x):
    return x

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def find_variables_from_jinja_template(template: str):
    from jinja2 import meta
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)

def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward = "([0-9\.]+)")

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

def divide_chunks(l, n):
    for i in range(0, len(l), n):  
        yield l[i:i + n]


DEFAULT_LLM_AS_JUDGE_PROMPT = """<extra_id_0>System

<extra_id_1>User
Review the user's question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective,
addressing the user's question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user's instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: <total points>"
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we'll
systematically attribute points based on the outlined criteria.
<extra_id_1>Assistant
"""

DEFAULT_REWARD_REGEX_TEMPLATE = "(?i)(?:Score|Points): {{ reward }}"


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
        exp_manager,
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
            self.train_dataloader.batch_sampler.total_samples = min(self.train_dataloader.batch_sampler.total_samples, self.cfg.limit_train_batches * self.train_dataloader.batch_sampler.global_batch_size)
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

        self.num_responses_to_gen = self.model.cfg.spin.num_responses_to_gen
        self.num_evals_to_average = self.model.cfg.spin.num_evals_to_average
        self.first_iteration_sft = self.model.cfg.spin.get("first_iteration_sft", False)
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
        
        # llm-as-judge prompt and reward
        #self.prompt_template = dedent(DEFAULT_LLM_AS_JUDGE_PROMPT)
        #self.reward_regex_template = dedent(DEFAULT_REWARD_REGEX_TEMPLATE)
        self.prompt_template = DEFAULT_LLM_AS_JUDGE_PROMPT
        self.reward_regex_template = DEFAULT_REWARD_REGEX_TEMPLATE
        
        assert find_variables_from_jinja_template(self.prompt_template) == {'prompt', 'response'}, 'template must include prompt and response templating variables'
        self.template_fn = jinja2_env.from_string(self.prompt_template).render
        
        self.parse_reward_fn = create_parse_reward_fn(self.reward_regex_template)
        
        # storage for generated responses which we want to save
        #if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
        #    self.generations_fh = open(os.path.join(exp_manager.explicit_log_dir, "generations.jsonl"), "a", encoding="utf_8", newline="\n")
        #else:
        #    self.generations_fh = None
        
        self.use_trtllm_generation = self.cfg.trt_llm.get("enable", False) if "trt_llm" in self.cfg else False
        if self.use_trtllm_generation:
            assert HAVE_TRTLLM, "TRTLLM generation was enabled but TRTLLM libraries could not be successfully imported"
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.model.cfg,
                max_generation_length=self.length_params["max_length"],
                max_input_len=self.cfg.trt_llm.get("max_input_len", 1024),
                max_input_tokens=self.cfg.trt_llm.get("max_input_tokens", 4096),
                generation_batch_size=self.model.cfg.spin.get("rollout_micro_batch_size", 4),
                unload_engine_train=self.cfg.trt_llm.get("unload_engine_train", False),
                trt_model_type=self.cfg.trt_llm.get("model_type", "llama"),
                end_strings=self.sampling_params["end_strings"],
                reshard_model=False,
                sample_temperature=self.sampling_params["temperature"],
                sample_top_k=self.sampling_params["top_k"],
                sample_top_p=self.sampling_params["top_p"],
                repetition_penalty=self.sampling_params["repetition_penalty"],
                use_greedy=self.sampling_params.get("use_greedy", False),
                tokenizer=self.model.tokenizer,
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
            #self.model.prepare_for_validation()

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

            #self.model.finish_validation()

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


        num_samples = 0
        gen_lengths_chosen = 0
        gen_lengths_reject = 0
        num_bad_samples = 0
        num_bad_ends = 0
        sum_chosen_rewards = 0
        sum_reject_rewards = 0
        num_samples += global_batch["actual"].shape[0]
        num_bad_samples += global_batch["bad_samples"].sum()
        num_bad_ends += global_batch["bad_ends"].sum()
        gen_lengths_chosen += (global_batch["chosen_gen_lens"] - global_batch["chosen_prompt_lens"]).sum()
        gen_lengths_reject += (global_batch["reject_gen_lens"] - global_batch["reject_prompt_lens"]).sum()
        sum_chosen_rewards += global_batch["chosen_rewards"][global_batch["chosen_rewards"] != -1].sum()
        sum_reject_rewards += global_batch["reject_rewards"][global_batch["reject_rewards"] != -1].sum()
        tensor_to_accumulate = torch.tensor(
            [gen_lengths_chosen, gen_lengths_reject, num_bad_samples, num_bad_ends, num_samples, sum_chosen_rewards, sum_reject_rewards], dtype=torch.float32, device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group())

        (global_chosen_response_lengths, global_reject_response_lengths, GBS_sum_bad_samples, GBS_sum_bad_ends, GBS_num_samples, global_chosen_rewards, global_reject_rewards,) = tensor_to_accumulate.tolist()
        metrics["avg_chosen_lengths"] = global_chosen_response_lengths / GBS_num_samples
        metrics["avg_reject_lengths"] = global_reject_response_lengths / GBS_num_samples
        metrics["avg_bad_samples_per_GBS"] = GBS_sum_bad_samples / GBS_num_samples
        metrics["avg_bad_ends_per_GBS"] = GBS_sum_bad_ends / (GBS_num_samples * self.num_responses_to_gen)
        metrics["avg_chosen_generated_rewards"] = global_chosen_rewards / GBS_num_samples
        metrics["avg_rejected_generated_rewards"] = global_reject_rewards / GBS_num_samples


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
            #torch.cuda.synchronize()
            actor_output = self.trtllm_generate.generate(inputs)
            #torch.cuda.synchronize()
            response_tokens = actor_output["response_tokens"]
            response_lengths = actor_output["response_lengths"]
            
            prev = response_tokens[torch.arange(response_tokens.size(0)), response_lengths - 1]
            # double check with nemo logic to make sure it ended
            is_end = strategy.end_of_generation_condition(
                response_tokens, prev, self.model.tokenizer.eos_id, self.sampling_params["end_strings"]
            )
            
            for idx in range(len(response_tokens)):
                if torch.min(response_tokens[idx]).item() < 0 or torch.max(response_tokens[idx]).item() >= self.model.tokenizer.vocab_size:
                    is_end[idx] = False
                    response_tokens[idx] = torch.clamp(response_tokens[idx], min=self.model.tokenizer.eos_id, max=self.model.tokenizer.vocab_size - 1)
        else:
            generations = self.model.generate(
                inputs=(prompt_tokens, prompt_lengths),
                length_params=self.length_params | {"max_length": adj_generation_length},
                sampling_params=self.sampling_params,
                strategy=strategy,
            )
    
            # this is a 1D LongTensor with the length of the responses where response is prompt+response
            response_lengths, is_end = strategy.get_lengths(return_is_end=True)
            max_response_length = response_lengths.max().item()
            response_tokens = torch.LongTensor(generations["token_ids"])
    
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

        self.model.finish_inference()
        if self.use_trtllm_generation:
            self.trtllm_generate.free()

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_end.cpu()
    
    def get_rewards(self, list_of_batches):
        reward_scores = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in list_of_batches]))]
        for _ in range(self.num_evals_to_average):
            reward_responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_batches)
            batch_responses_str = []
            for t,s,e in zip(reward_responses, prompt_lengths.tolist(), resp_lengths.tolist()):
                response = self.model.tokenizer.ids_to_text(t[s:e].tolist())
                batch_responses_str.append(response)
            #print("*** batch_responses_str_len: ", len(batch_responses_str))
            #print("*** sample_reward_response: ", batch_responses_str[0])
            rewards = [self.parse_reward_fn(resp_str) for resp_str in batch_responses_str]
            #print("*** rewards_after_parse: ", rewards)
            for idx, (r, end) in enumerate(zip(rewards, is_end.tolist())):
                #if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank() and r is None:
                #    print("*** none_reward_for_this_resp: ", batch_responses_str[idx])
                if r is not None and r > 10000:
                    print("*** high_score_response: ", batch_responses_str[idx])
                # we can choose to invalidate scores where is_end==False, but there's really no need because so long as we get
                # a valid score, it's all good, we don't need correctness beyond that
                #reward_scores[idx].append(r if end else None)
                reward_scores[idx].append(r)
        #print("*** reward_scores_get_rewards: ", reward_scores)
        assert all([len(b) == self.num_evals_to_average for b in reward_scores]), "did not get generate the correct number of reward scores"
        reward_scores = [[*filter(exists, b)] for b in reward_scores]
        
        reward_means = [(np.mean(b) if len(b) > 0 else None) for b in reward_scores]
        
        return reward_means

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
            
            #self.generated_responses.clear()

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

                        if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
                            for idx in range(len(global_batch["bad_samples"])):
                                if not global_batch["bad_samples"][idx]:
                                    self.train_df.loc[len(self.train_df)] = [
                                        self.step,
                                        self.model.tokenizer.ids_to_text(global_batch["actual"][idx][:global_batch["chosen_prompt_lens"][idx].item()].tolist()),
                                        self.model.tokenizer.ids_to_text(
                                            global_batch["actual"][idx][global_batch["chosen_prompt_lens"][idx].item():global_batch["chosen_gen_lens"][idx].item()].tolist()
                                        ),
                                        self.model.tokenizer.ids_to_text(
                                            global_batch["generated"][idx][global_batch["reject_prompt_lens"][idx].item():global_batch["reject_gen_lens"][idx].item()].tolist()
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
                    
                    '''
                    generations = batch_pad_to_fixed_len(global_batch["generated"], self.model.cfg.encoder_seq_length, pad_token=self.model.tokenizer.eos_id)
                    generations_list = gather_tensor(
                        generations, dst=parallel_state.get_data_parallel_src_rank(), group=parallel_state.get_data_parallel_group()
                    )
                    if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank() and generations_list is not None:
                        for t in generations_list:
                            for p in t:
                                payload = {"iteration": self.iteration, "epoch": self.epoch, "step": self.step - 1, "response": self.model.tokenizer.ids_to_text(p.long().tolist())}
                                self.generations_fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    '''


            # update the reference policy weights
            self.model.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(
                self.model, megatron_amp_O2=self.model.cfg.get("megatron_amp_O2", False)
            )

        self.logger.finalize()
        
        if self.use_trtllm_generation and hasattr(self.trtllm_generate.trt_llm_exporter.model_runner, "session"):
            #self.trtllm_generate.unload_engine_train = True
            del self.trtllm_generate.trt_llm_exporter.model_runner.session
        
        #if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
        #    self.generations_fh.close()

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

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations and ref policy log probs"""
        iter_dataloader = iter(dataloader)
        buffer = []
        done = False
        while not done:
            try:
                batches = next(iter_dataloader)
                if self.first_iteration_sft and self.iteration == 0:
                    batch = self.train_dataloader.dataset.collate_fn(batches)
                else:
                    batch = self_rewarding_custom_collate(batches,
                                                          eos_id=self.model.tokenizer.eos_id,
                                                          reset_position_ids=self.model.cfg.data.get("reset_position_ids", False),
                                                          reset_attention_mask=self.model.cfg.data.get("reset_attention_mask", False),
                                                          eod_mask_loss=self.model.cfg.data.get("eod_mask_loss", False),
                                                          )
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
                    candidate_responses_with_rewards = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))]
                    for _ in range(self.num_responses_to_gen):
                        # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                        gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(buffer)
                        
                        # Transform into batch of LLM-as-judge template samples for reward scoring
                        reward_buffer = []
                        for t,s,e in zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist()):
                            prompt = self.model.tokenizer.ids_to_text(t[:s].tolist())
                            response = self.model.tokenizer.ids_to_text(t[s:e].tolist())
                            reward_prompt_str = self.template_fn(prompt=prompt.replace("<extra_id_0>System\n\n", ""), response=response.replace("\n<extra_id_1>", ""))
                            reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                            if len(reward_prompt) > (self.model.cfg.encoder_seq_length - self.max_gen_seq_len):
                                prompt_and_response = self.model.tokenizer.ids_to_text(t.tolist())
                                reward_prompt_str = self.template_fn(prompt=re.findall(r"(?s)(?<=<extra_id_1>User\n).*?(?=\n<extra_id_1>)", prompt_and_response)[0], response=re.findall(r"(?s)(?<=<extra_id_1>Assistant\n).*?(?=\n<extra_id_1>)", prompt_and_response)[0])
                                reward_prompt = self.model.tokenizer.text_to_ids(reward_prompt_str)
                            reward_buffer.append({"prompt_lengths": torch.LongTensor([len(reward_prompt)]), "prompts_only": torch.LongTensor(reward_prompt).unsqueeze(0)})
                        
                        # list of floats, same length as gen_tokens_buf
                        reward_scores = self.get_rewards(reward_buffer)
                        #print("*** reward_scores: ", reward_scores)
                        for idx, (t, s, e, r, end) in enumerate(zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist(), reward_scores, is_end.tolist())):
                            candidate_responses_with_rewards[idx].append((r,t,s,e,end))
                    
                    final_buffer = []
                    # now we need to pick the chosen/rejected
                    for cand_list in candidate_responses_with_rewards:
                        scores = [b[0] for b in cand_list]
                        ends = [b[-1] for b in cand_list]
                        filtered_scores = [(s,idx) for idx, (s,e) in enumerate(zip(scores, ends)) if (s is not None) and e]
                        bad_sample = False
                        
                        # TODO: sample more from the underlying Dataset instead
                        # if we have just one non-None value, take it as the chosen, and randomly choose from the others for reject
                        #if len(filtered_scores) == 1:
                            #idx_chosen = np.where(np.array(scores) == filtered_scores[0])[0][0]
                            #idx_chosen = filtered_scores[0][-1]
                            #idx_reject = np.random.choice(list(set(range(len(scores))) - set([idx_chosen])), 1, replace=False).item()
                            #bad_sample = True
                        # if all scores are identical (even all None) we just randomly choose
                        if len(filtered_scores) <= 1 or all([filtered_scores[0][0] == s[0] for s in filtered_scores]):
                            idx_chosen, idx_reject = np.random.choice(len(scores), size=2, replace=False)
                            bad_sample = True
                        elif len(filtered_scores) > 1:
                            idx_chosen = filtered_scores[np.argmax([s[0] for s in filtered_scores])][-1]
                            idx_reject = filtered_scores[np.argmin([s[0] for s in filtered_scores])][-1]
                        else:
                            print(f"*** final_scores [ {scores} ]  final_filtered_scores [ {filtered_scores} ]")
                            raise RuntimeError("hit strange score selection state, please investigate")
                        
                        #if torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank() and bad_sample:
                        #    print(f"*** Scores [ {scores} ]  Ends [ {ends} ]  filtered_scores [ {filtered_scores} ] ***")
                            #for idx in range(len(cand_list)):
                            #    gen_text = self.model.tokenizer.ids_to_text(cand_list[idx][1][cand_list[idx][2]:cand_list[idx][3]].tolist())
                            #    print(f"*** cand_idx [ {idx} ]  cand_text: {gen_text}")
                        
                        # 1 x max_len tensor
                        chosen_tokens = cand_list[idx_chosen][1]
                        chosen_prompt_len = cand_list[idx_chosen][2]
                        chosen_gen_len = cand_list[idx_chosen][3]
                        chosen_score = scores[idx_chosen]
                        reject_tokens = cand_list[idx_reject][1]
                        reject_prompt_len = cand_list[idx_reject][2]
                        reject_gen_len = cand_list[idx_reject][3]
                        reject_score = scores[idx_reject]
                        
                        final_buffer.append({
                            "chosen_tokens": chosen_tokens,
                            "chosen_prompt_len": chosen_prompt_len,
                            "chosen_gen_len": chosen_gen_len,
                            "chosen_score": chosen_score,
                            "reject_tokens": reject_tokens,
                            "reject_prompt_len": reject_prompt_len,
                            "reject_gen_len": reject_gen_len,
                            "reject_score": reject_score,
                            "bad_sample": bad_sample,
                            "bad_ends": sum(~np.array(ends)),
                            }
                        )

                original_gbs_size = len(buffer[0]["prompt_lengths"])
                for batch in divide_chunks(final_buffer, original_gbs_size):
                    chosen_prompt_lens = torch.LongTensor([b["chosen_prompt_len"] for b in batch])
                    chosen_gen_lens = torch.LongTensor([b["chosen_gen_len"] for b in batch])
                    chosen_scores = torch.FloatTensor([(-1 if b["chosen_score"] is None else b["chosen_score"]) for b in batch])
                    reject_prompt_lens = torch.LongTensor([b["reject_prompt_len"] for b in batch])
                    reject_gen_lens = torch.LongTensor([b["reject_gen_len"] for b in batch])
                    reject_scores = torch.FloatTensor([(-1 if b["reject_score"] is None else b["reject_score"]) for b in batch])
                    bad_samples = torch.BoolTensor([b["bad_sample"] for b in batch])

                    max_batch_len = max([len(b["chosen_tokens"]) for b in batch] + [len(b["reject_tokens"]) for b in batch])
                    
                    '''
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
                    '''
                    # only works without the outer wrapping because it's a 1D tensor instead of 2D
                    chosen_tokens_pad = batch_pad_to_fixed_len([b["chosen_tokens"] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id)
                    reject_tokens_pad = batch_pad_to_fixed_len([b["reject_tokens"] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id)
                    #chosen_labels = batch_pad_to_fixed_len([torch.LongTensor(([-100] * b["chosen_prompt_len"]) + b["chosen_tokens"].tolist()[b["chosen_prompt_len"]:]) for b in batch], max_batch_len, pad_token=-100)
                    #reject_labels = batch_pad_to_fixed_len([torch.LongTensor(([-100] * b["reject_prompt_len"]) + b["reject_tokens"].tolist()[b["reject_prompt_len"]:]) for b in batch], max_batch_len, pad_token=-100)

                    chosen_mask = create_mask(chosen_tokens_pad, chosen_prompt_lens, chosen_gen_lens) * ~bad_samples.unsqueeze(-1)
                    reject_mask = create_mask(reject_tokens_pad, reject_prompt_lens, reject_gen_lens) * ~bad_samples.unsqueeze(-1)

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

                    '''
                    new_batch = {
                        "chosen": chosen_tokens_pad,
                        "rejected": reject_tokens_pad,
                        "chosen_length": chosen_gen_lens,
                        "rejected_length": reject_gen_lens,
                        "chosen_labels": chosen_labels,
                        "rejected_labels": reject_labels,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "chosen_rewards": chosen_scores,
                        "rejected_rewards": reject_scores,
                    }

                    logprobs = self.model.get_ref_policy_logprobs(new_batch).cpu()
                    chosen_logps, reject_logps = torch.split(logprobs, len(logprobs) // 2, dim=0)

                    new_batch["ref_policy_log_probs_chosen"] = chosen_logps
                    new_batch["ref_policy_log_probs_rejected"] = reject_logps
                    '''
                    
                    new_batch = {
                        "actual": chosen_tokens_pad,
                        "generated": reject_tokens_pad,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "actual_mask": chosen_mask,
                        "generated_mask": reject_mask,
                        "chosen_rewards": chosen_scores,
                        "reject_rewards": reject_scores,
                        "chosen_prompt_lens": chosen_prompt_lens,
                        "reject_prompt_lens": reject_prompt_lens,
                        "chosen_gen_lens": chosen_gen_lens,
                        "reject_gen_lens": reject_gen_lens,
                        "bad_samples": bad_samples,
                        "bad_ends": torch.IntTensor([b["bad_ends"] for b in batch]),
                    }

                    assert (chosen_gen_lens - chosen_prompt_lens >= 0).all(), "negative generated length encountered in chosen"
                    assert (reject_gen_lens - reject_prompt_lens >= 0).all(), "negative generated length encountered in rejected"

                    logprobs = self.model.get_ref_policy_logprobs(new_batch).cpu()
                    act_logps, gen_logps = torch.split(logprobs, len(logprobs) // 2, dim=0)

                    new_batch["ref_policy_log_probs_actual"] = act_logps
                    new_batch["ref_policy_log_probs_generated"] = gen_logps

                    yield new_batch
                    #del logprobs, chosen_logps, reject_logps, new_batch
                    del logprobs, act_logps, gen_logps, new_batch

                buffer.clear()

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs

    @property
    def iteration(self):
        return (self.step // self.num_steps_per_epoch) // self.cfg.max_epochs
