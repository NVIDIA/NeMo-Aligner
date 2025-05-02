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
from statistics import mean

import numpy as np
import pandas as pd
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.common.tokenizers import AutoTokenizer
#from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import SyncTimer, broadcast_tensor_within_pp
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import clip_gradients, set_eval, set_train
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import (
    batch_pad_to_fixed_len,
    clear_memory,
    configure_batch_sizes,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)

from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM


"""
GPTSFTChatDataset output is dict with keys: ['input_ids', 'mask', 'context_ids', 'answer_ids', 'metadata']

input_ids: torch.LongTensor - the entire prompt + response, including the system preamble which is specified by "system" in the jsonl
mask: torch.BoolTensor with False for the preamble+prompt, and True for the response
context_ids: torch.LongTensor - the entire preamble + prompt
answer_ids: torch.LongTensor - the entire response only
metadata: dict - with keys "system" for the preamble, and "mask" which is "User" or "Assistant"
"""


def self_taught_custom_collate(batch, eos_id):
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
        "dataset_mask": batch[0]['metadata']['mask'] if 'metadata' in batch[0] else "",
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
    from jinja2 import meta

    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_regex_fn(reward_regex_template):
    def parse_reward_fn(llm_response: str):
        result = list(set(re.findall(reward_regex_template, llm_response)))

        if result is None or len(result) == 0:
            return None
        elif len(result) > 1:
            print(f"*** REGEX_MORE_THAN_ONE [ {reward_regex_template} ] : {llm_response}")
            return None
        
        return result[0].strip().replace('"','').upper()

    return parse_reward_fn


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


DEFAULT_BAD_RESPONSE_PROMPT = """<extra_id_0>System

<extra_id_1>User
Below is a conversation between an user and an AI Assistant.

{{ orig_prompt }}

The start of Assistant's Answer
{{ orig_response }}
The end of Assistant's Answer

Please first generate a modified instruction that is highly relevant but not semantically identical to
the instruction above from the user. Then write a high-quality answer which is a good response to the
modified instruction but not a good response to the original user question. IMPORTANT: Please strictly
follow the following format:

User Question Modified
<provide a modified instruction here>

[assistant modified instruction start]
<provide a high-quality response to the modified instruction>
[assistant modified instruction end]
<extra_id_1>Assistant
"""


DEFAULT_BAD_RESPONSE_PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between an user and an AI Assistant.

{{ orig_prompt }}

The start of Assistant's Answer
{{ orig_response }}
The end of Assistant's Answer

Please first generate a modified instruction that is highly relevant but not semantically identical to
the instruction above from the user. Then write a high-quality answer which is a good response to the
modified instruction but not a good response to the original user question. IMPORTANT: Please strictly
follow the following format:

User Question Modified
<provide a modified instruction here>

[assistant modified instruction start]
<provide a high-quality response to the modified instruction>
[assistant modified instruction end]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


DEFAULT_JUDGEMENT_ANNOTATION = """<extra_id_0>System

<extra_id_1>User
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants
to the user question displayed below. You should choose the assistant that follows the user's instructions
and answers the user's question better. Your evaluation should consider factors such as the helpfulness,
relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by
comparing the two responses and provide a short explanation. Avoid any position biases and ensure that
the order in which the responses were presented does not influence your decision. Do not allow the length
of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective
as possible. After providing your explanation, output your final verdict by strictly following this format:
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[[User Question]]
{{ orig_prompt }}

[The Start of Assistant A's Answer]
{{ response_A }}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{ response_B }}
[The End of Assistant B's Answer]
<extra_id_1>Assistant
"""


DEFAULT_JUDGEMENT_ANNOTATION_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants
to the user question displayed below. You should choose the assistant that follows the user's instructions
and answers the user's question better. Your evaluation should consider factors such as the helpfulness,
relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by
comparing the two responses and provide a short explanation. Avoid any position biases and ensure that
the order in which the responses were presented does not influence your decision. Do not allow the length
of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective
as possible. After providing your explanation, output your final verdict by strictly following this format:
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[[User Question]]
{{ orig_prompt }}

[The Start of Assistant A's Answer]
{{ response_A }}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{ response_B }}
[The End of Assistant B's Answer]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


#DEFAULT_BAD_REGEX_TEMPLATE_PROMPT = r"(?s)(?<=\[User Question Modified Start\])(.*?)(?=\[User Question Modified End\])"
DEFAULT_BAD_REGEX_TEMPLATE_RESP = r"(?i)(?s)(?<=\[assistant modified instruction start\])(.*?)(?=\[assistant modified instruction end\])"
DEFAULT_JUDGEMENT_REGEX_TEMPLATE = r"(?i)\[\[(A|B|\"A\"|\"B\")\]\]"


def ids_to_text(self, ids):
    tokens = self.ids_to_tokens(ids)
    text = self.tokens_to_text(tokens)
    return text


class SelfTaughtTrainer:
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

        self.num_responses_to_gen = self.model.cfg.self_taught.num_responses_to_gen
        self.length_params = OmegaConf.to_container(self.model.cfg.self_taught.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.model.cfg.self_taught.sampling_params, resolve=True)
        self.max_gen_seq_len = self.length_params["max_length"]
        dp_batch_size = self.model.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
        assert (
            self.model.cfg.self_taught.rollout_micro_batch_size % dp_batch_size == 0
        ), f"rollout_micro_batch_size [{self.model.cfg.self_taught.rollout_micro_batch_size}] must be a multiple of GBS [{self.model.cfg.global_batch_size}] // DP [{parallel_state.get_data_parallel_world_size()}]"
        self.rollout_micro_batch_size = self.model.cfg.self_taught.rollout_micro_batch_size
        assert self.rollout_micro_batch_size > 0, "`rollout_micro_batch_size` must be > 0"

        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response"])

        if isinstance(self.model.tokenizer, AutoTokenizer):
            self.tokenizer = copy.copy(self.model.tokenizer)
            self.tokenizer.ids_to_text = partial(ids_to_text, self.tokenizer)
        else:
            self.tokenizer = self.model.tokenizer

        self.bad_response_template = DEFAULT_BAD_RESPONSE_PROMPT_LLAMA3 #self.model.cfg.self_taught.get("bad_response_template", DEFAULT_BAD_RESPONSE_PROMPT).strip()
        self.llm_judge_template = DEFAULT_JUDGEMENT_ANNOTATION_LLAMA3 #self.model.cfg.self_taught.get("llm_judge_template", DEFAULT_JUDGEMENT_ANNOTATION).strip()
        #self.bad_response_regex_prompt = self.model.cfg.self_taught.get("bad_response_regex_prompt", DEFAULT_BAD_REGEX_TEMPLATE_PROMPT)
        self.bad_response_regex_resp = DEFAULT_BAD_REGEX_TEMPLATE_RESP #self.model.cfg.self_taught.get("bad_response_regex_response", DEFAULT_BAD_REGEX_TEMPLATE_RESP)
        self.judgement_regex = DEFAULT_JUDGEMENT_REGEX_TEMPLATE #self.model.cfg.self_taught.get("judgement_regex", DEFAULT_JUDGEMENT_REGEX_TEMPLATE)

        assert find_variables_from_jinja_template(self.bad_response_template) == {
            "orig_prompt",
            "orig_response",
        }, "bad_response_template must include `orig_prompt` and `orig_response` templating variables"
        assert find_variables_from_jinja_template(self.llm_judge_template) == {
            "orig_prompt",
            "response_A",
            "response_B",
        }, "llm_judge_template must include `orig_prompt`, `response_A`, and `response_B` templating variables"

        self.bad_response_template_fn = jinja2_env.from_string(self.bad_response_template).render
        self.llm_judge_template_fn = jinja2_env.from_string(self.llm_judge_template).render
        #self.bad_response_regex_prompt_fn = create_parse_regex_fn(self.bad_response_regex_prompt)
        self.bad_response_regex_resp_fn = create_parse_regex_fn(self.bad_response_regex_resp)
        self.judgement_regex_fn = create_parse_regex_fn(self.judgement_regex)
        
        rng_generator = torch.Generator(device="cpu")
        seed = 1234 if self.model.cfg.get("seed", None) is None else self.model.cfg.get("seed")
        rng_generator.manual_seed(seed + parallel_state.get_data_parallel_rank())
        self.rng_generator = rng_generator

        self.use_trtllm_generation = self.cfg.trt_llm.get("enable", False) if "trt_llm" in self.cfg else False
        if self.use_trtllm_generation:
            #assert HAVE_TRTLLM, "TRTLLM generation was enabled but TRTLLM libraries could not be successfully imported"
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.model.cfg,
                end_strings=self.sampling_params["end_strings"],
                tokenizer=self.model.tokenizer,
                sample_temperature=self.sampling_params["temperature"],
                sample_top_k=self.sampling_params["top_k"],
                sample_top_p=self.sampling_params["top_p"],
                repetition_penalty=self.sampling_params["repetition_penalty"],
                max_generation_length=self.length_params["max_length"],
                max_input_len=self.cfg.trt_llm.get("max_input_len", self.model.cfg.encoder_seq_length // 2),
                generation_batch_size=self.cfg.trt_llm.get(
                    "generation_batch_size", self.model.cfg.self_taught.get("rollout_micro_batch_size", 4)
                ),
                use_greedy=self.sampling_params.get("use_greedy", False),
                trt_model_type=self.cfg.trt_llm.get("model_type", "llama"),
                seed=self.model.cfg.get("seed", None),
                unload_engine_train=self.cfg.trt_llm.get("unload_engine_train", False),
                reshard_model=False,
            )

    def validation_step(self, global_batch):
        # these things should go into a GPTModel wrapper
        self.model.prepare_for_validation_step()

        loss_mean, metrics = self.model.get_loss_and_metrics(batch=global_batch, forward_only=True)

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
        
        num_samples = global_batch["tokens"].shape[0]
        num_bad_samples = global_batch["bad_samples"].sum()
        num_bad_ends = global_batch["bad_ends"].sum()
        gen_lengths_chosen = (global_batch["gen_lens"] - global_batch["prompt_lens"]).sum()
        tensor_to_accumulate = torch.tensor(
            [
                gen_lengths_chosen,
                num_bad_samples,
                num_bad_ends,
                num_samples,
            ],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group())

        (
            global_chosen_response_lengths,
            GBS_sum_bad_samples,
            GBS_sum_bad_ends,
            GBS_num_samples,
        ) = tensor_to_accumulate.tolist()
        metrics["response_lengths"] = global_chosen_response_lengths / GBS_num_samples
        metrics["bad_samples_per_GBS"] = GBS_sum_bad_samples / GBS_num_samples
        metrics["bad_ends_per_GBS"] = GBS_sum_bad_ends / (GBS_num_samples * self.num_responses_to_gen)

        return loss_mean, {**metrics, **trainer_metrics}

    @torch.no_grad()
    def get_generations(self, list_of_batches, prepare_for_inference=False):
        if prepare_for_inference:
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
            if generations is None:
                response_tokens = None
            else:
                max_len_list = max([len(x) for x in generations["token_ids"]])
                padded_list = [x + [self.model.tokenizer.eos_id] * (max_len_list - len(x)) for x in generations["token_ids"]]
                response_tokens = torch.tensor(padded_list, dtype=torch.long, device='cuda')
            #response_tokens = torch.tensor(generations["token_ids"], dtype=torch.long, device='cuda') if generations else None
            response_tokens = broadcast_tensor_within_pp(response_tokens, dtype=torch.long)
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

        if prepare_for_inference:
            self.model.finish_inference()
            if self.use_trtllm_generation:
                self.trtllm_generate.free()

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_valid.cpu()
    
    def get_perturbed_responses(self, list_of_batches):
        #perturbed_responses = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in list_of_batches]))]
        perturbed_responses = []
        perturb_responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_batches)
        batch_responses_str = []
        for t, s, e, end in zip(perturb_responses, prompt_lengths.tolist(), resp_lengths.tolist(), is_end.tolist()):
            response = self.tokenizer.ids_to_text(t[s:e].tolist())
            batch_responses_str.append(response)
            if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
                print(f"*** PERTURBED_PROMPT_AND_RESP  [ {self.tokenizer.ids_to_text(t[:e].tolist())} ]")
        perturbs_as_str = [self.bad_response_regex_resp_fn(resp_str.strip()) for resp_str in batch_responses_str]
        for idx, (r, end) in enumerate(zip(perturbs_as_str, is_end.tolist())):
            perturbed_responses.append(
                r if (end and (r is not None)) else None
            )

        #perturbed_responses = [[*filter(exists, b)] for b in perturbed_responses]

        return perturbed_responses

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
                    loss, metrics = self.train_single_step_sft(global_batch)
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
                        if torch.distributed.get_rank() == 0:
                            for idx in range(len(global_batch["bad_samples"])):
                                if not global_batch["bad_samples"][idx]:
                                    self.train_df.loc[len(self.train_df)] = [
                                        self.step,
                                        self.tokenizer.ids_to_text(
                                            global_batch["tokens"][idx][
                                                : global_batch["prompt_lens"][idx].item()
                                            ].tolist()
                                        ),
                                        self.tokenizer.ids_to_text(
                                            global_batch["tokens"][idx][
                                                global_batch["prompt_lens"][idx]
                                                .item() : global_batch["gen_lens"][idx]
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
                    #self.model.finish_training()

            # update the reference policy weights
            if self.model.cfg.self_taught.get("update_ref_policy", True):
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

        #self.model.finish_training()

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
    
    def extract_prompt_elements(self, prompt, response, dataset_mask):
        if self.cfg.trt_llm.get("model_type", "gptnext").lower() == "llama":
            p_list = re.findall(
                rf"(?s)(?<={re.escape(self.model.cfg.data.chat_prompt_tokens.end_of_turn + dataset_mask)}\n\n).*?(?={re.escape(self.model.cfg.data.chat_prompt_tokens.end_of_turn)})",
                prompt,
            )
            r_list = re.findall(
                rf"(?s)(?<={re.escape(self.model.cfg.data.chat_prompt_tokens.end_of_turn + dataset_mask.replace('user', 'assistant'))}\n\n).*?(?={re.escape(self.model.cfg.data.chat_prompt_tokens.end_of_turn)})",
                prompt,
            )
            resp_raw = response.replace(self.model.cfg.data.chat_prompt_tokens.end_of_turn, "").strip()
        else:
            p_list = re.findall(
                rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}User\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})",
                prompt,
            )
            r_list = re.findall(
                rf"(?s)(?<={self.model.cfg.data.chat_prompt_tokens.turn_start}Assistant\n).*?(?=\n{self.model.cfg.data.chat_prompt_tokens.turn_start})",
                prompt,
            )
            resp_raw = response.replace(f"\n{self.model.cfg.data.chat_prompt_tokens.turn_start}", "")

        return p_list, r_list, resp_raw

    def normalise_prompt(self, prompt, response, dataset_mask):
        p_list, r_list, resp_raw = self.extract_prompt_elements(prompt, response, dataset_mask)
        if len(p_list) == 1 and len(r_list) == 0:
            return "User: " + p_list[0], resp_raw, p_list[-1]
        elif len(p_list) == len(r_list) + 1:
            comp = "User: " + p_list[0]
            for p, r in zip(p_list[1:], r_list):
                comp += "\n\nAssistant: " + r
                comp += "\n\nUser: " + p
            return comp, resp_raw, p_list[-1]
        else:
            raise RuntimeError(
                f"Received strange normalise payload PROMPT [ {prompt} ]  P_LIST [ {p_list} ]  R_LIST [ {r_list} ]"
            )

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations and ref policy log probs"""
        iter_dataloader = iter(dataloader)
        buffer = []
        done = False
        while not done:
            try:
                batch = next(iter_dataloader)
            except StopIteration:
                done = True
            else:
                buffer.append(batch)

            if (done and buffer) or sum(
                [len(b["prompts_and_answers"]) for b in buffer]
            ) == self.rollout_micro_batch_size:
                # generations use the reference model weights, as per the paper
                with cpu_weight_swap(
                    self.model, self.model.ref_policy_state_dict, megatron_amp_O2=self.model.megatron_amp_O2
                ):
                    self.model.prepare_for_inference()
                    if self.use_trtllm_generation:
                        # at this point self.model is the reference policy from cpu_weight_swap
                        self.trtllm_generate.refit(self.model)
                        clear_memory()

                    #candidate_responses_with_perturbs = [
                    #    [] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))
                    #]
                    #for _ in range(self.num_responses_to_gen):
                        # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                    gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(buffer)

                    # Transform into batch of LLM-as-judge template samples for reward scoring
                    perturb_buffer = []
                    orig_prompts_and_responses = []
                    for t, s, e in zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist()):
                        
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
                        
                        # llama3
                        # prompt = self.tokenizer.ids_to_text(t[:s].tolist()).replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "")
                        # response = self.tokenizer.ids_to_text(t[s:e].tolist()).replace("<|eot_id|>", "").strip()
                        #prompt, response = self.normalise_prompt(self.tokenizer.ids_to_text(t[:s].tolist()), self.tokenizer.ids_to_text(t[s:e].tolist()), buffer[0]["dataset_mask"])
                        perturb_prompt_str = self.bad_response_template_fn(orig_prompt=prompt, orig_response=response)
                        perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)
                        # if len(reward_prompt) > (self.model.cfg.encoder_seq_length - self.max_gen_seq_len):
                        if len(perturb_prompt) > self.model.cfg.data.train_ds.max_seq_length:
                            #prompt_and_response = self.tokenizer.ids_to_text(t[:e].tolist())
                            #try:
                            p_list, _, _ = self.extract_prompt_elements(
                                prompt, response, buffer[0]["dataset_mask"]
                            )
                            if len(p_list) == 0:
                                prompt_ft = prompt
                            else:
                                prompt_ft = p_list[-1]
                            response_ft = response
                            # llama3
                            # prompt_ft = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>user\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt_and_response)[0]
                            # response_ft = re.findall(r"(?s)(?<=\<\|eot_id\|\>\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n\n).*?(?=\<\|eot_id\|\>)", prompt_and_response)[0]
                            perturb_prompt_str = self.bad_response_template_fn(orig_prompt=prompt_ft, orig_response=response_ft)
                            perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)

                            while len(perturb_prompt) > (
                                self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                            ):
                                overage = len(perturb_prompt) - self.model.cfg.data.train_ds.max_seq_length
                                if overage > len(self.model.tokenizer.text_to_ids(response_ft)):
                                    print(f"*** OVERAGE_NOT_FIT_RESPONSE: {perturb_prompt_str}")
                                    perturb_prompt_str = self.bad_response_template_fn(
                                        orig_prompt="How does one make tea?", orig_response="I have no answer at all."
                                    )
                                    perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)
                                    break
                                response_ft = self.tokenizer.ids_to_text(
                                    self.model.tokenizer.text_to_ids(response_ft)[:-overage]
                                )
                                perturb_prompt_str = self.bad_response_template_fn(orig_prompt=prompt_ft, orig_response=response_ft)
                                perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)
                            prompt = prompt_ft
                            response = response_ft
                            '''
                            except:
                                print(f"*** TOO_LONG: {prompt_and_response}")
                                # overage = len(reward_prompt) - (self.model.cfg.encoder_seq_length - self.max_gen_seq_len)
                                while len(perturb_prompt) > (
                                    self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                ):
                                    overage = len(perturb_prompt) - self.model.cfg.data.train_ds.max_seq_length
                                    if len(self.model.tokenizer.text_to_ids(response)) >= overage:
                                        # truncate response only
                                        response = self.tokenizer.ids_to_text(
                                            self.model.tokenizer.text_to_ids(response)[:-overage]
                                        )
                                        perturb_prompt_str = self.bad_response_template_fn(orig_prompt=prompt, orig_response=response)
                                        perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)
                                    else:
                                        # truncate response and prompt *SHOULD NEVER HAPPEN*
                                        print("*** PROMPT_AND_RESPONSE_NEED_TRUNCATION")
                                        perturb_prompt_str = self.bad_response_template_fn(
                                            orig_prompt="How does one make tea?", orig_response="I have no answer at all."
                                        )
                                        perturb_prompt = self.model.tokenizer.text_to_ids(perturb_prompt_str)
                                        break
                            '''
                            assert len(perturb_prompt) <= (
                                self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                            ), f"truncation of response only failed [ {len(perturb_prompt)} ]: {perturb_prompt_str}"

                        perturb_buffer.append(
                            {
                                "prompts_only": torch.LongTensor(perturb_prompt).unsqueeze(0),
                                "prompt_lengths": torch.LongTensor([len(perturb_prompt)]),
                            }
                        )
                        orig_prompts_and_responses.append( (prompt, response) )

                    judgement_cand_list = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))]
                    perturbed_responses_as_str = self.get_perturbed_responses(perturb_buffer)
                    
                    for _ in range(self.num_responses_to_gen):
                        judgement_buffer = []
                        corrects = []
                        for (orig_prompt, resp_good), resp_bad in zip(orig_prompts_and_responses, perturbed_responses_as_str):
                            if resp_bad is None:
                                resp_bad = "I cannot come up with a response"
                            if torch.rand([1], generator=self.rng_generator) < 0.5:
                                resp_A = resp_good
                                resp_B = resp_bad
                                correct_ans = 'A'
                            else:
                                resp_A = resp_bad
                                resp_B = resp_good
                                correct_ans = 'B'
                            judge_template_str = self.llm_judge_template_fn(orig_prompt=orig_prompt, response_A=resp_A, response_B=resp_B)
                            judge_template = self.model.tokenizer.text_to_ids(judge_template_str)
                            if len(judge_template) > self.model.cfg.data.train_ds.max_seq_length:
                                while len(judge_template) > (
                                    self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                ):
                                    overage = len(judge_template) - self.model.cfg.data.train_ds.max_seq_length
                                    overage_2 = overage // 2
                                    if len(self.model.tokenizer.text_to_ids(resp_A)) >= overage_2 and len(self.model.tokenizer.text_to_ids(resp_B)) >= overage_2:
                                        # both responses equally
                                        resp_A = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_A)[:-overage_2])
                                        resp_B = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_B)[:-overage_2])
                                        judge_template_str = self.llm_judge_template_fn(orig_prompt=orig_prompt, response_A=resp_A, response_B=resp_B)
                                        judge_template = self.model.tokenizer.text_to_ids(judge_template_str)
                                    elif len(self.model.tokenizer.text_to_ids(resp_A)) >= overage and len(self.model.tokenizer.text_to_ids(resp_B)) < overage:
                                        resp_A = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_A)[:-overage])
                                        judge_template_str = self.llm_judge_template_fn(orig_prompt=orig_prompt, response_A=resp_A, response_B=resp_B)
                                        judge_template = self.model.tokenizer.text_to_ids(judge_template_str)
                                    elif len(self.model.tokenizer.text_to_ids(resp_A)) < overage and len(self.model.tokenizer.text_to_ids(resp_B)) >= overage:
                                        resp_B = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_B)[:-overage])
                                        judge_template_str = self.llm_judge_template_fn(orig_prompt=orig_prompt, response_A=resp_A, response_B=resp_B)
                                        judge_template = self.model.tokenizer.text_to_ids(judge_template_str)
                                    else:
                                        # truncate response and prompt *SHOULD NEVER HAPPEN*
                                        print("*** PROMPT_AND_RESPONSE_NEED_TRUNCATION")
                                        judge_template_str = self.llm_judge_template_fn(orig_prompt="How does one make tea?", response_A="I have no answer at all.", response_B="I have no answer at all.")
                                        judge_template = self.model.tokenizer.text_to_ids(judge_template_str)
                                        break
                            assert len(judge_template) <= (
                                self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                            ), f"truncation of judgement prompt failed [ {len(judge_template)} ]: {judge_template_str}"
                            judgement_buffer.append(
                                {
                                    "prompts_only": torch.LongTensor(judge_template).unsqueeze(0),
                                    "prompt_lengths": torch.LongTensor([len(judge_template)]),
                                }
                            )
                            corrects.append(correct_ans)
                        # send to generator
                        gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(judgement_buffer)
                        for idx, (t, s, e, (p_o, resp_A), resp_B, end, c_ans) in enumerate(
                            zip(
                                gen_tokens_buf,
                                gen_prompt_lengths_buf.tolist(),
                                gen_lengths_buf.tolist(),
                                orig_prompts_and_responses,
                                perturbed_responses_as_str,
                                is_end.tolist(),
                                corrects,
                            )
                        ):
                            resp_str = self.tokenizer.ids_to_text(t[s:e].tolist())
                            winner = self.judgement_regex_fn(resp_str.replace("\n<extra_id_1>", "").replace("<|eot_id|>", "").strip())
                            if (resp_B is None) or (resp_B == "I cannot come up with a response") or (winner not in ['A', 'B']):
                                winner = None
                            if winner != c_ans:
                                print(f"*** WINNER_WRONG: [ {winner} ]  [ {c_ans} ]  [ {resp_B} ]  [ {resp_str} ]")
                            judgement_cand_list[idx].append((winner == c_ans, t, s, e, p_o, resp_A, resp_B, end))

                    final_buffer = []
                    for cand_list in judgement_cand_list:
                        scores = [b[0] for b in cand_list]
                        ends = [b[-1] for b in cand_list]
                        resp_lengths = [len(b[1][b[2]:b[3]]) for b in cand_list]
                        filtered_scores = [
                            (s, r, idx)
                            for idx, (s, r, e) in enumerate(zip(scores, resp_lengths, ends))
                            if (s is not None and s == True) and e
                        ]
                        bad_sample = False

                        # if all scores are identical (even all None) we just randomly choose
                        if len(filtered_scores) == 0:
                            idx_chosen = 0
                            bad_sample = True
                        elif len(filtered_scores) > 0:
                            idx_chosen = filtered_scores[torch.randint(0, len(filtered_scores), [1], generator=self.rng_generator).item()][-1]
                        else:
                            print(f"*** final_scores [ {scores} ]  final_filtered_scores [ {filtered_scores} ]")
                            raise RuntimeError("hit strange score selection state, please investigate")

                        # 1 x max_len tensor
                        chosen_prompt_len = cand_list[idx_chosen][2]
                        chosen_resp_len = cand_list[idx_chosen][3]
                        chosen_prompt_and_resp_tokens = cand_list[idx_chosen][1][:chosen_resp_len]
                        chosen_prompt_tokens = cand_list[idx_chosen][1][:chosen_prompt_len]
                        chosen_response_tokens = cand_list[idx_chosen][1][chosen_prompt_len:chosen_resp_len]
                        bad_ends = sum(~np.array([cand_list[idx_chosen][-1]]))

                        final_buffer.append(
                            {
                                "chosen_prompt_and_resp_tokens": chosen_prompt_and_resp_tokens,
                                "chosen_prompt_len": chosen_prompt_len,
                                "chosen_resp_len": chosen_resp_len,
                                "chosen_prompt_tokens": chosen_prompt_tokens,
                                "chosen_response_tokens": chosen_response_tokens,
                                "bad_sample": bad_sample,
                                "bad_ends": bad_ends,
                            }
                        )
                
                self.model.finish_inference()
                if self.use_trtllm_generation:
                    self.trtllm_generate.free()

                original_gbs_size = len(buffer[0]["prompt_lengths"])
                for batch in divide_chunks(final_buffer, original_gbs_size):
                    chosen_prompt_lens = torch.LongTensor([b["chosen_prompt_len"] for b in batch])
                    chosen_resp_lens = torch.LongTensor([b["chosen_resp_len"] for b in batch])
                    bad_samples = torch.BoolTensor([b["bad_sample"] for b in batch])

                    max_batch_len = max([len(b["chosen_prompt_and_resp_tokens"]) - 1 for b in batch])
                    #max_batch_len = self.model.cfg.encoder_seq_length

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
                        [b["chosen_prompt_and_resp_tokens"][:-1] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id
                    )
                    #chosen_labels = batch_pad_to_fixed_len([torch.LongTensor(([-100] * b["chosen_prompt_len"]) + b["chosen_response_tokens"].tolist()) for b in batch], max_batch_len, pad_token=-100)
                    chosen_labels = batch_pad_to_fixed_len([b["chosen_prompt_and_resp_tokens"][1:] for b in batch], max_batch_len, pad_token=self.model.tokenizer.eos_id)

                    chosen_mask = create_mask(
                        chosen_tokens_pad, chosen_prompt_lens, chosen_resp_lens
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
                        "tokens": chosen_tokens_pad,
                        "position_ids": position_ids,
                        "attention_mask": attention_mask,
                        "labels": chosen_labels,
                        "loss_mask": chosen_mask,
                        "prompt_lens": chosen_prompt_lens,
                        "gen_lens": chosen_resp_lens,
                        "bad_samples": bad_samples,
                        "bad_ends": torch.IntTensor([b["bad_ends"] for b in batch]),
                    }

                    assert (
                        chosen_resp_lens - chosen_prompt_lens >= 0
                    ).all(), "negative generated length encountered in chosen"

                    yield new_batch
                    
                    # the del runs after all code connected to receiving the yielded new_batch
                    del new_batch

                buffer.clear()

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs

    @property
    def iteration(self):
        return (self.step // self.num_steps_per_epoch) // self.cfg.max_epochs
