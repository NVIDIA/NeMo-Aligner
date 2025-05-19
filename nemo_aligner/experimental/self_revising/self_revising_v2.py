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
import itertools
from statistics import mean
from textwrap import dedent

import numpy as np
import pandas as pd
import torch
from jinja2 import meta
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
from nemo_aligner.utils.distributed import SyncTimer, broadcast_tensor_within_pp
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

try:
    from nemo_aligner.utils.verifiers.instruction_following.instructions_registry import INSTRUCTION_DICT

    HAVE_VERIFIER = True
except (ImportError, ModuleNotFoundError) as e:
    logging.info(f"got error message {e} when importing aligner verifier, disabling")
    HAVE_VERIFIER = False

assert HAVE_VERIFIER, "IF-VERIFIER FAILED TO IMPORT"

"""
GPTSFTChatDataset output is dict with keys: ['input_ids', 'mask', 'context_ids', 'answer_ids', 'metadata']

input_ids: torch.LongTensor - the entire prompt + response, including the system preamble which is specified by "system" in the jsonl
mask: torch.BoolTensor with False for the preamble+prompt, and True for the response
context_ids: torch.LongTensor - the entire preamble + prompt
answer_ids: torch.LongTensor - the entire response only
metadata: dict - with keys "system" for the preamble, and "mask" which is "User" or "Assistant"
"""


def self_revising_custom_collate(batch, eos_id):
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
        "dataset_mask": batch[0]["metadata"]["mask"] if "metadata" in batch[0] else "",
        "verifier_args": [
            (x["metadata"]["verifier_args"] if "verifier_args" in x["metadata"] else None) for x in batch
        ],
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

def ids_to_text(self, ids):
    tokens = self.ids_to_tokens(ids)
    text = self.tokens_to_text(tokens)
    return text


DEFAULT_CRITIQUE_PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

Please assess the Assistant's Answer inside the brackets above and provide a detailed critique of how helpful you believe this Answer to be in relation to the User's query. Provide a moderate length (2-10 sentences / 50-250 words) justification for your critique of the Answer.

Do not include links used for fact-checking
Avoid first person statements ("I think that...")
Avoid vague statements/lack of specificity
Avoid lists (numbered, bulleted, etc.)
Ensure all sentences are complete and with no grammatical/spelling errors

You should provide negative feedback for Answers which are too verbose or overly long, especially Answers which have repetition of sentences or phrases throughout their response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


DEFAULT_CRITIQUE_PROMPT_LLAMA3_VERIFIER_MATH = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

The ideal, Expected Answer to the User's question is shown below:

{{ expected_answer }}

Please judge and critique the Assistant's Answer inside the brackets above against the Expected Answer shown above, and provide constructive feedback on how the Assistant can improve their answer so that it is as close as possible to the Expected Answer. Provide a moderate length (2-10 sentences / 50-250 words) justification for your critique.

Do not include links used for fact-checking
Avoid first person statements ("I think that...")
Avoid vague statements/lack of specificity
Avoid lists (numbered, bulleted, etc.)
Ensure all sentences are complete and with no grammatical/spelling errors

You should provide negative feedback for Answers which are too verbose or overly long, especially Answers which have repetition of sentences or phrases throughout their response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


DEFAULT_CRITIQUE_PROMPT_LLAMA3_VERIFIER_IFEVAL = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

Please assess the Assistant's Answer inside the brackets above and provide a detailed critique of how helpful you believe this Answer to be in relation to the User's query. Provide a moderate length (2-10 sentences / 50-250 words) justification for your critique of the Answer.

Do not include links used for fact-checking
Avoid first person statements ("I think that...")
Avoid vague statements/lack of specificity
Avoid lists (numbered, bulleted, etc.)
Ensure all sentences are complete and with no grammatical/spelling errors

Additionally, the Assistant's Answer has failed to follow the explicit instructions in the User's Prompt, with an Instruction Following Correctness Score of {{ correctness }} out of 1.0.
The specific instruction following issues detected which need to be critiqued and corrected are:

{{ v_details }}

You should focus your critique on the fact that these instructions were not followed, and provide suggestions on how the response can be revised to better follow these instructions.

You should provide negative feedback for Answers which are too verbose or overly long, especially Answers which have repetition of sentences or phrases throughout their response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


DEFAULT_REVISE_PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

The Assistant's Answer was sent to {{ num_annotators }} human annotator{{ s_or_not }} to evaluate its quality.
Below are the comments of the annotator{{ s_or_not }}:

[Start of annotator comments]
{{ critique }}
[End of annotator comments]

You must revise the Assistant's Answer to improve it according to the feedback above, only changing what is required to address this feedback.
Reply with the Revised Response only, do not include any additional introductory text.

Ensure your revised response does NOT contain text which states that this is a revised response of some kind. Do not include any phrases of the following form: "Here is the revised response", "I have revised the response", "This is the revised response", etc.
Do NOT provide any feedback to the annotator comments, nor any justifications or explanations for your revisions.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


DEFAULT_SELECTOR_ANNOTATION_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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


#DEFAULT_CRITIQUE_REGEX = r"(?i)(?s)(?<=\[CRITIQUE RESPONSE START\])(.*?)(?=\[CRITIQUE RESPONSE END\])"
#DEFAULT_REVISED_REGEX = r"(?i)(?s)(?<=\[REVISED RESPONSE START\])(.*?)(?=\[REVISED RESPONSE END\])"
#DEFAULT_REVISED_REGEX = r"(?i)(?s)(?<=\[REVISED RESPONSE START\]).*"

DEFAULT_JUDGEMENT_REGEX_TEMPLATE = r"(?i)\[\[(A|B|\"A\"|\"B\")\]\]"


class SelfRevisingTrainer:
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
        alt_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.alt_dataloader = alt_dataloader
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
        self.num_steps_per_epoch = compute_num_steps_per_epoch(self.train_dataloader.batch_sampler)

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

        '''
        self.spin_config = OmegaConf.to_container(self.model.cfg.spin, resolve=True)
        if isinstance(self.spin_config["length_control"], (float, int)):
            self.rho = self.spin_config["length_control"]
        elif isinstance(self.spin_config["length_control"], list):
            self.rho = 0.0
        else:
            raise TypeError(
                f"`length_control` must be a scalar or list, but got {type(self.spin_config['length_control'])}"
            )
        '''

        #self.num_responses_to_gen = self.model.cfg.spin.num_responses_to_gen
        self.num_critiques_to_gen = self.model.cfg.spin.num_critiques_to_gen
        
        self.use_meta_critiques = self.model.cfg.spin.get("use_meta_critiques", False)
        self.meta_critiques_pcnt = self.model.cfg.spin.get("meta_critiques_pcnt", -1.0)

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

        self.critique_template = DEFAULT_CRITIQUE_PROMPT_LLAMA3 #self.model.cfg.spin.get("critique_prompt_template")
        self.critique_verifier_math_template = DEFAULT_CRITIQUE_PROMPT_LLAMA3_VERIFIER_MATH #self.model.cfg.spin.get("critique_verifier_math_prompt_template")
        self.critique_verifier_ifeval_template = DEFAULT_CRITIQUE_PROMPT_LLAMA3_VERIFIER_IFEVAL #self.model.cfg.spin.get("critique_verifier_ifeval_prompt_template")
        self.revise_template = DEFAULT_REVISE_PROMPT_LLAMA3 #self.model.cfg.spin.get("revise_prompt_template")
        self.select_template = DEFAULT_SELECTOR_ANNOTATION_LLAMA3 #self.model.cfg.spin.get("select_prompt_template")
        #self.critique_regex = DEFAULT_CRITIQUE_REGEX #self.model.cfg.spin.get("critique_regex")
        #self.revised_regex = DEFAULT_REVISED_REGEX #self.model.cfg.spin.get("revised_regex")
        self.select_regex = DEFAULT_JUDGEMENT_REGEX_TEMPLATE #self.model.cfg.spin.get("select_regex")

        assert find_variables_from_jinja_template(self.critique_template) == {
            "prompt",
            "response",
        }, "critique_prompt_template must include `prompt` and `response` templating variables"
        assert find_variables_from_jinja_template(self.critique_verifier_math_template) == {
            "prompt",
            "response",
            "expected_answer",
        }, "critique_verifier_math_template must include `prompt`, `response`, and `expected_answer` templating variables"
        assert find_variables_from_jinja_template(self.critique_verifier_ifeval_template) == {
            "prompt",
            "response",
            "correctness",
            "v_details",
        }, "critique_verifier_ifeval_template must include `prompt`, `response`, `correctness`, and `v_details` templating variables"
        assert find_variables_from_jinja_template(self.revise_template) == {
            "prompt",
            "response",
            "num_annotators",
            "s_or_not",
            "critique",
        }, "revise_prompt_template must include `prompt`, `response`, `num_annotators`, `s_or_not`, and `critique` templating variables"
        assert find_variables_from_jinja_template(self.select_template) == {
            "orig_prompt",
            "response_A",
            "response_B",
        }, "select_prompt_template must include `orig_prompt`, `response_A`, and `response_B` templating variables"

        self.critique_template_fn = jinja2_env.from_string(self.critique_template).render
        self.critique_verified_math_template_fn = jinja2_env.from_string(self.critique_verifier_math_template).render
        self.critique_verified_ifeval_template_fn = jinja2_env.from_string(self.critique_verifier_ifeval_template).render
        self.revise_template_fn = jinja2_env.from_string(self.revise_template).render
        self.select_template_fn = jinja2_env.from_string(self.select_template).render
        #self.critique_regex_fn = create_parse_regex_fn(self.critique_regex)
        #self.revised_regex_fn = create_parse_regex_fn(self.revised_regex)
        self.select_regex_fn = create_parse_regex_fn(self.select_regex)

        #seed = 1234 if self.model.cfg.get("seed", None) is None else self.model.cfg.get("seed")
        #self.rng_generator = np.random.default_rng(seed + parallel_state.get_data_parallel_rank())
        
        rng_generator = torch.Generator(device="cpu")
        seed = 1234 if self.model.cfg.get("seed", None) is None else self.model.cfg.get("seed")
        rng_generator.manual_seed(seed + parallel_state.get_data_parallel_rank())
        self.rng_generator = rng_generator

        self.use_trtllm_generation = self.cfg.trt_llm.get("enable", False) if "trt_llm" in self.cfg else False
        if self.use_trtllm_generation:
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
        metrics["bad_ends_per_GBS"] = GBS_sum_bad_ends / (GBS_num_samples)
        metrics["chosen_generated_rewards"] = global_chosen_rewards / GBS_num_samples
        metrics["rejected_generated_rewards"] = global_reject_rewards / GBS_num_samples

        return loss_mean, {**metrics, **trainer_metrics}
    
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
        '''
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
        metrics["bad_ends_per_GBS"] = GBS_sum_bad_ends / (GBS_num_samples)
        '''

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
                padded_list = [
                    x + [self.model.tokenizer.eos_id] * (max_len_list - len(x)) for x in generations["token_ids"]
                ]
                response_tokens = torch.tensor(padded_list, dtype=torch.long, device="cuda")
            # response_tokens = torch.tensor(generations["token_ids"], dtype=torch.long, device='cuda') if generations else None
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
    
    def instruction_following_rewards(self, prompt, response, args):
        """Tests response to see if instrutions are followed."""
        try:
            task_args = args
            instruction_list = task_args["instruction_id_list"]
            is_following_list = []

            for index, instruction_id in enumerate(instruction_list):
                try:
                    instruction_cls = INSTRUCTION_DICT[instruction_id]
                    instruction = instruction_cls(instruction_id)

                    kwargs = (
                        task_args["instruction_kwargs"][index]
                        if task_args["instruction_kwargs"][index] is not None
                        else {}
                    )
                    instruction.build_description(**kwargs)
                    instruction_args = instruction.get_instruction_args()
                    if instruction_args and "prompt" in instruction_args:
                        instruction.build_description(prompt=prompt)

                    if response.strip() and instruction.check_following(response):
                        is_following_list.append((True, instruction_id, instruction_args))
                    else:
                        is_following_list.append((False, instruction_id, instruction_args))
                except Exception as e:
                    print(f"Error in instruction_following_rewards: {e}, task: {args}")

            #score_low, score_high = 0, 5
            correctness = sum([x[0] for x in is_following_list]) / len(is_following_list)
            #score = score_low + (score_high - score_low) * correctness
            return correctness, [x[1:] for x in is_following_list if x[0] == False], True
        except Exception as e:
            print(f"Error in instruction_following_rewards: {e}")
            return 0, [], False
    
    def get_tagged_responses(self, list_of_batches, regex_fn=None, prepare_for_inference=False):
        tagged_responses = []
        responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_batches, prepare_for_inference=prepare_for_inference)
        batch_responses_str = []
        for t, s, e, end in zip(responses, prompt_lengths.tolist(), resp_lengths.tolist(), is_end.tolist()):
            response = self.tokenizer.ids_to_text(t[s:e].tolist())
            batch_responses_str.append(response)
            if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
                print(f"*** TAGGED_PROMPT_AND_RESP: {self.tokenizer.ids_to_text(t[:e].tolist())}")
                print(f"*** TAGGED_RESP_ONLY: {self.tokenizer.ids_to_text(t[s:e].tolist())}")
        if regex_fn is not None:
            tagged_as_str = [regex_fn(resp_str.strip()) for resp_str in batch_responses_str]
        else:
            tagged_as_str = [resp_str.replace(self.model.cfg.data.chat_prompt_tokens.end_of_turn, "").strip() for resp_str in batch_responses_str]
        if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
            for idx, tag in enumerate(tagged_as_str):
                print(f"*** REGEX_TAG [{idx}]: {tag}")
        for idx, (r, end) in enumerate(zip(tagged_as_str, is_end.tolist())):
            tagged_responses.append(
                #r if end else None
                r
            )

        return tagged_responses

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
            #self.set_rho_by_iteration(self.iteration)

            # print(f"*** Iteration [ {self.iteration} ]  RHO [ {self.rho} ] ***")
            
            self.alt_iter = None
            if self.alt_dataloader is not None:
                self.alt_iter = iter(self.alt_dataloader)

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
                    if "tokens" in global_batch:
                        loss, metrics = self.train_single_step_sft(global_batch)
                    elif "chosen" in global_batch:
                        loss, metrics = self.train_single_step_dpo(global_batch)
                    else:
                        raise RuntimeError(f"Unrecognised local batch keys: {global_batch.keys()}")
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
                        if torch.distributed.get_rank() == 0 and "bad_samples" in global_batch:
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
            # reset optimizer
            self.optimizer.zero_grad()
            for state_bucket in self.optimizer.state_dict(state_dict_format=1, gather_on_root=False)["state"]["buckets"]:
                exp_avg_shard = getattr(state_bucket, "exp_avg_shard")
                exp_avg_sq_shard = getattr(state_bucket, "exp_avg_sq_shard")
                if exp_avg_shard is not None:
                    exp_avg_shard.zero_()
                if exp_avg_sq_shard is not None:
                    exp_avg_sq_shard.zero_()
            if "step" in self.optimizer.state_dict(state_dict_format=1, gather_on_root=False)["state"]:
                print(f'**** optimizer_step_A_reset : type {type(self.optimizer.state_dict(state_dict_format=1, gather_on_root=False)["state"]["step"])} ****')
                self.optimizer.state_dict(state_dict_format=1, gather_on_root=False)["state"]["step"] = 0

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
    
    def get_batch_from_alt_dataloader_old(self, iter_alt):
        try:
            batch = next(iter_alt)
            return batch, iter_alt
        except:
            iter_alt = iter(self.alt_dataloader)
            batch = next(iter_alt)
            return batch, iter_alt
    
    def get_batch_from_alt_dataloader(self):
        try:
            batch = next(self.alt_iter)
            return batch
        except:
            self.alt_iter = iter(self.alt_dataloader)
            batch = next(self.alt_iter)
            return batch
        
    def truncate_critique(self, prompt, response, buffer, vargs):
        p_list, _, resp_trunc = self.extract_prompt_elements(
            prompt, response, buffer[0]["dataset_mask"]
        )
        if len(p_list) == 0:
            prompt_ft = prompt
        else:
            prompt_ft = p_list[-1]
        response_ft = resp_trunc
        if vargs is None:
            critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
        elif "generated_solution" in vargs:
            critique_prompt_str = self.critique_verified_math_template_fn(prompt=prompt_ft, response=response_ft, expected_answer=vargs["generated_solution"])
        elif "instruction_id_list" in vargs:
            v_correctness, v_details, v_status = self.instruction_following_rewards(prompt_ft, response_ft, vargs)
            if v_status:
                critique_prompt_str = self.critique_verified_ifeval_template_fn(prompt=prompt_ft, response=response_ft, correctness=v_correctness, v_details=v_details)
            else:
                critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
        critique_prompt = self.model.tokenizer.text_to_ids(critique_prompt_str)

        while len(critique_prompt) > (
            self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
        ):
            overage = len(critique_prompt) - self.model.cfg.data.train_ds.max_seq_length
            if overage > len(self.model.tokenizer.text_to_ids(response_ft)):
                # print(f"*** OVERAGE_NOT_FIT_RESPONSE: {reward_prompt_str}")
                critique_prompt_str = self.critique_template_fn(
                    prompt="How does one make tea?", response="I have no answer at all."
                )
                critique_prompt = self.model.tokenizer.text_to_ids(critique_prompt_str)
                break
            response_ft = self.tokenizer.ids_to_text(
                self.model.tokenizer.text_to_ids(response_ft)[:-overage]
            )
            if vargs is None:
                critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
            elif "generated_solution" in vargs:
                critique_prompt_str = self.critique_verified_math_template_fn(prompt=prompt_ft, response=response_ft, expected_answer=vargs["generated_solution"])
            elif "instruction_id_list" in vargs:
                v_correctness, v_details, v_status = self.instruction_following_rewards(prompt_ft, response_ft, vargs)
                if v_status:
                    critique_prompt_str = self.critique_verified_ifeval_template_fn(prompt=prompt_ft, response=response_ft, correctness=v_correctness, v_details=v_details)
                else:
                    critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
            critique_prompt = self.model.tokenizer.text_to_ids(critique_prompt_str)

        return critique_prompt
    
    def truncate_revise(self, prompt, response, buffer, critique, bad_idxs, idx):
        p_list, r_list, resp_raw = self.extract_prompt_elements(
            prompt, response, buffer[0]["dataset_mask"]
        )
        if len(p_list) == 0:
            print("*** PROMPT_ALREADY_PRUNED: ", prompt)
            prompt_ft = prompt
        else:
            prompt_ft = p_list[-1]
        response_ft = resp_raw
        revise_prompt_str = self.revise_template_fn(prompt=prompt_ft, response=response_ft, num_annotators="1", s_or_not="", critique=critique)
        revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)

        while len(revise_prompt) > (
            self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
        ):
            overage = len(revise_prompt) - self.model.cfg.data.train_ds.max_seq_length
            if overage > len(self.model.tokenizer.text_to_ids(prompt_ft)):
                spillover = overage - len(self.model.tokenizer.text_to_ids(prompt_ft))
                response_ft = self.tokenizer.ids_to_text(
                    self.model.tokenizer.text_to_ids(response_ft)[:-spillover]
                )
                revise_prompt_str = self.revise_template_fn(prompt="", response=response_ft, num_annotators="1", s_or_not="", critique=critique)
                revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                break
            prompt_ft = self.tokenizer.ids_to_text(
                self.model.tokenizer.text_to_ids(prompt_ft)[:-overage]
            )
            revise_prompt_str = self.revise_template_fn(prompt=prompt_ft, response=response_ft, num_annotators="1", s_or_not="", critique=critique)
            revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
        
        if len(revise_prompt) > self.model.cfg.data.train_ds.max_seq_length:
            revise_prompt_str = self.revise_template_fn(prompt="", response=response_ft, num_annotators="1", s_or_not="", critique="I have no critique.")
            revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
            bad_idxs.append(idx)
        
        return revise_prompt
    
    def truncate_select(self, orig_prompt, buffer, resp_A, resp_B):
        p_list, r_list, _ = self.extract_prompt_elements(
            orig_prompt, "", buffer[0]["dataset_mask"]
        )
        if len(p_list) == 0:
            prompt_ft = orig_prompt
        else:
            prompt_ft = p_list[-1]
        
        select_prompt_str = self.select_template_fn(orig_prompt=prompt_ft, response_A=resp_A, response_B=resp_B)
        select_prompt = self.tokenizer.text_to_ids(select_prompt_str)
        
        resp_A_star = resp_A
        resp_B_star = resp_B
        while len(select_prompt) > (
            self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
        ):
            overage = len(select_prompt) - self.model.cfg.data.train_ds.max_seq_length
            overage //= 2
            
            resp_A_star = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_A_star)[:-overage])
            resp_B_star = self.tokenizer.ids_to_text(self.model.tokenizer.text_to_ids(resp_B_star)[:-overage])
            select_prompt_str = self.select_template_fn(orig_prompt=prompt_ft, response_A=resp_A_star, response_B=resp_B_star)
            select_prompt = self.tokenizer.text_to_ids(select_prompt_str)
        
        return select_prompt
    
    def create_meta_sample(self, orig_prompt, orig_response, crit_chosen, crit_reject):
        critique_prompt_str = self.critique_template_fn(prompt=orig_prompt, response=orig_response)
        prompt_tokens = torch.LongTensor(self.tokenizer.text_to_ids(critique_prompt_str))
        prompt_len = len(prompt_tokens)
        
        chosen_resp_text = crit_chosen + self.model.cfg.data.chat_prompt_tokens.end_of_turn
        chosen_resp_tokens = torch.LongTensor(self.tokenizer.text_to_ids(chosen_resp_text))
        reject_resp_text = crit_reject + self.model.cfg.data.chat_prompt_tokens.end_of_turn
        reject_resp_tokens = torch.LongTensor(self.tokenizer.text_to_ids(reject_resp_text))
        
        # 1 x max_len tensor
        chosen_prompt_len = prompt_len
        chosen_tokens = torch.cat([prompt_tokens, chosen_resp_tokens], dim=0)
        chosen_gen_len = len(chosen_tokens)
        chosen_score = 1.0
        
        reject_prompt_len = prompt_len
        reject_tokens = torch.cat([prompt_tokens, reject_resp_tokens], dim=0)
        reject_gen_len = len(reject_tokens)
        reject_score = 0.0
        
        if chosen_gen_len > self.model.cfg.encoder_seq_length or reject_gen_len > self.model.cfg.encoder_seq_length:
            return None
        else:
            return {
                        "chosen_tokens": chosen_tokens,
                        "chosen_prompt_len": chosen_prompt_len,
                        "chosen_gen_len": chosen_gen_len,
                        "chosen_score": chosen_score,
                        "reject_tokens": reject_tokens,
                        "reject_prompt_len": reject_prompt_len,
                        "reject_gen_len": reject_gen_len,
                        "reject_score": reject_score,
                        "bad_sample": False,
                        "bad_ends": 0,
                    }
        

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations and ref policy log probs"""
        iter_dataloader = iter(dataloader)
        #iter_alt = None
        #if self.alt_dataloader is not None:
        #    iter_alt = iter(self.alt_dataloader)
        buffer, meta_buffer = [], []
        done = False
        samples_main = samples_seen = samples_replaced = 0
        while not done:
            try:
                batches = next(iter_dataloader)
                if self.alt_iter is not None and samples_main % 2 == 1:
                    batch_sft = self.get_batch_from_alt_dataloader()
                    yield batch_sft
                batch = self_revising_custom_collate(batches, eos_id=self.model.tokenizer.eos_id)
            except StopIteration:
                done = True
            else:
                buffer.append(batch)
                samples_main += 1

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
                    
                    
                    candidate_responses_with_critiques = []
                    # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                    gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(buffer)

                    # Transform into batch of LLM-as-judge template samples for reward scoring
                    orig_prompts_and_responses, critique_buffer = [], []
                    verifier_args = [x for b in buffer for x in b["verifier_args"]]
                    for t, s, e, vargs in zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist(), verifier_args):
                        prompt_orig = self.tokenizer.ids_to_text(t[:s].tolist())
                        response_orig = self.tokenizer.ids_to_text(t[s:e].tolist())
                        if self.cfg.trt_llm.get("model_type", "gptnext").lower() == "llama":
                            prompt = prompt_orig.replace(
                                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", ""
                            )
                            response = response_orig.replace("<|eot_id|>", "").strip()
                        else:
                            prompt = prompt_orig.replace("<extra_id_0>System\n\n", "")
                            response = response_orig.replace("\n<extra_id_1>", "").strip()
                        
                        #prompt, response, last_prompt = self.normalise_prompt(
                        #    self.tokenizer.ids_to_text(t[:s].tolist()),
                        #    self.tokenizer.ids_to_text(t[s:e].tolist()),
                        #    buffer[0]["dataset_mask"],
                        #)
                        if vargs is None:
                            critique_prompt_str = self.critique_template_fn(prompt=prompt, response=response)
                        elif "generated_solution" in vargs:
                            critique_prompt_str = self.critique_verified_math_template_fn(prompt=prompt, response=response, expected_answer=vargs["generated_solution"])
                        elif "instruction_id_list" in vargs:
                            v_correctness, v_details, v_status = self.instruction_following_rewards(prompt, response, vargs)
                            if v_status:
                                critique_prompt_str = self.critique_verified_ifeval_template_fn(prompt=prompt, response=response, correctness=v_correctness, v_details=v_details)
                            else:
                                critique_prompt_str = self.critique_template_fn(prompt=prompt, response=response)
                        critique_prompt = self.tokenizer.text_to_ids(critique_prompt_str)
                        if len(critique_prompt) > self.model.cfg.data.train_ds.max_seq_length:
                            critique_prompt = self.truncate_critique(prompt, response, buffer, vargs)

                        assert len(critique_prompt) <= (
                            self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                        ), f"truncation of critique template failed [ {len(critique_prompt)} ]: {critique_prompt_str}"
                        
                        critique_buffer.append(
                            {
                                "prompt_lengths": torch.LongTensor([len(critique_prompt)]),
                                "prompts_only": torch.LongTensor(critique_prompt).unsqueeze(0),
                            }
                        )
                        orig_prompts_and_responses.append( (prompt, response) )

                    
                    critiques_with_revisions_list = [
                        [] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))
                    ]
                    for _ in range(self.num_critiques_to_gen):
                        critiques_as_str = self.get_tagged_responses(critique_buffer, regex_fn=None)
                    
                        revise_buffer, bad_idxs = [], []
                        for idx, ((prompt, response), critique) in enumerate(zip(orig_prompts_and_responses, critiques_as_str)):
                            if critique is None or len(critique) == 0:
                                critique = "I have no critique."
                                bad_idxs.append(idx)
                            revise_prompt_str = self.revise_template_fn(prompt=prompt, response=response, num_annotators="1", s_or_not="", critique=critique)
                            revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                            if len(revise_prompt) > self.model.cfg.data.train_ds.max_seq_length:
                                revise_prompt = self.truncate_revise(prompt, response, buffer, critique, bad_idxs, idx)
                        
                            assert len(revise_prompt) <= (
                                    self.model.cfg.encoder_seq_length - self.max_gen_seq_len - 8
                                ), f"truncation of revise template failed [ {len(revise_prompt)} ]: {revise_prompt_str}"
                            
                            revise_buffer.append(
                                {
                                    "prompt_lengths": torch.LongTensor([len(revise_prompt)]),
                                    "prompts_only": torch.LongTensor(revise_prompt).unsqueeze(0),
                                }
                            )
                        
                        if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
                            print("**** BAD_IDXS: ", bad_idxs)
                        revisions_as_str = self.get_tagged_responses(revise_buffer, regex_fn=None)
                        revisions_as_str = [(None if i in bad_idxs else x) for i,x in enumerate(revisions_as_str)]
                        
                        for idx, (critique_as_str, revised_resp_str) in enumerate(zip(critiques_as_str, revisions_as_str)):
                            critiques_with_revisions_list[idx].append( (critique_as_str, revised_resp_str) )
                    
                    #elo_score_buffer = []
                    chosen_and_reject_responses = []
                    for idx in range(len(critiques_with_revisions_list)):
                        orig_prompt, orig_response = orig_prompts_and_responses[idx]
                        individual_select_buffer = []
                        candidate_responses = [orig_response] + [x[-1] for x in critiques_with_revisions_list[idx]]
                        candidate_critiques = [x[0] for x in critiques_with_revisions_list[idx]]
                        combinations = list(itertools.combinations(candidate_responses, 2))
                        for resp_A, resp_B in combinations:
                            #resp_A = resp_A_t[-1]
                            #resp_B = resp_B_t[-1]
                            if resp_A is None:
                                resp_A = "NULL"
                            if resp_B is None:
                                resp_B = "NULL"
                            select_prompt_str_AB = self.select_template_fn(orig_prompt=orig_prompt, response_A=resp_A, response_B=resp_B)
                            select_prompt_AB = self.tokenizer.text_to_ids(select_prompt_str_AB)
                            if len(select_prompt_AB) > self.model.cfg.data.train_ds.max_seq_length:
                                select_prompt_AB = self.truncate_select(orig_prompt, buffer, resp_A, resp_B)
                            select_prompt_str_BA = self.select_template_fn(orig_prompt=orig_prompt, response_A=resp_B, response_B=resp_A)
                            select_prompt_BA = self.tokenizer.text_to_ids(select_prompt_str_BA)
                            if len(select_prompt_BA) > self.model.cfg.data.train_ds.max_seq_length:
                                select_prompt_BA = self.truncate_select(orig_prompt, buffer, resp_B, resp_A)
                            
                            individual_select_buffer.append(
                                {
                                    "prompt_lengths": torch.LongTensor([len(select_prompt_AB)]),
                                    "prompts_only": torch.LongTensor(select_prompt_AB).unsqueeze(0),
                                    #"resp_dict": {"A": resp_A, "B": resp_B},
                                }
                            )
                            individual_select_buffer.append(
                                {
                                    "prompt_lengths": torch.LongTensor([len(select_prompt_BA)]),
                                    "prompts_only": torch.LongTensor(select_prompt_BA).unsqueeze(0),
                                    #"resp_dict": {"A": resp_A, "B": resp_B},
                                }
                            )
                        selections_as_str = []
                        for batch in divide_chunks(individual_select_buffer, self.rollout_micro_batch_size):
                            batch_output = self.get_tagged_responses(batch, regex_fn=self.select_regex_fn)
                            selections_as_str.extend(batch_output)
                        elo_scores = self.get_elo_scores(len(candidate_responses), selections_as_str)
                        #elo_score_buffer.append( (elo_scores, candidate_responses) )
                        if np.sum(elo_scores == elo_scores.max()) > 1:
                            chosen_idx = torch.multinomial(torch.FloatTensor(elo_scores == elo_scores.max()), num_samples=1, replacement=False, generator=self.rng_generator).item()
                        else:
                            chosen_idx = np.argmax(elo_scores)
                        #if np.sum(elo_scores == elo_scores.min()) > 1:
                        #    reject_idx = torch.multinomial(torch.FloatTensor(elo_scores == elo_scores.min()), num_samples=1, replacement=False, generator=self.rng_generator).item()
                        #else:
                        #    reject_idx = np.argmin(elo_scores)
                        if sum(elo_scores != elo_scores[chosen_idx]) == 0:
                            reject_idx = torch.multinomial(torch.FloatTensor(elo_scores == elo_scores.min()), num_samples=1, replacement=False, generator=self.rng_generator).item()
                        else:
                            reject_idx = torch.multinomial(torch.FloatTensor(elo_scores != elo_scores[chosen_idx]), num_samples=1, replacement=False, generator=self.rng_generator).item()
                        chosen_and_reject_responses.append( {"chosen": candidate_responses[chosen_idx], "reject": candidate_responses[reject_idx], "chosen_is_orig": chosen_idx == 0} )
                        
                        if self.use_meta_critiques and len(meta_buffer) < self.model.cfg.global_batch_size * 30:
                            cand_scores_with_idx = [(i,c) for i,c in enumerate(elo_scores[1:])]
                            for crit_A, crit_B in itertools.combinations(sorted(cand_scores_with_idx, key=lambda kk: kk[-1], reverse=True), 2):
                                if crit_A[-1] > crit_B[-1]:
                                    crit_chosen = candidate_critiques[crit_A[0]]
                                    crit_reject = candidate_critiques[crit_B[0]]
                                    meta_sample = self.create_meta_sample(orig_prompt, orig_response, crit_chosen, crit_reject)
                                    if meta_sample is not None:
                                        meta_buffer.append(meta_sample)
                            

                    for idx, (t, s, e, end, op_and_r, critique_and_revise, select) in enumerate(
                        zip(
                            gen_tokens_buf,
                            gen_prompt_lengths_buf.tolist(),
                            gen_lengths_buf.tolist(),
                            is_end.tolist(),
                            orig_prompts_and_responses,
                            critiques_with_revisions_list,
                            chosen_and_reject_responses,
                        )
                    ):
                        candidate_responses_with_critiques.append( (t, s, e, op_and_r[0], op_and_r[-1], [x[0] for x in critique_and_revise], [x[-1] for x in critique_and_revise], select, end) )

                    final_buffer = []
                    # now we need to pick the chosen/rejected
                    for cand_selected in candidate_responses_with_critiques:
                        bad_sample = False
                        
                        # DPO format
                        prompt_len = cand_selected[1]
                        prompt_tokens = cand_selected[0][:prompt_len]
                        
                        chosen_resp_str = (cand_selected[-2]["chosen"] if cand_selected[-2] is not None else "NULL") + self.model.cfg.data.chat_prompt_tokens.end_of_turn
                        chosen_resp_tokens = torch.LongTensor(self.tokenizer.text_to_ids(chosen_resp_str)).to(prompt_tokens.device)
                        reject_resp_str = (cand_selected[-2]["reject"] if cand_selected[-2] is not None else "NULL") + self.model.cfg.data.chat_prompt_tokens.end_of_turn
                        reject_resp_tokens = torch.LongTensor(self.tokenizer.text_to_ids(reject_resp_str)).to(prompt_tokens.device)

                        # 1 x max_len tensor
                        chosen_prompt_len = prompt_len
                        chosen_tokens = torch.cat([prompt_tokens, chosen_resp_tokens], dim=0)
                        chosen_gen_len = len(chosen_tokens)
                        chosen_score = 1.0
                        
                        reject_prompt_len = prompt_len
                        reject_tokens = torch.cat([prompt_tokens, reject_resp_tokens], dim=0)
                        reject_gen_len = len(reject_tokens)
                        reject_score = 0.0
                        
                        bad_ends = sum(~np.array([cand_selected[-1], cand_selected[-1]]))

                        if (cand_selected[-2]["chosen_is_orig"]) or torch.equal(chosen_tokens, reject_tokens) or (all([b is None for b in cand_selected[-3]])) or (cand_selected[-2] is None) or (all([b is None for b in cand_selected[-4]])):
                            bad_sample = True

                        samples_seen += 1
                        
                        if (
                            self.use_meta_critiques
                            and (
                                (bad_ends > 0 or bad_sample)
                                #or (
                                #    self.rng_generator.random()
                                #    <= self.meta_judge_pcnt - (samples_replaced / samples_seen)
                                #)
                            )
                            and len(meta_buffer) > 0
                        ):
                            #final_buffer.append(meta_buffer.pop(0))
                            # if you want to pop a random element instead, uncomment the below
                            final_buffer.append(meta_buffer.pop(torch.randint(0, len(meta_buffer), size=(1,), generator=self.rng_generator).item()))

                            samples_replaced += 1
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
                        
                        if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
                            print(f"CHOSEN_TOKENS: {self.tokenizer.ids_to_text(chosen_tokens.tolist())}  REJECT_TOKENS: {self.tokenizer.ids_to_text(reject_tokens.tolist())}")
                
                    self.model.finish_inference()
                    if self.use_trtllm_generation:
                        self.trtllm_generate.free()

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
    
    def get_elo_scores(self, p, meta_reward_scores):
        SCALE = 400
        INIT_RATING = 1000
        
        players = list(range(p))
        Bm = itertools.combinations(players, 2)
        ptbl_a_win = np.zeros([p, p])
        ptbl_b_win = np.zeros([p, p])
        ptbl_tie = np.zeros([p, p])
        for (m_a, m_b), (ab, ba) in zip(Bm, [tuple(x) for x in divide_chunks(meta_reward_scores, 2)]):
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

    def set_rho_by_iteration(self, iteration):
        if isinstance(self.spin_config["length_control"], (float, int)):
            return
        elif isinstance(self.spin_config["length_control"], list):
            assert iteration < len(
                self.spin_config["length_control"]
            ), f"iteration [ {iteration} ] is out of bounds for length_control schedule {self.spin_config['length_control']}"

            self.rho = self.spin_config["length_control"][iteration]

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs

    @property
    def iteration(self):
        return (self.step // self.num_steps_per_epoch) // self.cfg.max_epochs
