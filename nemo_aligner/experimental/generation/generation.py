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

import json
import os
import pickle
from collections import defaultdict
from functools import partial
from statistics import mean
from textwrap import dedent

import numpy as np
import pandas as pd
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
from nemo_aligner.utils.distributed import SyncTimer, broadcast_2d_tensor_within_pp
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import clip_gradients, set_eval
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


def generate_sft_custom_collate(batch, eos_id):
    context_ids = [item["context_ids"] for item in batch]
    context_lengths = torch.LongTensor([len(x) for x in context_ids])

    context_ids = torch.nn.utils.rnn.pad_sequence(context_ids, batch_first=True, padding_value=eos_id)

    output = {
        "prompts_only": context_ids,
        "prompt_lengths": context_lengths,
        "metadata": [
            (x["metadata"] if "metadata" in x else None) for x in batch
        ],
    }

    return output


class GenerationTrainer:
    """Trainer class for running generation in aligner
    """

    def __init__(
        self, cfg: DictConfig, model, train_dataloader, logger, ckpt_callback, run_timer, exp_manager,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.cfg = cfg

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.step = 0
        self.consumed_samples = 0

        self.ckpt_callback = ckpt_callback

        assert self.cfg.max_epochs == 1, "`generation.max_epochs` must be equal to 1 for generation"

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(self.train_dataloader.batch_sampler)

        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        self.num_responses_to_gen = self.model.cfg.generation.num_responses_to_gen
        self.length_params = OmegaConf.to_container(self.model.cfg.generation.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.model.cfg.generation.sampling_params, resolve=True)
        self.max_gen_seq_len = self.length_params["max_length"]
        dp_batch_size = self.model.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
        # model_parallel.source_rank ?
        # storage for generated responses which we want to save
        if torch.distributed.get_rank() == 0:
            os.makedirs(os.path.join(exp_manager.explicit_log_dir, "generations"), exist_ok=True)
            self.generations_fh = open(
                os.path.join(exp_manager.explicit_log_dir, "generations", "generations.jsonl"),
                "a",
                encoding="utf_8",
                newline="\n",
            )
        else:
            self.generations_fh = None

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
                generation_batch_size=dp_batch_size,
                use_greedy=self.sampling_params.get("use_greedy", False),
                trt_model_type=self.cfg.trt_llm.get("model_type", "gptnext"),
                seed=self.model.cfg.get("seed", None),
                unload_engine_train=self.cfg.trt_llm.get("unload_engine_train", False),
                reshard_model=False,
            )

    @torch.no_grad()
    def get_generations(self, list_of_batches):
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

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_valid.cpu()

    def generate(self):
        self.model._reset_activation_checkpointing_args()
        self.model._reset_sequence_parallelism_args()
        set_eval(self.model)

        if self.use_trtllm_generation:
            self.trtllm_generate.refit(self.model)
            clear_memory()

        self.run_timer.start_time()

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
                desc="Generating steps",
            )

            for _, global_batch in zip(loop_iter, global_pbar):

                dp_group = parallel_state.get_data_parallel_group()

                gen_tokens = global_batch["prompt_and_response"]
                prompt_lens = global_batch["prompt_lens"]
                gen_lens = global_batch["gen_lens"]
                valids = global_batch["valids"]
                metadata = global_batch["metadata"] if "metadata" in global_batch else None

                gen_tokens_list = [torch.zeros_like(gen_tokens) for _ in range(dp_group.size())]
                prompt_lens_list = [torch.zeros_like(prompt_lens) for _ in range(dp_group.size())]
                gen_lens_list = [torch.zeros_like(gen_lens) for _ in range(dp_group.size())]
                valids_list = [torch.zeros_like(valids) for _ in range(dp_group.size())]

                torch.distributed.all_gather(gen_tokens_list, gen_tokens, group=dp_group)
                torch.distributed.all_gather(prompt_lens_list, prompt_lens, group=dp_group)
                torch.distributed.all_gather(gen_lens_list, gen_lens, group=dp_group)
                torch.distributed.all_gather(valids_list, valids, group=dp_group)
                
                if metadata is not None:
                    local_dp_meta_size = torch.as_tensor([metadata.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
                    metadata_length_list = [torch.zeros_like(local_dp_meta_size) for _ in range(dp_group.size())]
                    torch.distributed.all_gather(metadata_length_list, local_dp_meta_size, group=dp_group)
                    metadata_list = [torch.zeros(x.item(), device=torch.cuda.current_device(), dtype=torch.int8) for x in metadata_length_list]
                    torch.distributed.all_gather(metadata_list, metadata, group=dp_group)
                    metadata_dict_list = [pickle.loads(x.cpu().numpy().tobytes()) for x in metadata_list]
                else:
                    metadata_dict_list = [None] * dp_group.size()

                self.consumed_samples += self.model.cfg.global_batch_size
                self.step += 1

                if torch.distributed.get_rank() == 0:
                    for t, s, e, v, meta in zip(gen_tokens_list, prompt_lens_list, gen_lens_list, valids_list, metadata_dict_list):
                        buffer = [[] for _ in range(t.shape[1])]
                        for idx in range(len(t)):
                            for pdx, (t_, s_, e_, v_) in enumerate(
                                zip(t[idx], s[idx].tolist(), e[idx].tolist(), v[idx].tolist())
                            ):
                                prompt = self.model.tokenizer.ids_to_text(t_[:s_].long().tolist())
                                response = self.model.tokenizer.ids_to_text(t_[s_:e_].long().tolist())
                                if v_:
                                    buffer[pdx].append((prompt, response))

                        for cand_list, meta_ in zip(buffer, meta):
                            if len(cand_list) == 0:
                                continue
                            assert all([cand_list[0][0] == x[0] for x in cand_list]), "all prompts in group not equal"
                            payload = {
                                "step": self.step,
                                "consumed_samples": self.consumed_samples,
                                "prompt": cand_list[0][0],
                                "metadata": meta_,
                                #"responses": list(set([x[1] for x in cand_list])),
                                "responses": [x[1] for x in cand_list],
                            }
                            self.generations_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    #self.generations_fh.flush()
                torch.distributed.barrier()

                run_time_exceeded = self.run_timer.is_finished()
                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

        self.logger.finalize()

        if torch.distributed.get_rank() == 0:
            self.generations_fh.close()

        if self.use_trtllm_generation:
            self.trtllm_generate.free()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

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
        """Augment dataloader with generations"""
        iter_dataloader = iter(dataloader)
        while True:
            try:
                batches = next(iter_dataloader)
                batch = generate_sft_custom_collate(batches, eos_id=self.model.tokenizer.eos_id)

                gen_tokens, prompt_lens, gen_lens, valids = [], [], [], []
                for _ in range(self.num_responses_to_gen):
                    # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                    gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations([batch])
                    # candidate_responses.append((gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end))
                    gen_tokens.append(gen_tokens_buf)
                    prompt_lens.append(gen_prompt_lengths_buf)
                    gen_lens.append(gen_lengths_buf)
                    valids.append(is_end)

                assert all([len(x) == self.num_responses_to_gen for x in [gen_tokens, prompt_lens, gen_lens, valids]])
                # if you want to pad to the global DP batch instead of model.cfg.encoder_seq_length you can uncomment this
                # max_seq_length = torch.tensor([x.size(-1) for x in gen_tokens], dtype=torch.float32, device=torch.cuda.current_device()).max().unsqueeze(0)
                # torch.distributed.all_reduce(max_seq_length, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group())
                # max_seq_length = int(max_seq_length)

                new_batch = {
                    "prompt_and_response": torch.stack(
                        [
                            batch_pad_to_fixed_len(
                                x, self.model.cfg.encoder_seq_length, pad_token=self.model.tokenizer.eos_id
                            )
                            for x in gen_tokens
                        ],
                        dim=0,
                    ).cuda(non_blocking=True),
                    "prompt_lens": torch.stack(prompt_lens, dim=0).cuda(non_blocking=True),
                    "gen_lens": torch.stack(gen_lens, dim=0).cuda(non_blocking=True),
                    "valids": torch.stack(valids, dim=0).cuda(non_blocking=True),
                }
                
                if batch['metadata'] is not None:
                    nbuffer = np.ascontiguousarray(np.frombuffer(pickle.dumps(batch['metadata']), dtype=np.int8))
                    new_batch['metadata'] = torch.from_numpy(np.copy(nbuffer)).contiguous().cuda(non_blocking=True)

                yield new_batch
                del new_batch, gen_tokens, prompt_lens, gen_lens, valids
            except StopIteration:
                break

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs
