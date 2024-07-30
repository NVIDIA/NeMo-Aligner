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
from nemo_aligner.utils.train_utils import clip_gradients, set_eval
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
def generate_sft_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    #input_ids = [item["input_ids"] for item in batch]
    #masks = [item["mask"] for item in batch]
    context_ids = [item["context_ids"] for item in batch]
    #answer_ids = [item["answer_ids"] for item in batch]
    context_lengths = torch.LongTensor([len(x) for x in context_ids])
    #combined_lengths = torch.LongTensor([len(x) for x in input_ids])

    #input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=eos_id)
    #masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    context_ids = torch.nn.utils.rnn.pad_sequence(context_ids, batch_first=True, padding_value=eos_id)
    #answer_ids = torch.nn.utils.rnn.pad_sequence(answer_ids, batch_first=True, padding_value=eos_id)

    output = {
        #"prompts_and_answers": input_ids,
        #"masks": masks,
        "prompts_only": context_ids,
        #"answers_only": answer_ids,
        "prompt_lengths": context_lengths,
        #"combined_lengths": combined_lengths,
    }

    return output

def eye(x):
    return x


class GenerationTrainer:
    """Trainer to coordinate Self-Rewarding training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        train_dataloader,
        logger,
        ckpt_callback,
        run_timer,
        exp_manager,
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

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(
            self.train_dataloader.batch_sampler, self.cfg.get("limit_train_batches", 1.0)
        )
        
        if isinstance(self.cfg.get("limit_train_batches", 1.0), int):
            self.train_dataloader.batch_sampler.total_samples = min(self.train_dataloader.batch_sampler.total_samples, self.cfg.limit_train_batches * self.train_dataloader.batch_sampler.global_batch_size)
            if hasattr(self.train_dataloader.batch_sampler, "last_batch_size"):
                self.train_dataloader.batch_sampler.last_batch_size = 0

        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        self.num_responses_to_gen = self.model.cfg.spin.num_responses_to_gen
        self.length_params = OmegaConf.to_container(self.model.cfg.spin.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.model.cfg.spin.sampling_params, resolve=True)
        self.max_gen_seq_len = self.length_params["max_length"]
        #dp_batch_size = self.model.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
        #assert (
        #    self.model.cfg.spin.rollout_micro_batch_size % dp_batch_size == 0
        #), f"rollout_micro_batch_size [{self.model.cfg.spin.rollout_micro_batch_size}] must be a multiple of GBS [{self.model.cfg.global_batch_size}] // DP [{parallel_state.get_data_parallel_world_size()}]"
        #self.rollout_micro_batch_size = self.model.cfg.spin.rollout_micro_batch_size
        #assert self.rollout_micro_batch_size > 0, "`rollout_micro_batch_size` must be > 0"
        
        # storage for generated responses which we want to save
        if torch.distributed.get_rank() == 0:
            self.generations_fh = open(os.path.join(exp_manager.explicit_log_dir, "generations.jsonl"), "a", encoding="utf_8", newline="\n")
        else:
            self.generations_fh = None
        
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
    
    @torch.no_grad()
    def get_generations(self, list_of_batches):
        #self.model.prepare_for_inference()
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
            actor_output = self.trtllm_generate.generate(inputs)
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

        #self.model.finish_inference()
        if self.use_trtllm_generation:
            self.trtllm_generate.free()

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_end.cpu()

    def generate(self):
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
        
        self.model._reset_activation_checkpointing_args()
        self.model._reset_sequence_parallelism_args()
        set_eval(self.model)

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
                
                gen_tokens_list = [torch.zeros_like(gen_tokens) for _ in range(dp_group.size())]
                prompt_lens_list = [torch.zeros_like(prompt_lens) for _ in range(dp_group.size())]
                gen_lens_list = [torch.zeros_like(gen_lens) for _ in range(dp_group.size())]
                valids_list = [torch.zeros_like(valids) for _ in range(dp_group.size())]
                
                torch.distributed.all_gather(gen_tokens_list, gen_tokens, group=dp_group)
                torch.distributed.all_gather(prompt_lens_list, prompt_lens, group=dp_group)
                torch.distributed.all_gather(gen_lens_list, gen_lens, group=dp_group)
                torch.distributed.all_gather(valids_list, valids, group=dp_group)
                
                self.consumed_samples += self.model.cfg.global_batch_size
                self.step += 1
                
                if torch.distributed.get_rank() == 0 and gen_tokens_list is not None:
                    for t, s, e, v in zip(gen_tokens_list, prompt_lens_list, gen_lens_list, valids_list):
                        buffer = [[] for _ in range(t.shape[1])]
                        for idx in range(len(t)):
                            for pdx, (t_, s_, e_, v_) in enumerate(zip(t[idx], s[idx].tolist(), e[idx].tolist(), v[idx].tolist())):
                                prompt = self.model.tokenizer.ids_to_text(t_[:s_].long().tolist())
                                response = self.model.tokenizer.ids_to_text(t_[s_:e_].long().tolist())
                                if v_:
                                    buffer[pdx].append((prompt, response))
                        
                        for cand_list in buffer:
                            if len(cand_list) == 0:
                                continue
                            assert all([cand_list[0][0] == x[0] for x in cand_list]), "all prompts in group not equal"
                            payload = {"step": self.step, "consumed_samples": self.consumed_samples, "prompt": cand_list[0][0], "responses": list(set([x[1] for x in cand_list]))}
                            self.generations_fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
                torch.distributed.barrier()
                
                run_time_exceeded = self.run_timer.is_finished()
                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

        self.logger.finalize()
        
        if torch.distributed.get_rank() == 0 and self.generations_fh is not None:
            self.generations_fh.close()
        
        if self.use_trtllm_generation:
            self.trtllm_generate.free(force_unload=True)

    def save(self, extra_candidates=None, is_train_end=False):
        # load back in the adam states if needed
        self.model.prepare_for_training()
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

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
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

    '''
    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations"""
        iter_dataloader = iter(dataloader)
        buffer = []
        done = False
        while not done:
            try:
                batches = next(iter_dataloader)
                batch = generate_sft_custom_collate(batches,
                                                      eos_id=self.model.tokenizer.eos_id,
                                                      reset_position_ids=self.model.cfg.data.get("reset_position_ids", False),
                                                      reset_attention_mask=self.model.cfg.data.get("reset_attention_mask", False),
                                                      eod_mask_loss=self.model.cfg.data.get("eod_mask_loss", False),
                                                      )
                print(f" rank [{torch.distributed.get_rank()}] *** RAW_BATCH_SHAPE: ", batch["prompts_only"].shape)
            except StopIteration:
                done = True
            else:
                buffer.append(batch)
            
            if (done and buffer) or sum(
                [len(b["prompts_only"]) for b in buffer]
            ) == self.rollout_micro_batch_size:
                
                #candidate_responses = [[] for _ in range(sum([len(b["prompt_lengths"]) for b in buffer]))]
                #candidate_responses = []
                gen_tokens, prompt_lens, gen_lens, valids = [], [], [], []
                for _ in range(self.num_responses_to_gen):
                    # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                    gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(buffer)
                    #candidate_responses.append((gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end))
                    gen_tokens.append(gen_tokens_buf)
                    prompt_lens.append(gen_prompt_lengths_buf)
                    gen_lens.append(gen_lengths_buf)
                    valids.append(is_end)
                    
                    #for idx, (t, s, e, end) in enumerate(zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist(), is_end.tolist())):
                    #    candidate_responses[idx].append((t,s,e,end))
                
                #final_batch = []
                #for cand_list in candidate_responses:
                #    prompt_only = cand_list[0][0][:cand_list[0][1]]
                #    resp_list = batch_pad_to_fixed_len([s[0][s[1]:s[2]] for s in cand_list], max_batch_len, pad_token=self.model.tokenizer.eos_id)
                
                new_batch = {
                    "prompt_and_response": torch.stack([batch_pad_to_fixed_len(x, self.model.cfg.encoder_seq_length, pad_token=self.model.tokenizer.eos_id) for x in gen_tokens], dim=0).cuda(non_blocking=True),
                    "prompt_lens": torch.stack(prompt_lens, dim=0).cuda(non_blocking=True),
                    "gen_lens": torch.stack(gen_lens, dim=0).cuda(non_blocking=True),
                    "valids": torch.stack(valids, dim=0).cuda(non_blocking=True),
                }
                
                yield new_batch
                del new_batch, gen_tokens, prompt_lens, gen_lens, valids

                buffer.clear()
    '''

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations"""
        iter_dataloader = iter(dataloader)
        while True:
            try:
                batches = next(iter_dataloader)
                batch = generate_sft_custom_collate(batches,
                                                      eos_id=self.model.tokenizer.eos_id,
                                                      reset_position_ids=self.model.cfg.data.get("reset_position_ids", False),
                                                      reset_attention_mask=self.model.cfg.data.get("reset_attention_mask", False),
                                                      eod_mask_loss=self.model.cfg.data.get("eod_mask_loss", False),
                                                      )
                #print(f" rank [{torch.distributed.get_rank()}] *** RAW_BATCH_SHAPE: ", batch["prompts_only"].shape)
                
                gen_tokens, prompt_lens, gen_lens, valids = [], [], [], []
                for _ in range(self.num_responses_to_gen):
                    # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
                    gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations([batch])
                    #candidate_responses.append((gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end))
                    gen_tokens.append(gen_tokens_buf)
                    prompt_lens.append(gen_prompt_lengths_buf)
                    gen_lens.append(gen_lengths_buf)
                    valids.append(is_end)
                
                # if you want to pad to the global DP batch instead of model.cfg.encoder_seq_length you can uncomment this
                #max_seq_length = torch.tensor([x.size(-1) for x in gen_tokens], dtype=torch.float32, device=torch.cuda.current_device()).max().unsqueeze(0)
                #torch.distributed.all_reduce(max_seq_length, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group())
                #max_seq_length = int(max_seq_length)
                
                new_batch = {
                    "prompt_and_response": torch.stack([batch_pad_to_fixed_len(x, self.model.cfg.encoder_seq_length, pad_token=self.model.tokenizer.eos_id) for x in gen_tokens], dim=0).cuda(non_blocking=True),
                    "prompt_lens": torch.stack(prompt_lens, dim=0).cuda(non_blocking=True),
                    "gen_lens": torch.stack(gen_lens, dim=0).cuda(non_blocking=True),
                    "valids": torch.stack(valids, dim=0).cuda(non_blocking=True),
                }
                
                yield new_batch
                del new_batch, gen_tokens, prompt_lens, gen_lens, valids
            except StopIteration:
                break

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs
