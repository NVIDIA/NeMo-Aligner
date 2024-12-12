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
import time

import pytest
import torch
from megatron.inference.text_generation import beam_search_and_post_process, generate_and_post_process
from omegaconf import OmegaConf

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp
from nemo_aligner.utils.text_generation_utils import TrackLengthGPTModelTextGenerationStrategy
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM


@pytest.mark.mpi
def test_trtllm_matches_mcore_inf(dummy_actor_gpt_model_with_pp):
    # torch.distributed.breakpoint()
    batch_size = 4
    max_seq_len = 100
    prompt_tokens = torch.ones((batch_size, max_seq_len), dtype=torch.int64, device="cuda")
    prompt_lengths = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="cuda")

    cfg_max_input_len = (
        max_seq_len // 2
    )  # in PPO this is set to: dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length // 2
    cfg_max_output_len = (
        max_seq_len // 2
    )  # in PPO this is set to: dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length // 2

    # Sanity check that checkpoint doesn't have an unusually small seq len
    assert max_seq_len <= dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length

    ###########################
    # Megatron-Core Inference #
    ###########################
    # TODO re-enable
    ##assert dummy_actor_gpt_model_with_pp.model.config.flash_decode
    class generation_args:
        max_position_embeddings = dummy_actor_gpt_model_with_pp.cfg.max_position_embeddings
        # TODO: appears to be a soft check for total num of input tokens, for now don't check
        max_tokens_to_oom = float("inf")
        inference_max_seq_length = max_seq_len
        enable_cuda_graph = dummy_actor_gpt_model_with_pp.model.config.enable_cuda_graph
        eos_id = dummy_actor_gpt_model_with_pp.tokenizer.eos_id
        inference_batch_times_seqlen_threshold = -1
        # TODO: Look into whether it's okay to just use the tokenizer's args (is it padded?)
        padded_vocab_size = dummy_actor_gpt_model_with_pp.tokenizer.vocab_size

    # from megatron.inference.text_generation.forward_step import ForwardStep
    # from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    # class ForwardStepSideStepNeMosForward(ForwardStep):
    #    def _forward(self, tokens, position_ids, attention_mask):
    #        assert isinstance(self.model, MegatronGPTModel), type(self.model)
    #        from inspect import signature as sig
    #        torch.distributed.breakpoint()
    #        return self.model(tokens, position_ids, attention_mask, inference_params=self.inference_params)

    start = time.time()
    mcore_result = generate_and_post_process(
        model=dummy_actor_gpt_model_with_pp.model,
        # forward_step=ForwardStepSideStepNeMosForward,
        prompts=(prompt_tokens, prompt_lengths),
        tokens_to_generate=cfg_max_output_len,  # Must be 0 if passing prompt_tokens + prompt_lengths
        prevent_newline_after_colon=False,  # Turning this on requires a global tokenizer, so we leave it off
        top_k_sampling=1,  # == greedy
        generation_args=generation_args,
    )
    # Mcore inference returns None if not on the first PP rank
    if mcore_result is not None:
        _, _, _, mcore_output_ids = mcore_result
        mcore_output_ids = torch.tensor(mcore_output_ids, dtype=torch.long, device="cuda")
    else:
        mcore_output_ids = None
    mcore_output_ids = broadcast_2d_tensor_within_pp(mcore_output_ids, dtype=torch.long, from_last=False)
    print(f"{mcore_output_ids=}")
    first_mcore_generate_sec = time.time() - start
    start = time.time()
    _ = generate_and_post_process(
        model=dummy_actor_gpt_model_with_pp.model,
        # forward_step=ForwardStepSideStepNeMosForward,
        prompts=(prompt_tokens, prompt_lengths),
        tokens_to_generate=cfg_max_output_len,  # Must be 0 if passing prompt_tokens + prompt_lengths
        prevent_newline_after_colon=False,  # Turning this on requires a global tokenizer, so we leave it off
        generation_args=generation_args,
    )
    second_mcore_generate_sec = time.time() - start

    ##########
    # TRTLLM #
    ##########
    trtllm_generate = GPTGenerateTRTLLM(
        model_cfg=dummy_actor_gpt_model_with_pp.cfg,
        end_strings=["<extra_id_1>"],
        tokenizer=dummy_actor_gpt_model_with_pp.tokenizer,
        max_input_len=cfg_max_input_len,
        max_generation_length=cfg_max_output_len,
        use_greedy=True,
    )
    trtllm_generate.refit(dummy_actor_gpt_model_with_pp)

    start = time.time()
    output_ids, response_lengths = trtllm_generate._generate([prompt_tokens, prompt_lengths])
    trtllm_generate_sec = time.time() - start
    max_length = response_lengths.max().item()
    # torch.distributed.breakpoint()

    #################
    # Nemo generate #
    #################
    # length argument for autoregressive sampling
    length_params = {
        # max length means max amount of tokens to generate
        "max_length": cfg_max_output_len,  # Set to ${int_div:${model.encoder_seq_length}, 2} in ppo
        "min_length": 1,
    }
    # sampling parameters for generation
    sampling_params = {
        "use_greedy": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "add_BOS": False,
        "all_probs": False,
        "compute_logprob": False,
        # will be used in NeMo version > 1.20.0
        # keeping it for now
        "end_strings": ["<|endoftext|>", "<extra_id_1>"],
    }
    strategy = TrackLengthGPTModelTextGenerationStrategy(
        model=dummy_actor_gpt_model_with_pp, context_lengths=prompt_lengths, max_length=length_params["max_length"]
    )
    start = time.time()
    actor_output = dummy_actor_gpt_model_with_pp.generate(
        inputs=(prompt_tokens, prompt_lengths),
        length_params=length_params,
        sampling_params=sampling_params,
        strategy=strategy,
    )
    nemo_generate_sec = time.time() - start
    response_tokens = torch.cuda.LongTensor(actor_output["token_ids"]) if actor_output else None
    response_tokens = broadcast_2d_tensor_within_pp(response_tokens, dtype=torch.long)
    response_lengths = strategy.get_lengths()

    print(f"{(first_mcore_generate_sec, second_mcore_generate_sec, trtllm_generate_sec, nemo_generate_sec)=}")
    print(f"[NEMO][rank={torch.distributed.get_rank()}]   {response_tokens[0][:12].cpu().tolist()}")
    print(f"[MCORE][rank={torch.distributed.get_rank()}]  {mcore_output_ids[0][:12].cpu().tolist()}")
    print(f"[TRTLLM][rank={torch.distributed.get_rank()}] {output_ids[0][:12].cpu().tolist()}")
