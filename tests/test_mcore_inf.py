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
import torch.distributed
from megatron.inference.text_generation import beam_search_and_post_process, generate_and_post_process

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp
from nemo_aligner.utils.generation import MegatronGenerator, NemoGenerator, TRTLLMGenerator
from nemo_aligner.utils.text_generation_utils import TrackLengthGPTModelTextGenerationStrategy
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM


def assert_sequence_prefix_almost_match(ref: list[int], candidate: list[int], proportion: float = 1.0):
    total_ref_len = len(ref)
    total_matches = 0
    for r, c in zip(ref, candidate):
        if r != c:
            break
        total_matches += 1
    match_ratio = total_matches / total_ref_len
    assert (
        total_matches / total_ref_len >= proportion
    ), f"Expected {proportion * 100:0.1f}% of the sequences to match, but only got {total_matches}/{total_ref_len} = {match_ratio * 100:.1f}%"


@pytest.mark.mpi
def test_scratch_all_inf_frameworks_greedy_generations_match(dummy_actor_gpt_model_with_pp):
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
    assert dummy_actor_gpt_model_with_pp.model.config.flash_decode
    assert dummy_actor_gpt_model_with_pp.model.config.enable_cuda_graph

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

    start = time.time()
    mcore_result = generate_and_post_process(
        model=dummy_actor_gpt_model_with_pp.model,
        prompts=(prompt_tokens, prompt_lengths),
        tokens_to_generate=cfg_max_output_len,  # Must be 0 if passing prompt_tokens + prompt_lengths
        prevent_newline_after_colon=False,  # Turning this on requires a global tokenizer, so we leave it off
        top_k_sampling=1,  # 1 == greedy
        generation_args=generation_args,
    )
    # Mcore inference returns None if not on the first PP rank
    if mcore_result is not None:
        mcore_output_ids, mcore_lengths = mcore_result.tokens, mcore_result.lengths
        mcore_output_ids = torch.tensor(mcore_output_ids, dtype=torch.long, device="cuda")
        mcore_lengths = torch.tensor(mcore_lengths, dtype=torch.long, device="cuda")
    else:
        mcore_output_ids = None
    mcore_output_ids = broadcast_2d_tensor_within_pp(mcore_output_ids, dtype=torch.long, from_last=False)
    mcore_lengths = broadcast_2d_tensor_within_pp(mcore_lengths, dtype=torch.long, from_last=False)
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
    trtllm_output_ids, trtllm_lengths = trtllm_generate._generate((prompt_tokens, prompt_lengths))
    trtllm_generate_sec = time.time() - start

    #################
    # Nemo generate #
    #################
    # Perf can be affected if previously run with cuda graphs. This allows us to
    # run nemo-generate after mcore-generate, but this surgery on the model is
    # just for correctness. For perf, a new model should be generated with
    # enable_cuda_graph = False.
    dummy_actor_gpt_model_with_pp.model.config.enable_cuda_graph = False
    for m in dummy_actor_gpt_model_with_pp.model.modules():
        if hasattr(m, "cudagraph_manager"):
            del m.cudagraph_manager

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
    nemo_output_ids = torch.cuda.LongTensor(actor_output["token_ids"]) if actor_output else None
    nemo_output_ids = broadcast_2d_tensor_within_pp(nemo_output_ids, dtype=torch.long)
    nemo_lengths = strategy.get_lengths()

    print(f"{(first_mcore_generate_sec, second_mcore_generate_sec, trtllm_generate_sec, nemo_generate_sec)=}")
    first_n = 30
    print(
        f"[NEMO][rank={torch.distributed.get_rank()}]   token_ids[0][:{first_n}]{nemo_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(
        f"[MCORE][rank={torch.distributed.get_rank()}]  token_ids[0][:{first_n}]{mcore_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(
        f"[TRTLLM][rank={torch.distributed.get_rank()}] token_ids[0][:{first_n}]{trtllm_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(f"[NEMO][rank={torch.distributed.get_rank()}]   lengths={nemo_lengths.cpu().tolist()}")
    print(f"[MCORE][rank={torch.distributed.get_rank()}]  lengths={mcore_lengths.cpu().tolist()}")
    print(f"[TRTLLM][rank={torch.distributed.get_rank()}] lengths={trtllm_lengths.cpu().tolist()}")

    for b_i, prompt_and_resp_len in enumerate(nemo_lengths):
        assert_sequence_prefix_almost_match(
            nemo_output_ids[b_i][:prompt_and_resp_len], mcore_output_ids[b_i][:prompt_and_resp_len], proportion=1.0
        )
        # TRTLLM usually "almost" matches for a dummy model (around 93% match)
        assert_sequence_prefix_almost_match(
            nemo_output_ids[b_i][:prompt_and_resp_len], trtllm_output_ids[b_i][:prompt_and_resp_len], proportion=0.9
        )

    # Cuda graphs should make the second call faster
    assert first_mcore_generate_sec > second_mcore_generate_sec
    torch.distributed.breakpoint()


@pytest.mark.mpi
def test_all_inf_frameworks_greedy_generations_match(dummy_actor_gpt_model_with_pp):
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

    generator_kwargs = {
        "model": dummy_actor_gpt_model_with_pp,
        "end_strings": ["<|endoftext|>", "<extra_id_1>"],
        "sample_temperature": 1.0,
        "sample_top_k": 1,
        "sample_top_p": 0.0,
        "repetition_penalty": 1.0,
        "max_input_len": cfg_max_input_len,
        "max_generation_length": cfg_max_output_len,
        "generation_batch_size": 4,
        "use_greedy": True,
        "seed": 1,
        "unload_engine_train": False,
        "reshard_model": False,
        "refit_model": False,
    }
    mcore_generator = MegatronGenerator(**generator_kwargs)
    nemo_generator = NemoGenerator(**generator_kwargs)
    trtllm_generator = TRTLLMGenerator(**generator_kwargs)

    ###########################
    # Megatron-Core Inference #
    ###########################
    start = time.time()
    mcore_result = mcore_generator.generate((prompt_tokens, prompt_lengths))
    first_mcore_generate_sec = time.time() - start
    mcore_output_ids, mcore_lengths = mcore_result.response_tokens, mcore_result.response_lengths

    start = time.time()
    _ = mcore_generator.generate((prompt_tokens, prompt_lengths))
    second_mcore_generate_sec = time.time() - start

    ##########
    # TRTLLM #
    ##########
    trtllm_generator.refit()

    start = time.time()
    trtllm_result = trtllm_generator.generate((prompt_tokens, prompt_lengths))
    trtllm_output_ids, trtllm_lengths = trtllm_result.response_tokens, trtllm_result.response_lengths

    trtllm_generate_sec = time.time() - start

    #################
    # Nemo generate #
    #################
    # Perf can be affected if previously run with cuda graphs. This allows us to
    # run nemo-generate after mcore-generate, but this surgery on the model is
    # just for correctness. For perf, a new model should be generated with
    # enable_cuda_graph = False.
    dummy_actor_gpt_model_with_pp.model.config.enable_cuda_graph = False
    for m in dummy_actor_gpt_model_with_pp.model.modules():
        if hasattr(m, "cudagraph_manager"):
            del m.cudagraph_manager

    start = time.time()
    nemo_result = nemo_generator.generate((prompt_tokens, prompt_lengths))
    nemo_generate_sec = time.time() - start
    nemo_output_ids, nemo_lengths = nemo_result.response_tokens, nemo_result.response_lengths

    print(f"{(first_mcore_generate_sec, second_mcore_generate_sec, trtllm_generate_sec, nemo_generate_sec)=}")
    first_n = 30
    print(
        f"[NEMO][rank={torch.distributed.get_rank()}]   token_ids[0][:{first_n}]{nemo_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(
        f"[MCORE][rank={torch.distributed.get_rank()}]  token_ids[0][:{first_n}]{mcore_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(
        f"[TRTLLM][rank={torch.distributed.get_rank()}] token_ids[0][:{first_n}]{trtllm_output_ids[0][:first_n].cpu().tolist()}"
    )
    print(f"[NEMO][rank={torch.distributed.get_rank()}]   lengths={nemo_lengths.cpu().tolist()}")
    print(f"[MCORE][rank={torch.distributed.get_rank()}]  lengths={mcore_lengths.cpu().tolist()}")
    print(f"[TRTLLM][rank={torch.distributed.get_rank()}] lengths={trtllm_lengths.cpu().tolist()}")

    for b_i, prompt_and_resp_len in enumerate(nemo_lengths):
        assert_sequence_prefix_almost_match(
            nemo_output_ids[b_i][:prompt_and_resp_len], mcore_output_ids[b_i][:prompt_and_resp_len], proportion=1.0
        )
        # TRTLLM usually "almost" matches for a dummy model (around 93% match)
        assert_sequence_prefix_almost_match(
            nemo_output_ids[b_i][:prompt_and_resp_len], trtllm_output_ids[b_i][:prompt_and_resp_len], proportion=0.9
        )

    # Cuda graphs should make the second call faster
    assert first_mcore_generate_sec > second_mcore_generate_sec
