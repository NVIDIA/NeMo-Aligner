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
import pytest
import torch

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM


@pytest.mark.mpi
def test_trtllm_does_not_insert_padding(dummy_actor_gpt_model_with_pp):
    trtllm_generate = GPTGenerateTRTLLM(
        model_cfg=dummy_actor_gpt_model_with_pp.cfg,
        end_strings=["<extra_id_1>"],
        tokenizer=dummy_actor_gpt_model_with_pp.tokenizer,
        max_input_len=dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length // 2,
        max_generation_length=dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length // 2,
    )
    trtllm_generate.refit(dummy_actor_gpt_model_with_pp)

    batch_size = 4
    max_seq_len = dummy_actor_gpt_model_with_pp.cfg.encoder_seq_length
    prompt_tokens = torch.ones((batch_size, max_seq_len), dtype=torch.int32)
    prompt_lengths = torch.tensor([10, 20, 30, 40])

    output_ids, response_lengths = trtllm_generate._generate([prompt_tokens, prompt_lengths])
    max_length = response_lengths.max().item()

    # TRTLLM with PP has sometimes erroneously inserts padding:
    # As an example when we have the input:
    #     [[prompt tok, PAD, PAD], [prompt tok, prompt tok, prompt tok]]
    # The output when PP is enabled becomes:
    #     [[prompt tok, PAD, PAD, resp_tok, resp_tok], [prompt tok, prompt tok, prompt tok, resp_tok, resp_tok]]
    # Therefore we need this logic to get rid of the padding in the middle of the tensor.
    # Furthermore, TRTLLM only produces valid outputs on the source rank, so we can only process it here
    # and rely on the aligner broadcast to get it to the other ranks. Miraculously, the length
    # is still correct on the non src ranks
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.get_model_parallel_src_rank() == torch.distributed.get_rank()
    ):
        valid_tokens = output_ids != trtllm_generate.pad_id
        # we can't just naively use the response length here
        # because there are cases where the model generates
        # stop strings after it has stopped. so we need to
        # be slightly inefficient and then remove the excess later on
        valid_token_lengths = valid_tokens.sum(-1, keepdims=True)
        max_unpadded_length = valid_token_lengths.max()
        assert max_length <= max_unpadded_length, (
            "max unpadded length should be more or equal to max length. This assertion is probably happening because TRT-LLM considered a "
            "pad tokens in the response length"
        )

        _output_ids = torch.full(
            (response_lengths.size(0), max_unpadded_length),
            fill_value=trtllm_generate.pad_id,
            dtype=output_ids.dtype,
            device=output_ids.device,
        )

        # only fill up to the amount of valid tokens
        src_index_mask = (
            torch.arange(max_unpadded_length, device=response_lengths.device).view(1, -1) < valid_token_lengths
        )

        _output_ids[src_index_mask] = output_ids[valid_tokens]

        invalid_response_mask = torch.arange(max_unpadded_length, device=response_lengths.device).view(
            1, -1
        ) >= response_lengths.view(-1, 1)
        _output_ids[invalid_response_mask] = trtllm_generate.pad_id

        torch.testing.assert_close(output_ids, _output_ids)
