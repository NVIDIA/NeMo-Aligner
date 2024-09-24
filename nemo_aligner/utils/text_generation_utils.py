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

"""Utilities for generating text."""

from typing import Any, List

import torch

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.text_generation_strategy import (
    GPTModelTextGenerationStrategy,
    TextGenerationStrategy,
)
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp


class TrackLengthGPTModelTextGenerationStrategy(GPTModelTextGenerationStrategy):
    """
    Text generation strategy that tracks the length of the generated text.

    TODO This is a temporary workaround until NeMo's `generate()` function returns this information.
    """

    def __init__(self, model: Any, context_lengths: torch.Tensor, max_length: int):
        super().__init__(model)
        self._context_lengths = context_lengths
        self._max_length = max_length
        self._end_idx = torch.full_like(context_lengths, fill_value=-1)

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        is_end = super().end_of_generation_condition(tokens=tokens, prev=prev, eod_id=eod_id, end_strings=end_strings)
        assert len(is_end) == len(tokens)
        if len(tokens) != len(self._context_lengths):
            raise RuntimeError(
                "Batch size mismatch: the `context_lengths` tensor provided in the constructor has batch size "
                f"{len(self._context_lengths)}, while the generated tokens have batch size {len(tokens)}"
            )
        context_length = tokens.size(1) - 1  # the input tokens come from `tokens[:, : context_length + 1]`
        started = self._context_lengths <= context_length
        # The generation ends right now when three conditions hold:
        #   - it has started
        #   - the end generation is triggered now
        #   - it did *not* end before
        self._end_idx = torch.where(started & is_end & (self._end_idx < 0), context_length, self._end_idx)
        return is_end

    def get_lengths(self) -> torch.Tensor:
        """
        Return the total lengths of the generated sequences, in # of tokens.

        The total length of a generated sequence counts both:
            * the context tokens (i.e., the input prompt)
            * the token(s) that ended generation, if any (e.g. the `EOS` token or the token(s) corresponding to
              an element of `sampling_params.end_strings`)
        """
        lengths = None
        if parallel_state.is_pipeline_last_stage():  # only the last stage actually has access to lengths
            lengths = torch.where(self._end_idx >= 0, self._end_idx + 1, self._context_lengths + self._max_length)
            lengths = lengths.to(torch.int64).view((-1, 1))
        lengths = broadcast_2d_tensor_within_pp(lengths, dtype=torch.int64).flatten()
        return lengths


def tokenize_batch(sentences, tokenizer, max_len, add_BOS=False, add_EOS=False):
    """convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
    """

    def tokenize(sentence):
        output = tokenizer.text_to_ids(sentence)

        if add_BOS:
            output = [tokenizer.bos_id] + output

        if add_EOS:
            output.append(tokenizer.eos_id)

        return output

    context_tokens = list(map(tokenize, sentences))
    max_sequence_length = max(len(x) for x in context_tokens)

    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len - max_sequence_length)
    context_tokens = [x[:max_len] for x in context_tokens]
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor


def verify_is_valid_and_clamp_range_(
    response_tokens, response_lengths, strategy: TextGenerationStrategy, tokenizer, end_strings=None
):
    """Function to verify if the tokens have properly ended, and clamp the tokens within the tokenizer range
    """
    if end_strings is None:
        end_strings = []

    prev = response_tokens[torch.arange(response_tokens.size(0)), response_lengths - 1]
    is_valid = strategy.end_of_generation_condition(response_tokens, prev, tokenizer.eos_id, end_strings)

    mask = (0 <= response_tokens) & (response_tokens < tokenizer.vocab_size)
    is_valid = is_valid & torch.all(mask, dim=-1)

    response_tokens.clamp_(0, tokenizer.vocab_size - 1)
    return is_valid
