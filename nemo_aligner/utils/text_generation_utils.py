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

from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.utils import logging

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
        lengths = broadcast_2d_tensor_within_pp(lengths, dtype=torch.int64)
        return lengths.flatten()


def pad_batch(batch, pad_id):
    """batch each element of the batch to be the size of the longest sequence
    """
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length:
            tokens.extend([pad_id] * (max_context_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def tokenize_batch(tokenizer, sentences, max_len, add_BOS, add_EOS=False):
    """convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
    Args:
        sentences (List[str]): list of input sentences in str format.
        max_len (int): max number of tokens to generate.
        add_BOS (bool): whether to add the BOS token at the beginning
    Returns:
        Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
    """

    def tokenize(sentence):
        output = tokenizer.text_to_ids(sentence)

        if add_BOS:
            output = [tokenizer.bos_id] + output

        if add_EOS:
            output.append(tokenizer.eos_id)

        return output

    context_tokens = list(map(tokenize, sentences))

    exceeded = [False] * len(context_tokens)

    for i, x in enumerate(context_tokens):
        if len(x) > max_len:
            logging.warning(f"max seq len of {max_len} exceeded, chunking")
            exceeded[i] = True

    context_tokens = [x[:max_len] for x in context_tokens]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor, exceeded
