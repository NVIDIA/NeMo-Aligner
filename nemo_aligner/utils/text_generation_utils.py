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


import torch
from nemo.utils import logging


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
