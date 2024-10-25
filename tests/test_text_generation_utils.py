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

import torch

from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    tokenize_batch,
    verify_is_valid_and_clamp_range_,
)


class MockTokenizer:
    def __init__(self):
        self.vocab = dict()
        self.bos_id = 0
        self.eos_id = 1
        self.vocab["<bos>"] = self.bos_id
        self.vocab["<eos>"] = self.eos_id

    def text_to_ids(self, text):
        tokens = list(text)
        ids = [self.vocab.get(token, len(self.vocab)) for token in tokens]
        return ids


def test_tokenize_batch():
    sentences = ["I went to the store.", "I bought a zoo."]
    tokenizer = MockTokenizer()
    max_len = 30
    context_tokens_tensor, context_length_tensor = tokenize_batch(
        sentences, tokenizer, max_len, add_BOS=False, add_EOS=False
    )
    assert context_tokens_tensor.shape == (
        2,
        30,
    ), f"expected context_tokens_tensor shape to be (2, 30) but got {context_tokens_tensor.shape}"
    assert context_length_tensor.shape == (
        2,
    ), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
    assert context_length_tensor.tolist() == [
        20,
        15,
    ], f"expected context_length_tensor to be [20, 15] but got {context_length_tensor.tolist()}"


def test_tokenize_batch_with_sentence_longer_than_max_len():
    sentences = ["I went to the store.", "I bought a zoo."]
    tokenizer = MockTokenizer()
    max_len = 10
    context_tokens_tensor, context_length_tensor = tokenize_batch(
        sentences, tokenizer, max_len, add_BOS=False, add_EOS=False
    )
    assert context_tokens_tensor.shape == (
        2,
        10,
    ), f"expected context_tokens_tensor shape to be (2, 10) but got {context_tokens_tensor.shape}"
    assert context_length_tensor.shape == (
        2,
    ), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
    assert context_length_tensor.tolist() == [
        10,
        10,
    ], f"expected context_length_tensor to be [10, 10] but got {context_length_tensor.tolist()}"


def test_verify_is_valid_and_clamp_range(dummy_gpt_model):
    max_gen_length = 8

    random_gen = [9, 8]  # chosen arbitrarily
    extra_id_1_ids = dummy_gpt_model.tokenizer.text_to_ids("<extra_id_1>")
    extra_id_2_ids = dummy_gpt_model.tokenizer.text_to_ids("<extra_id_2>")
    eos_id = dummy_gpt_model.tokenizer.eos_id

    # response contains prompt + generation
    response_tokens = [
        [1] + random_gen,  # doesn't end with an eos
        [1, 1] + random_gen + [eos_id],
        [1] + random_gen + extra_id_1_ids,
        [1, 1] + random_gen + extra_id_1_ids,
        [1] + random_gen + extra_id_2_ids,
    ]

    # The padding has to be eos_id
    response_tokens = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in response_tokens], batch_first=True, padding_value=eos_id
    )

    context_lengths = torch.tensor([1, 2, 1, 2, 1])
    generation_lengths = torch.tensor([0, 1, len(extra_id_1_ids), len(extra_id_2_ids), len(extra_id_2_ids)]) + len(
        random_gen
    )
    response_lengths = context_lengths + generation_lengths

    strategy = TrackLengthGPTModelTextGenerationStrategy(dummy_gpt_model, context_lengths, max_gen_length)
    is_end = verify_is_valid_and_clamp_range_(
        response_tokens=response_tokens,
        response_lengths=response_lengths,
        strategy=strategy,
        tokenizer=dummy_gpt_model.tokenizer,
        end_strings=["<extra_id_1>"],
    )
    assert is_end.tolist() == [False, True, True, True, False]
