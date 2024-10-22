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

from nemo_aligner.utils.utils import calculate_response_lengths, select_log_probs
from nemo_aligner.utils.text_generation_utils import tokenize_batch


def test_calculate_response_lengths():
    pad_id = 99999
    toks = [list(range(10)) + [pad_id] * 10, list(range(20))]

    len_first, len_second = calculate_response_lengths(torch.as_tensor(toks), pad_id)

    assert len_first == 10, f"expected response length at idx 0 to be 10 but got {len_first}"
    assert len_second == 20, f"expected response length at idx 1 to be 20 but got {len_second}"


def test_select_log_probs():
    """test the log probs slicing

    consider the following hand derivation for BS = 1

    [ 0, 1, 2, 3, 4 -> prediction 2nd token
      5, 6, 7, 8, 9 -> prediction for 3rd token
      10,11,12,13,14] -> not used

    labels
    [0
     1
     2]

    then the output should be 1, 7
    """
    # B x S x V
    full_log_probs = torch.arange(2 * 3 * 5).reshape(2, 3, 5).float()
    labels = torch.cat([torch.arange(3), torch.arange(3)]).long().reshape(2, 3)

    log_probs, full_log_probs = select_log_probs(full_log_probs, labels)

    log_probs = log_probs.tolist()

    # hand derived solution
    assert log_probs[0] == [1, 7], f"expected log probs at idx 0 to be [1,7] but got {log_probs[0]}"
    assert log_probs[1] == [16, 22], f"expected log probs at idx 1 to be [16,22] but got {log_probs[1]}"


class _MockTokenizer:
  def __init__(self):
    self.vocab = dict()
    self.bos_id = 0
    self.eos_id = 1
    self.vocab['<bos>'] = self.bos_id
    self.vocab['<eos>'] = self.eos_id

  def text_to_ids(self, text):
    tokens = list(text)
    ids = [self.vocab.get(token, len(self.vocab)) for token in tokens]
    return ids

def test_tokenize_batch():
  sentences = ['I went to the store.', 'I bought a zoo.']
  tokenizer = _MockTokenizer()
  max_len = 30
  context_tokens_tensor, context_length_tensor = tokenize_batch(sentences, tokenizer, max_len, add_BOS=False, add_EOS=False)
  assert context_tokens_tensor.shape == (2, 30), f"expected context_tokens_tensor shape to be (2, 30) but got {context_tokens_tensor.shape}"
  assert context_length_tensor.shape == (2,), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
  assert context_length_tensor.tolist() == [20, 15], f"expected context_length_tensor to be [20, 15] but got {context_length_tensor.tolist()}"

def test_tokenize_batch_with_sentence_longer_than_max_len():
  sentences = ['I went to the store.', 'I bought a zoo.']
  tokenizer = _MockTokenizer()
  max_len = 10
  context_tokens_tensor, context_length_tensor = tokenize_batch(sentences, tokenizer, max_len, add_BOS=False, add_EOS=False)
  assert context_tokens_tensor.shape == (2, 10), f"expected context_tokens_tensor shape to be (2, 30) but got {context_tokens_tensor.shape}"
  assert context_length_tensor.shape == (2,), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
  assert context_length_tensor.tolist() == [10, 10], f"expected context_length_tensor to be [20, 15] but got {context_length_tensor.tolist()}"


