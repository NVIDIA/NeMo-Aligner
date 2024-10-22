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

from nemo_aligner.utils.text_generation_utils import tokenize_batch

class MockTokenizer:
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
  tokenizer = MockTokenizer()
  max_len = 30
  context_tokens_tensor, context_length_tensor = tokenize_batch(sentences, tokenizer, max_len, add_BOS=False, add_EOS=False)
  assert context_tokens_tensor.shape == (2, 30), f"expected context_tokens_tensor shape to be (2, 30) but got {context_tokens_tensor.shape}"
  assert context_length_tensor.shape == (2,), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
  assert context_length_tensor.tolist() == [20, 15], f"expected context_length_tensor to be [20, 15] but got {context_length_tensor.tolist()}"

def test_tokenize_batch_with_sentence_longer_than_max_len():
  sentences = ['I went to the store.', 'I bought a zoo.']
  tokenizer = MockTokenizer()
  max_len = 10
  context_tokens_tensor, context_length_tensor = tokenize_batch(sentences, tokenizer, max_len, add_BOS=False, add_EOS=False)
  assert context_tokens_tensor.shape == (2, 10), f"expected context_tokens_tensor shape to be (2, 10) but got {context_tokens_tensor.shape}"
  assert context_length_tensor.shape == (2,), f"expected context_length_tensor shape to be (2,) but got {context_length_tensor.shape}"
  assert context_length_tensor.tolist() == [10, 10], f"expected context_length_tensor to be [10, 10] but got {context_length_tensor.tolist()}"
