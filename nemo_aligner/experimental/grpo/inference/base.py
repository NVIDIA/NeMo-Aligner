# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

class InferenceBackendBase(ABC):
    @abstractmethod
    def __init__(self, model_cfg, tokenizer, **kwargs):
        """
        Initialize the backend with model configuration and tokenizer.
        Additional backend-specific arguments can be passed via kwargs.
        """
        pass

    @abstractmethod
    def refit(self, model):
        """
        Refit or recompile the model for inference.
        """
        pass

    @abstractmethod
    def generate(self, inputs):
        """
        Generate outputs based on the given inputs.
        Args:
            inputs: A tuple of tensors (input_tokens, input_lengths).
        Returns:
            dict containing:
                - 'response_tokens': Generated response tensors.
                - 'response_lengths': Corresponding lengths of the responses.
        """
        pass

    @abstractmethod
    def free(self):
        """
        Free up any resources or memory used by the backend.
        """
        pass
