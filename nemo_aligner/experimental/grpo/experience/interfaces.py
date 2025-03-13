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
import abc
from typing import Dict, List, Tuple, Optional

from torch import Tensor
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.experimental.grpo.experience.rollout_batch import GPTRolloutBatch

EnvironmentReturn = Tuple[Optional[List[List[str]]], Optional[List[Dict]], Tensor, Tensor]

class EnvironmentInterface(abc.ABC):
    @abc.abstractmethod
    def start_step(
            self,
            interactions: List[List[str]],
            metadata: List[Dict],
            *args, **kwargs) -> FutureResult:
        """
        Starts a step in an environment and returns a Future. 
        Allows for asynchrony with remote servers, but it's not required.
        
        interactions: batch of lists of strings that represent interactions with the LLM.
                      For example, if this were a Math Environment, then the interactions
                      would be ["problem", "response"], but if this were a code environment
                      with feedback, it would be:
                      ["problem", "response", "code result", "model response", ...].
        metadata:     batch of whatever the environment needs to keep track of. I.e.
                      math solutions, code unit tests, or agent states.
        """
    
    @abc.abstractmethod
    def finish_step(self, future: FutureResult, *args, **kwargs) -> EnvironmentReturn:
        """
        Consumes the FutureResult provided by start_step and returns:
        - Updated interactions (or None to not update)
        - Updated metadata (whatever the environment wants for the future or for metrics)
          (or None to not update)
        - rewards: tensor (b,). Currently only supports full sequence rewards.
        - is_complete: tensor (b,). 0/1, 0 -> incomplete
        """
        pass

    def step(
            self,
            interactions: List[List[str]],
            metadata: List[Dict],
            *args, **kwargs) -> EnvironmentReturn:
        """
        Convenience function for if you don't want to deal with asynchrony
        """
        return self.finish_step(self.start_step(interactions, metadata, *args, **kwargs))
    
    @abc.abstractmethod
    def global_post_process_and_metrics(self, batch) -> Tuple[GPTRolloutBatch, dict]:
        """
        Post processing function after all rollouts are done for the batch.
        Also returns metrics.
        """


class RolloutGeneratorInterface(abc.ABC):
    @abc.abstractmethod
    def generate_rollouts(
            self,
            dataloader_iter,
            policy_model,
            *args, **kwargs) -> GPTRolloutBatch:
        """
        Iterates over a dataloader iterator and runs the policy_model and environments to 
        generate rollouts for the model to train on. 

        returns a globally collected rollout buffer
        """
