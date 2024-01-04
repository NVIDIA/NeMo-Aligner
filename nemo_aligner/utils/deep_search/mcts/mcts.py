# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math

import numpy as np
from megatron.core import InferenceParams


class State:
    def __init__(self, depth=0, kv_cache=None):
        """LM search related states

        Args:
            depth (int): depth of the search tree
            kv_cache (dict): dictionary of KV cache
        """
        self.depth = depth
        # kv cache is a dictionary. key is layer number, value is a k, v tuple where both k and v are torch tensors of shape (seq_len, batch_size, ...)
        self.kv_cache = kv_cache

    @staticmethod
    def get_state(infer_params: InferenceParams, depth: int, context_len: int, batch_id: int):
        # only call this function after inference step is done
        # infer_params.sequence_len_offset is the number of tokens used in the kv cache before the inference step
        # after the inference step, there is one more token in the kv cache
        length = infer_params.sequence_len_offset + 1
        if depth == 0:
            # root state has all the context
            # key_value_memory_dict [length, batch_size, ...]
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][:context_len, 0].detach().numpy(),
                    infer_params.key_value_memory_dict[key][1][:context_len, 0].detach().numpy(),
                )
                for key in infer_params.key_value_memory_dict
            }
            state = State(depth, kv_cache)
        else:
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][length - 1 : length, batch_id].detach().numpy(),
                    infer_params.key_value_memory_dict[key][1][length - 1 : length, batch_id].detach().numpy(),
                )
                for key in infer_params.key_value_memory_dict
            }
            state = State(depth, kv_cache)
        return state


class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, visit_count=0, C=2.0):
        """Node used in MCTS
        Args:
            state (State): inference state 
            parent (Node): Parent node. Defaults to None for root node.
            action (int): The action taken for current node. Defaults to None for root node.
            prior (float): prior probability. Defaults to 0.0.
            visit_count (int): visit counts for the current node. Defaults to 0.
            C (float): weight of prior. Defaults to 2.0.
        """
        self.C = C
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0
        # whether the parent turn is skipped or not
        self.skip_parent = False

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0.0  #
        else:
            q_value = (child.value_sum / child.visit_count + 1) / 2
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child = Node(self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)
