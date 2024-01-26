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
from typing import List
import torch
import numpy as np
from megatron.core import InferenceParams
from pytriton.client import ModelClient
import hashlib
import tqdm


def context_to_id(context: str) -> str:
    # generate unique context ids using hash function
    output = hashlib.md5(context.encode())
    return output.hexdigest()


class ParallelSearch:
    """a class to store the state, root node, and current node of a single player game
    """
    def __init__(self, state, data_id):
        self.state = state # list of tokens 
        self.data_id = data_id # data id of the state
        # memory is a list of (state, improved policy, player) tuples
        # from the root state to the end of the game
        self.memory = []
        self.root = None
        self.node = None



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
        if depth == 0:
            # root state has all the context
            # key_value_memory_dict [length, batch_size, ...]
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][:context_len, batch_id].detach().cpu().type(torch.float32).numpy(),
                    infer_params.key_value_memory_dict[key][1][:context_len, batch_id].detach().cpu().type(torch.float32).numpy(),
                )
                for key in infer_params.key_value_memory_dict
            }
            state = State(depth, kv_cache)
        else:
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][context_len : context_len + 1, batch_id].detach().cpu().type(torch.float32).numpy(),
                    infer_params.key_value_memory_dict[key][1][context_len : context_len + 1, batch_id].detach().cpu().type(torch.float32).numpy(),
                )
                for key in infer_params.key_value_memory_dict
            }
            state = State(depth, kv_cache)
        return state


class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, visit_count=0, C=2.0, depth=0, context_id = None):
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
        self.value_sum = 0.0
        # whether the parent turn is skipped or not
        self.skip_parent = False
        self.depth = depth
        self.context_id = context_id

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
            q_value = child.value_sum / child.visit_count # assume the q_value is probability of winning
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy, actions):
        for action, prob in zip(actions, policy):
            action = action.item()
            prob = prob.item()
            child = Node([action], parent=self, action=action, prior=prob, visit_count=0, C=self.C, depth=self.depth + 1, context_id=self.context_id)
            self.children.append(child)
        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTSParallel:
    def __init__(self, args, url, model_name, tokenizer, session_info='session', score_fn=None, terminate_fns=None):
        self.args = args
        self.client = ModelClient(url, model_name) 
        self.tokenizer = tokenizer
        self.session = session_info
        self.score_fn = score_fn
        self.terminate_fns = terminate_fns
    
    def decode_text(self, state):
        decoded_text = self.tokenizer.decode(state)
        # decoded_text = "".join(state).replace('▁', ' ').lstrip()
        return decoded_text
    
    def get_text(self, node):
        all_tokens = []
        all_tokens += node.state[::-1]
        while node.parent is not None:
            node = node.parent
            all_tokens += node.state[::-1]
        text = self.decode_text(all_tokens[::-1])
        return text
   
    def get_value_and_terminated(self, text, data_id, depth):
        terminate = False
        for fun in self.terminate_fns:
            if fun(text, depth):
                terminate = True
                break
        
        value = 0.0
        if terminate:
            value = self.score_fn.score(text, data_id)
        print(text)
        # check 
        return value, terminate
       
    @torch.no_grad()
    def search(self, ps: List[ParallelSearch]):
        # states: (batch_size, row_count, column_count), neutral perspective
        # spGames: list of spGame
        inputs = [self.decode_text(spg.state) for spg in ps]

        context_input_map = {context_to_id(input_item): input_item for input_item in inputs}
        input_context_map = {input_item: context_to_id(input_item) for input_item in inputs}

        streamline_inputs = [b for _, b in context_input_map.items()]
        streamline_context_ids = [a for a, _ in context_input_map.items()]

        str_ndarray = np.array(streamline_inputs)[..., np.newaxis]
        input_data = np.char.encode(str_ndarray, "utf-8")

        context_ids = np.array(streamline_context_ids)[..., np.newaxis]
        context_data = np.char.encode(context_ids, "utf-8")

        result_dict = self.client.infer_batch(sentences=input_data, context_ids=context_data, parameters={"session": self.session})
        actions = result_dict['action']
        policy = result_dict['policy'] # [batch, top_k]

        input_action_map = {context_input[1]: action for context_input, action in zip(context_input_map.items(), actions)}
        input_policy_map = {context_input[1]: policy for context_input, policy in zip(context_input_map.items(), policy)}

        for i, spg in enumerate(ps):
            spg_input = self.decode_text(spg.state)
            spg_policy = input_policy_map[spg_input]
            spg_action = input_action_map[spg_input]
            context_id = input_context_map[spg_input]
            action_size = spg_action.shape[0]

            spg_policy = (1 - self.args['dirichlet_epsilon']) * spg_policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * action_size, size=1)[0]
 
            # no need to handle the case that no valid moves
            # because the we search the states[i] which has at least one valid move
            spg.root = Node(spg.state, parent=None, action=None, prior=0.0, visit_count=1, C=self.args['C'], depth=0, context_id=context_id)
            spg.root.expand(spg_policy, spg_action)
        
        for search in range(self.args['num_searches']):

            for spg in ps:
                # spg.node is to save the node that needs to be expanded
                spg.node = None
                # start from the root node
                node = spg.root

                # select the leaf node based on ucb score
                while node.is_fully_expanded():
                    node = node.select()
                
                # check the move is done or not, if yes, then backpropagate the value, no need to expand the node
                text = self.get_text(node)
                value, is_terminal = self.get_value_and_terminated(text, spg.data_id, node.depth)

                if is_terminal:
                    # if terminal, then backpropagate the value, and skip the expansion of the node because spg.node is None
                    node.backpropagate(value)
                else:
                    # if not terminal, then expand the node in the later part of the code
                    spg.node = node
                    
            # index of search instances that are expandable
            expandable_search = [mappingIdx for mappingIdx in range(len(ps)) if ps[mappingIdx].node is not None]
                    
            if len(expandable_search) > 0:
                # compute the batched policy and value for the expandable search nodes

                # get the action to execute and depths in the search tree
                actions = np.stack([ps[mappingIdx].node.action for mappingIdx in expandable_search])[:, np.newaxis].astype(np.int32)
                depths = np.stack([ps[mappingIdx].node.depth for mappingIdx in expandable_search])[:, np.newaxis].astype(np.int32)
                # get the context ids for the search nodes
                context_ids = np.stack([ps[mappingIdx].node.context_id for mappingIdx in expandable_search])[..., np.newaxis]
                context_data = np.char.encode(context_ids, "utf-8")

                result_dict = self.client.infer_batch(action=actions, depth=depths, context_ids=context_data, parameters={"session": self.session})

                actions = result_dict['action']
                policy = result_dict['policy'] # [batch, top_k]
                value = result_dict['value'] # [batch]
                
            for i, mappingIdx in enumerate(expandable_search):
                # node to expand
                node = ps[mappingIdx].node
                # corresponding policy and value
                spg_policy, spg_value, spg_action = policy[i], value[i], actions[i]
                
                node.expand(spg_policy, spg_action)

                node.backpropagate(spg_value.item())


def deep_search(parallel_searches: List[ParallelSearch], mcts: MCTSParallel, max_steps: int, temperature: float):
    # equavalent to the alpha zero self play
    # for a list of parallel_searche instances
    # do the mcts search to find the best action and improved policy
    # move on to next next state until either the end of chat is reached or the max_steps is reached
    # collect the memory from all the improved response during the self play
    return_memory = []
    
    # play each of the spg to the end 
    count = 0
    # add a progress bar to show the progress of the self play
    total_steps = max_steps
    pb = tqdm.tqdm(total=total_steps, desc=f'Self Play') 
    while len(parallel_searches) > 0:
        # TODO need to clear the session memory in the server
        count += 1
        pb.update(1)
        # show number os paralell searches left in the progress bar
        pb.set_postfix({'searches': len(parallel_searches)})

        # start to do the mcts search 
        mcts.search(parallel_searches)
        
        # loop from large to small so that we can remove search instances as we go
        for i in range(len(parallel_searches))[::-1]:
            spg = parallel_searches[i]
            action_size = len(spg.root.children)
            action_probs = np.zeros(action_size, dtype=np.float32)
            actions = np.zeros(action_size, dtype=np.int32)
            for child_id, child in enumerate(spg.root.children):
                action_probs[child_id] = child.visit_count
                actions[child_id] = child.action
            action_probs /= np.sum(action_probs)

            # the spg.root.state is the neutral state set at the beginning of the search 
            spg.memory.append((spg.root.state, action_probs, actions))

            temperature_action_probs = action_probs ** (1.0 / temperature)
            temperature_action_probs /= np.sum(temperature_action_probs)
            action_index = np.random.choice(action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error
            action = actions[action_index].item()

            spg.state = spg.state + [action]

            #  get the value and termination condition from the current taken `action`
            text = mcts.decode_text(spg.state)
            value, is_terminal = mcts.get_value_and_terminated(text, spg.data_id, i+1)

            if is_terminal:
                # loop through all the steps and add to the memory
                # need to update the value based on the game play at the end of the games
                for tokens, hist_action_probs, actions in spg.memory:
                    hist_outcome = value
                    return_memory.append((
                        tokens,
                        hist_action_probs,
                        hist_outcome
                    ))
                del parallel_searches[i]

    return return_memory
