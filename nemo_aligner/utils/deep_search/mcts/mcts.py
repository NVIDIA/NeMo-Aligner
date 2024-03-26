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

import hashlib
import math
import os
import sys
import threading
from typing import Callable, List

import numpy as np
import torch
import tqdm
from megatron.core import InferenceParams, parallel_state
from nemo_aligner.utils.utils import preemptable_save

sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000


class ParallelSearch:
    """a class to store the state, root node, and current node of a single player game
    """

    def __init__(self, state, data_id):
        self.state = state  # list of tokens
        self.data_id = data_id  # data id of the state
        # memory is a list of (state, improved policy, player) tuples
        # from the root state to the end of the game
        self.memory = []
        self.value_memory = set()
        self.root = None
        self.node = None


def get_state(infer_params: InferenceParams, init: bool, context_len: int, batch_id: int, use_cpu=False):
    # only call this function after inference step is done
    # infer_params.sequence_len_offset is the number of tokens used in the kv cache before the inference step
    # after the inference step, there is one more token in the kv cache
    if use_cpu:
        if init:
            # root state has all the context
            # key_value_memory_dict [length, batch_size, ...]
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][:context_len, batch_id].detach().cpu(),
                    infer_params.key_value_memory_dict[key][1][:context_len, batch_id].detach().cpu(),
                )
                for key in infer_params.key_value_memory_dict
            }
        else:
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][context_len : context_len + 1, batch_id].detach().cpu(),
                    infer_params.key_value_memory_dict[key][1][context_len : context_len + 1, batch_id].detach().cpu(),
                )
                for key in infer_params.key_value_memory_dict
            }
    else:
        if init:
            # root state has all the context
            # key_value_memory_dict [length, batch_size, ...]
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][:context_len, batch_id].detach().clone(),
                    infer_params.key_value_memory_dict[key][1][:context_len, batch_id].detach().clone(),
                )
                for key in infer_params.key_value_memory_dict
            }
        else:
            kv_cache = {
                key: (
                    infer_params.key_value_memory_dict[key][0][context_len : context_len + 1, batch_id]
                    .detach()
                    .clone(),
                    infer_params.key_value_memory_dict[key][1][context_len : context_len + 1, batch_id]
                    .detach()
                    .clone(),
                )
                for key in infer_params.key_value_memory_dict
            }
    return kv_cache


class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, visit_count=0, value_sum=0.0):
        """Node used in MCTS
        Args:
            state (State): inference state 
            parent (Node): Parent node. Defaults to None for root node.
            action (int): The action taken for current node. Defaults to None for root node.
            prior (float): prior probability. Defaults to 0.0.
            visit_count (int): visit counts for the current node. Defaults to 0.
            C (float): weight of prior. Defaults to 2.0.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = {}

        self.visit_count = visit_count
        self.value_sum = value_sum

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self, C):
        best_child = None
        best_ucb = -np.inf

        for child_action in self.children:
            child = self.children[child_action]
            ucb = self.get_ucb(child, C)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, C):
        if child.visit_count == 0:
            q_value = child.prior  # # use prior as initial value
        else:
            q_value = child.value_sum / child.visit_count  # assume the q_value is probability of winning
        return q_value + C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy, actions):
        for action, prob in zip(actions, policy):
            action = action.item()
            prob = prob.item()
            child = Node([action], parent=self, action=action, prior=prob, visit_count=0)
            self.children[action] = child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

    def get_all_tokens(self):
        node = self
        all_tokens = []
        all_tokens += node.state[::-1]
        while node.parent is not None:
            node = node.parent
            all_tokens += node.state[::-1]
        return all_tokens[::-1]


class MCTSParallel:
    def __init__(
        self,
        args,
        tokenizer,
        session_info="session",
        score_fn=None,
        terminate_fns=None,
        client_fun: Callable = None,
        has_value=True,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.session = session_info
        self.score_fn = score_fn
        self.terminate_fns = terminate_fns
        self.client_fun = client_fun
        self.cache = {}
        self.has_value = has_value

    def decode_text(self, state):
        decoded_text = self.tokenizer.decode(state)
        # decoded_text = "".join(state).replace('▁', ' ').lstrip()
        return decoded_text

    def token_to_context_id(self, all_tokens, node):
        if node.parent is None:
            # for the root node, the context id is the same as the state
            return tuple(all_tokens)
        else:
            # for the child node, the context id is the same as the parent node
            num_action_tokens = len(node.state)
            return tuple(all_tokens[:-num_action_tokens])

    def get_text(self, node):
        all_tokens = node.get_all_tokens()
        text = self.decode_text(all_tokens)
        return text

    def get_value_and_terminated(self, text, data_id, depth, tokens):
        terminate = False
        for fun in self.terminate_fns:
            if fun(text, depth, tokens):
                terminate = True
                break

        value = 0.0
        if terminate:
            value = self.score_fn.score(text, data_id)
        # check if the text ends properly
        end_properly = False
        for fun in self.terminate_fns:
            if fun.ends_by_end_strings(text, tokens):
                end_properly = True
                break
        has_answer = False
        for fun in self.terminate_fns:
            if fun.has_answer(text):
                has_answer = True
                break
        return value, terminate, end_properly, has_answer

    def get_input_action_depth(self, ps, expandable_search):
        # get the action to execute and depths in the search tree
        actions = np.stack([ps[mappingIdx].node["node"].action for mappingIdx in expandable_search])[
            :, np.newaxis
        ].astype(np.int32)
        # get the context ids for the search nodes
        # context_ids = [ps[mappingIdx].node.context_id for mappingIdx in expandable_search]
        context_ids = [
            self.token_to_context_id(ps[mappingIdx].node["all_tokens"], ps[mappingIdx].node["node"])
            for mappingIdx in expandable_search
        ]
        # # verify context ids are the same
        # for old_context_id, new_context_id in zip(context_ids, new_context_ids):
        #     assert old_context_id == new_context_id
        return actions, context_ids

    @torch.no_grad()
    def search(self, ps: List[ParallelSearch]):
        # states: (batch_size, row_count, column_count), neutral perspective

        if ps[0].node is not None:
            # the context is already infered
            # expandable_search = list(range(len(ps)))
            # actions, context_ids = self.get_input_action_depth(ps, expandable_search)
            for p in ps:
                p.root = p.node
                assert p.root.parent is None
                p.root.action = -1
            # spg.root = Node(spg.state, parent=None, action=-1, prior=0.0, visit_count=1)

            # # result_dict = self.client.infer_batch(action=actions, depth=depths, context_ids=context_data, parameters={"session": self.session})
            # # need to remove the last token from the context id
            # infer_context_ids = [context_id[:-1] for context_id in context_ids]
            # result_dict = self.client_fun(action=actions, context_ids=infer_context_ids, session_info=self.session)
            # spg_policys = result_dict["policy"]
            # spg_actions = result_dict["action"]
            # c_ids = context_ids  # [old_context + (new.item(),)  for old_context, new in zip(context_ids, actions)]
            # need to add the action to the context
        else:
            # we need to run inferecce for all context ids
            # init case, where the kv-cache is built at the server
            input_to_text_map = {tuple(spg.state): self.decode_text(spg.state) for spg in ps}

            streamline_context_ids = []
            streamline_inputs = []
            for streamline_context_id, streamline_input_text in input_to_text_map.items():
                streamline_context_ids.append(streamline_context_id)
                streamline_inputs.append(streamline_input_text)
            result_dict = self.client_fun(
                sentences=list(streamline_inputs), context_ids=list(streamline_context_ids), session_info=self.session
            )

            actions = result_dict["action"]
            policy = result_dict["policy"]  # [batch, top_k]

            input_action_map = {
                context_input[0]: action for context_input, action in zip(input_to_text_map.items(), actions)
            }
            input_policy_map = {
                context_input[0]: policy for context_input, policy in zip(input_to_text_map.items(), policy)
            }

            spg_policys = []
            spg_actions = []
            c_ids = []

            for i, spg in enumerate(ps):
                # spg_input = self.decode_text(spg.state)
                spg_policy = input_policy_map[tuple(spg.state)]
                spg_policys.append(spg_policy)
                spg_action = input_action_map[tuple(spg.state)]
                spg_actions.append(spg_action)
                context_id = tuple(spg.state)
                c_ids.append(context_id)

            for spg, spg_policy, spg_action, context_id in zip(ps, spg_policys, spg_actions, c_ids):
                action_size = spg_action.shape[0]

                spg_policy = (1 - self.args["dirichlet_epsilon"]) * spg_policy + self.args[
                    "dirichlet_epsilon"
                ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * action_size, size=1)[0]
                # no need to handle the case that no valid moves
                # because the we search the states[i] which has at least one valid move
                spg.root = Node(spg.state, parent=None, action=-1, prior=0.0, visit_count=1)
                spg.root.expand(spg_policy, spg_action)

        # dp_rank = parallel_state.get_data_parallel_rank()
        # use tqdm to show the progresso of the self play
        # for search in tqdm.tqdm(range(self.args["num_searches"]), desc=f"MCTS rank: {dp_rank}", leave=False):
        for search in range(self.args["num_searches"]):
            for spg in ps:
                # spg.node is to save the node that needs to be expanded
                spg.node = None
                # start from the root node
                depth = 0
                node = spg.root

                # select the leaf node based on ucb score
                while node.is_fully_expanded():
                    node = node.select(self.args["C"])
                    depth += 1

                # check the move is done or not, if yes, then backpropagate the value, no need to expand the node
                all_tokens = node.get_all_tokens()
                text = self.decode_text(all_tokens)
                value, is_terminal, ends_properly, has_answer = self.get_value_and_terminated(
                    text, spg.data_id, depth, all_tokens
                )

                if is_terminal:
                    if not self.args["oracle"]:
                        # if no oracle, then we need to run value inference to get the value

                        # cache the value for this node
                        all_tokens_tuple = tuple(all_tokens)
                        if all_tokens_tuple in self.cache:
                            value = self.cache[all_tokens_tuple]
                        else:
                            # spg.node is a dictory
                            spg.node = {
                                "node": node,
                                "ends_properly": ends_properly,
                                "all_tokens": all_tokens,
                                "text": text,
                                "is_terminal": is_terminal,
                            }
                            # skip the backpropagation and run inference later to get the value
                            continue

                    # if terminal, then backpropagate the value, and skip the expansion of the node because spg.node is None
                    node.backpropagate(value)
                    # collect the memory from the root to the terminal node
                    if ends_properly:
                        # returns the tokens, the improved policy, the outcome score, the actions for imporoved pollicy and the data id
                        spg.value_memory.add((tuple(node.get_all_tokens()), value))

                else:
                    # if not terminal, then expand the node in the later part of the code
                    # spg.node is a dictory
                    spg.node = {
                        "node": node,
                        "ends_properly": ends_properly,
                        "all_tokens": all_tokens,
                        "text": text,
                        "is_terminal": is_terminal,
                    }

            # index of search instances that are expandable
            expandable_search = [mappingIdx for mappingIdx in range(len(ps)) if ps[mappingIdx].node is not None]

            if len(expandable_search) > 0:
                # compute the batched policy and value for the expandable search nodes
                actions, context_ids = self.get_input_action_depth(ps, expandable_search)
                #             result_dict = self.client.infer_batch(action=actions, depth=depths, context_ids=context_data, parameters={"session": self.session})
                result_dict = self.client_fun(action=actions, context_ids=context_ids, session_info=self.session)

                actions = result_dict["action"]
                policy = result_dict["policy"]  # [batch, top_k]
                if self.has_value:
                    value = result_dict["value"]  # [batch]
                else:
                    value = [None] * len(policy)

            for i, mappingIdx in enumerate(expandable_search):
                # node to expand
                result_dict = ps[mappingIdx].node
                node = result_dict["node"]
                # corresponding policy and value
                spg_policy, spg_value, spg_action = policy[i], value[i], actions[i]
                if spg_value is not None:
                    value_head_output = spg_value.item()
                else:
                    value_head_output = node.prior
                if self.args["turn_off_value"]:
                    value_head_output = node.prior

                if result_dict["is_terminal"]:
                    # if the node is a tuple, then it means the node is terminal
                    # backpropagate the value
                    ends_properly = result_dict["ends_properly"]
                    all_tokens = result_dict["all_tokens"]
                    node.backpropagate(value_head_output)
                    if ends_properly:
                        # collect the memory from the root to the terminal node
                        # returns the tokens, the improved policy, the outcome score, the actions for imporoved pollicy and the data id
                        all_tokens = tuple(all_tokens)
                        ps[mappingIdx].value_memory.add((all_tokens, value_head_output))
                        self.cache[all_tokens] = value_head_output
                else:
                    node.expand(spg_policy, spg_action)

                    node.backpropagate(value_head_output)


class DeepSearch:
    def __init__(
        self,
        mcts: MCTSParallel,
        max_steps: int,
        temperature: float,
        strategy=None,
        timer_seconds: int = 10.0,
        cache_dir: str = None,
        inference_only: bool = False,
    ):
        self.mcts = mcts
        self.max_steps = max_steps
        self.temperature = temperature
        self.strategy = strategy
        self.save_flag = False
        self.cache_dir = cache_dir
        # if inference_only is True, then the search will only run the inference and not the self play
        self.inference_only = inference_only
        # Start the timer
        self.timer = threading.Timer(timer_seconds, self.save_data)
        self.timer.start()

    def save_data(self):
        print("### TIMER TRIGGER")
        self.save_flag = True

    def search(self, parallel_searches: List[ParallelSearch], filename):
        dp_rank = parallel_state.get_data_parallel_rank()
        # clear the cache
        self.mcts.cache = {}
        # serialize the partial result to disk

        if self.cache_dir is not None:
            filename = os.path.join(self.cache_dir, filename)

        # equavalent to the alpha zero self play
        # for a list of parallel_searche instances
        # do the mcts search to find the best action and improved policy
        # move on to next next state until either the end of chat is reached or the max_steps is reached
        # collect the memory from all the improved response during the self play
        return_memory = []

        backup_root_states = [spg.state.copy() for spg in parallel_searches]
        return_value_memory = []

        count = 0
        # load the partial result from disk
        if os.path.exists(filename) and self.strategy is not None:
            print("### LOADING CACHE FROM", filename)
            cache = torch.load(filename)
            parallel_searches = cache["parallel_searches"]
            count = cache["count"]
            backup_root_states = cache["backup_root_states"]
            return_memory = cache["return_memory"]
            return_value_memory = cache["return_value_memory"]
            self.strategy.load_state_dict(cache)

        # add a progress bar to show the progress of the self play
        total_steps = self.max_steps
        pb = tqdm.tqdm(total=total_steps, initial=count, desc=f"Self Play rank {dp_rank}", leave=True)
        while len(parallel_searches) > 0:
            # TODO need to clear the session memory in the server
            count += 1
            pb.update(1)
            # show number os paralell searches left in the progress bar
            pb.set_postfix({"searches": len(parallel_searches)})

            # start to do the mcts search
            self.mcts.search(parallel_searches)

            # loop from large to small so that we can remove search instances as we go
            for i in range(len(parallel_searches))[::-1]:
                spg = parallel_searches[i]
                action_size = len(spg.root.children)
                action_probs = np.zeros(action_size, dtype=np.float32)
                actions = np.zeros(action_size, dtype=np.int32)
                use_value_sum = False
                for child_id, child_action in enumerate(spg.root.children.keys()):
                    child = spg.root.children[child_action]
                    assert child_action == child.action
                    if use_value_sum:
                        action_probs[child_id] = child.value_sum
                    else:
                        action_probs[child_id] = child.visit_count
                    actions[child_id] = child.action
                action_probs /= np.sum(action_probs)

                # the spg.root.state is the neutral state set at the beginning of the search
                spg.memory.append((spg.state, action_probs, actions))

                temperature_action_probs = action_probs ** (1.0 / self.temperature)
                temperature_action_probs /= np.sum(temperature_action_probs)
                action_index = np.random.choice(
                    action_size, p=temperature_action_probs
                )  # Divide temperature_action_probs with its sum in case of an error
                action = actions[action_index].item()

                spg.state = spg.state + [action]
                fake_node = Node(spg.state, parent=None, action=action, prior=0.0, visit_count=0)
                # pass in the states from selected child node to the fake node
                child_node = spg.root.children[action]
                assert child_node.action == fake_node.action
                fake_node.children = child_node.children
                spg.node = fake_node

                #  get the value and termination condition from the current taken `action`
                text = self.mcts.decode_text(spg.state)
                pb.write(text)
                value, is_terminal, ends_properly, has_answer = self.mcts.get_value_and_terminated(
                    text, spg.data_id, count, spg.state
                )

                if is_terminal:
                    if self.inference_only:
                        # if inference only, we only collect the best tokens from the inference
                        all_tokens = tuple(spg.state)
                        if all_tokens in self.mcts.cache:
                            value = self.mcts.cache[all_tokens]
                        else:
                            value = -1
                        return_memory.append(
                            {
                                "tokens": spg.state,
                                "reward": value,
                                "data_id": spg.data_id,
                                "context_length": len(backup_root_states[i]),
                                "full_text": self.mcts.decode_text(spg.state),
                                "context": self.mcts.decode_text(backup_root_states[i]),
                            }
                        )
                        # need to clean up the mcts cache starting from backup root states
                        if self.strategy is not None:
                            # clean up the cache
                            self.strategy.clean_up_cache_for_context(self.mcts.session, backup_root_states[i])
                        # we can remove the search instance
                        del parallel_searches[i]
                        del backup_root_states[i]
                        continue
                    # loop through all the steps and add to the memory
                    # need to update the value based on the game play at the end of the games
                    # collects the value buffer if the response ends properly with <extra_id> or byte token
                    # or if the response has the answer inside it
                    if ends_properly or has_answer:
                        # only collect the memory if it ends properly
                        for tokens, hist_action_probs, actions in spg.memory:
                            hist_outcome = value
                            # returns the tokens, the improved policy, the outcome score, the actions for imporoved pollicy and the data id
                            return_memory.append(
                                {
                                    "tokens": tokens,
                                    "action_probs": hist_action_probs,
                                    "reward": hist_outcome,
                                    "actions": actions,
                                    "data_id": spg.data_id,
                                    "context_length": len(backup_root_states[i]),
                                }
                            )
                    return_value_memory.append(
                        {
                            "value_memory": list(spg.value_memory),
                            "data_id": spg.data_id,
                            "backup_root_states": backup_root_states[i],
                        }
                    )

                    # need to clean up the mcts cache starting from backup root states
                    if self.strategy is not None:
                        # clean up the cache
                        self.strategy.clean_up_cache_for_context(self.mcts.session, backup_root_states[i])

                    del parallel_searches[i]
                    del backup_root_states[i]
            if self.save_flag:
                if self.strategy is not None:
                    pb.write(f"saving the search to disk {filename}")

                    preemptable_save(
                        {
                            "parallel_searches": parallel_searches,
                            "count": count,
                            "backup_root_states": backup_root_states,
                            "return_memory": return_memory,
                            "return_value_memory": return_value_memory,
                        }
                        | self.strategy.state_dict(),
                        filename,
                    )
                    print("#### SAVING CACHE TO", filename)
                    # only save one
                self.save_flag = False

        return return_memory, return_value_memory
