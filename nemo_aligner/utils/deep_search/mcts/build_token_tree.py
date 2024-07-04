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
import time
from typing import Callable, List

import numpy as np
import torch
import tqdm
from megatron.core import InferenceParams, parallel_state
from nemo_aligner.utils.deep_search.mcts.mcts import Node, ParallelSearch, MCTSParallel


from nemo_aligner.utils.utils import preemptable_save

sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000



class ExpandTree(MCTSParallel):
    def __init__(
        self,
        args,
        tokenizer,
        pad_id,
        session_info="session",
        stop_criteria=None,
        client_fun: Callable = None,
        has_value=True,
        value_estimation_function=None,
        time_limit=3*3600+30*60,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.session = session_info
        self.stop_criteria = stop_criteria
        self.client_fun = client_fun
        self.cache = {}
        self.has_value = has_value
        self.pad_id = pad_id
        self.value_estimation_function = value_estimation_function
        self.exit = False
        self.time_limit = time_limit

    @torch.no_grad()
    def search(self, ps: List[ParallelSearch]):
        start_time = time.time()
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
                action_size = len(spg_action)

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
            current_time = time.time()
            if current_time - start_time > self.time_limit:
                break
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
                value, is_terminal, ends_properly, has_answer = self.stop_criteria.get_value_and_terminated(
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
                        spg.value_memory.add((tuple(node.get_all_tokens()), value, node))

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

            for i in range(len(ps)):
                data_id = ps[i].data_id
                if data_id in self.stop_criteria.terminate and self.stop_criteria.terminate[data_id]:
                    # skip the search if a good solution is found
                    ps[i].node = None

            # index of search instances that are expandable
            expandable_search = [mappingIdx for mappingIdx in range(len(ps)) if ps[mappingIdx].node is not None]

            if len(expandable_search) > 0:
                # compute the batched policy and value for the expandable search nodes
                input_actions, context_ids, data_ids = self.get_input_action_depth(ps, expandable_search)
                #             result_dict = self.client.infer_batch(action=actions, depth=depths, context_ids=context_data, parameters={"session": self.session})
                result_dict = self.client_fun(
                    actions=input_actions, context_ids=context_ids, session_info=self.session
                )

                actions = result_dict["action"]
                policy = result_dict["policy"]  # [batch, top_k]
                if self.has_value:
                    value = result_dict["value"]  # [batch]
                else:
                    value = [None] * len(policy)

                if self.value_estimation_function is not None:
                    value = self.value_estimation_function(
                        inputs=None, action=input_actions, context_ids=context_ids, data_ids=data_ids,
                    )

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
                        ps[mappingIdx].value_memory.add((all_tokens, value_head_output, node))
                        self.cache[all_tokens] = value_head_output
                else:
                    # add temperature to the policy
                    node.expand(spg_policy, spg_action)

                    node.backpropagate(value_head_output)


class BuildTheTree:
    def __init__(
        self,
        mcts: MCTSParallel,
        max_steps: int,
        temperature: float,
        strategy=None,
        top_k: int = 50,
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
        self.top_k = top_k

    def clear_search_db_cache(self, backup_root_node):
        if self.strategy is not None and self.strategy.use_kv_cache:
            # clean up the cache
            context_id = tuple(backup_root_node.state)
            # depth first search to go through all the notes
            stack = [(backup_root_node, context_id)]
            while len(stack) > 0:
                node, c_id = stack.pop()
                self.strategy.clean_up_cache_for_context(self.mcts.session, c_id)
                for child_action in node.children:
                    child = node.children[child_action]
                    if len(child.children) != 0:
                        stack.append((child, c_id + tuple(child.state)))

    def search(self, parallel_searches: List[ParallelSearch], filename):
        dp_rank = parallel_state.get_data_parallel_rank()
        # clear the cache
        self.mcts.cache = {}
        if self.mcts.value_estimation_function is not None:
            self.mcts.value_estimation_function.value_cache = {}
        self.mcts.stop_criteria.reset()

        if self.cache_dir is not None:
            filename = os.path.join(self.cache_dir, filename)

        return_memory = []
        return_value_memory = []
        return_postive_negative_smaples = []

        # load the partial result from disk
        if os.path.exists(filename):
            pass
            # print("### LOADING CACHE FROM", filename)
            # load_beg = time.time()
            # cache = torch.load(filename)
            # parallel_searches = cache["parallel_searches"]
            # count = cache["count"]
            # backup_root_states = cache["backup_root_states"]
            # return_memory = cache["return_memory"]
            # return_value_memory = cache["return_value_memory"]
            # backup_root_nodes = cache["backup_root_nodes"]
            # if "search_db" in cache:
            #     self.strategy.load_state_dict(cache)
            # load_end = time.time()
            # print(f"### LOADING CACHE TOOK {load_end - load_beg} SECONDS")

            # show number os paralell searches left in the progress bar

            # start to do the mcts search
        self.mcts.search(parallel_searches)
        backup_root_states = [spg.state.copy() for spg in parallel_searches]

        for i in range(len(parallel_searches))[::-1]:
            spg = parallel_searches[i]
            value_mems = []
            for tokens, value, node in spg.value_memory:
                all_values = []
                all_tokens = []
                all_tokens += node.state[::-1]
                for _ in range(len(node.state)):
                    all_values.append(node.value_sum / node.visit_count)
                while node.parent is not None:
                    node = node.parent
                    for _ in range(len(node.state)):
                        all_values.append(node.value_sum / node.visit_count)
                    all_tokens += node.state[::-1]
                all_tokens = all_tokens[::-1]
                all_values = all_values[::-1]
                assert tuple(all_tokens) == tuple(tokens)
                value_mems.append((tokens, all_values, value))
            return_value_memory.append(
                {
                    "value_memory": value_mems,
                    "data_id": spg.data_id,
                    "backup_root_states": backup_root_states[i],
                }
            )

        return return_memory, return_value_memory, return_postive_negative_smaples
