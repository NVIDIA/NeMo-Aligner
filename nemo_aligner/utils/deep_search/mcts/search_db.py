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
import numpy as np
import torch
from nemo_aligner.utils.deep_search.mcts.mcts import Node


class SearchDB:
    def __init__(self):
        self.db = {}
        # node_id
        self.node_ids = {}
        self.inference_params = {}
        self.attention_mask = {}
        self.position_ids = {}

    def add_init_obj(self, session_id, inference_params, attention_mask, position_ids):
        self.inference_params[session_id] = inference_params
        self.attention_mask[session_id] = attention_mask
        self.position_ids[session_id] = position_ids

    def add_root(self, session_info, context_id, node):
        if session_info not in self.db:
            session = {}
            self.db[session_info] = session
        else:
            session = self.db[session_info]
        for token in context_id[:-1]:
            if token not in session:
                session[token] = Node(None, None, None, None, None, None)
            session = session[token].children
        session[context_id[-1]] = node

    def get(self, session_info, context_id):
        if session_info not in self.db:
            raise ValueError(f"{session_info} not in db")
        db = self.db[session_info]
        for token in context_id[:-1]:
            db = db[token].children
        return db[context_id[-1]]

    def get_infer_cache(self, session_info, context_id, action):
        if session_info not in self.db:
            return None
        db = self.db[session_info]
        for token in context_id:
            if token not in db:
                return None
            db = db[token].children
        if action not in db:
            return None
        node = db[action]
        if node.value_sum is None:
            return None

        output = {}
        output["value"] = node.value_sum
        policy = node.prior[0]
        actions = node.prior[1]
        # actions = []
        # policy = []
        # for child_action in node.children:
        #     child = node.children[child_action]
        #     assert child.action == child_action
        #     actions.append(child.action)
        #     policy.append(child.prior)
        output["action"] = actions
        output["policy"] = policy
        output["value"] = np.array(node.value_sum)
        return output

    def get_attention_mask(self, session_id):
        return self.attention_mask[session_id]

    def get_position_ids(self, session_id):
        return self.position_ids[session_id]

    def get_inference_params(self, session_id):
        return self.inference_params[session_id]

    def add_inference_params(self, session_id, inference_params):
        self.inference_params[session_id] = inference_params

    def delete(self, session_id):
        del self.db[session_id]
        self.node_id = 0
        del self.inference_params[session_id]
        del self.attention_mask[session_id]
        del self.position_ids[session_id]


def get_kv_cache(selected_actions, session_info, context_ids, search_db: SearchDB):
    batched_kv_cache = []
    batched_tokens = []
    for action, context_id in zip(selected_actions, context_ids):
        action = action.item()
        node = search_db.get(session_info, context_id)
        # assert node.action == action
        tokens = []
        tokens.append(action)
        new_kv_cache = {key: [] for key in node.state.keys()}
#         while node.parent is not None:
        while True:
            # if node.action is List
            if isinstance(node.action, list):
                # make a copy
                context_tokens = node.action.copy()
                # reverse the order
                context_tokens.reverse()
                tokens.extend(context_tokens)
            else:
                tokens.append(node.action)
            for key in node.state.keys():
                new_kv_cache[key].append(node.state[key])
            if node.parent is None:
                break
            node = node.parent
        # reverse the tokens order
        tokens.reverse()
        for key in new_kv_cache.keys():
            list_kv = new_kv_cache[key]
            keys = [item[0] for item in list_kv]
            vals = [item[1] for item in list_kv]
            # reverse the order
            keys.reverse()
            vals.reverse()
            keys = np.concatenate(keys, axis=0)
            vals = np.concatenate(vals, axis=0)
            new_kv_cache[key] = (keys, vals)
        batched_kv_cache.append(new_kv_cache)
        batched_tokens.append(np.array(tokens))
    full_length = torch.cuda.IntTensor([len(c) + 1 for c in context_ids])
    # before concat, make sure the batch size is the same
    max_depth = full_length.max()
    min_depth = full_length.min()
    paddings = max_depth - full_length

    token_list = []
    for token, padding in zip(batched_tokens, paddings):
        padding = padding.item()
        token = np.pad(token, ((0, padding),), "constant", constant_values=0)
        token_list.append(token)

    # concat tokens, shape [batch_size, length]
    tokens = np.stack(token_list, axis=0)
    tokens = tokens[:, -(max_depth - min_depth + 1) :]

    # concat kv cache
    output_kv_cache = {}
    for key in batched_kv_cache[0].keys():
        keys = []
        vals = []
        for item, padding in zip(batched_kv_cache, paddings):
            padding = padding.item()
            key_item = item[key][0][:, None, ...]
            key_item = np.pad(key_item, ((0, padding + 1), (0, 0), (0, 0), (0, 0)), "constant", constant_values=0)
            keys.append(key_item)
            value_item = item[key][1][:, None, ...]
            value_item = np.pad(value_item, ((0, padding + 1), (0, 0), (0, 0), (0, 0)), "constant", constant_values=0)
            vals.append(value_item)
        keys = np.concatenate(keys, axis=1)
        vals = np.concatenate(vals, axis=1)
        output_kv_cache[key] = (keys, vals)
    return output_kv_cache, tokens
