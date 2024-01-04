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


class SearchDB:
    def __init__(self):
        self.db = {}
        # node_id
        self.node_ids = {}

    def add(self, session_id, depth, action, node):
        key = (depth, action)
        if session_id not in self.db:
            session = {}
            self.db[session_id] = session
        else:
            session = self.db[session_id]
        session[key] = node

    def get(self, session_id, depth, action):
        return self.db[session_id][(depth, action)]

    def delete(self, session_id):
        del self.db[session_id]
        self.node_id = 0


def init_first_node(session_id, node_id, node, search_db):
    root_node = node
    search_db.add(session_id, node_id, node)


def get_kv_cache(selected_actions, search_db: SearchDB):
    batched_kv_cache = []
    for action in selected_actions:
        node = search_db.get("session1", 1, action)
        new_kv_cache = {key: [node.state.kv_cache[key]] for key in node.state.kv_cache.keys()}
        while node.parent is not None:
            node = node.parent
            for key in node.state.kv_cache.keys():
                new_kv_cache[key].append(node.state.kv_cache[key])
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
    # concat kv cache
    output_kv_cache = {}
    for key in batched_kv_cache[0].keys():
        keys = [item[key][0][:, None, ...] for item in batched_kv_cache]
        vals = [item[key][1][:, None, ...] for item in batched_kv_cache]
        keys = np.concatenate(keys, axis=1)
        vals = np.concatenate(vals, axis=1)
        # pad zero to axis=0
        keys = np.pad(keys, ((0, 1), (0, 0), (0, 0), (0, 0)), "constant", constant_values=0)
        vals = np.pad(vals, ((0, 1), (0, 0), (0, 0), (0, 0)), "constant", constant_values=0)
        output_kv_cache[key] = (keys, vals)
    return output_kv_cache
