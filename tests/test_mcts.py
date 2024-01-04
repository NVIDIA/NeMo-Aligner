import numpy as np
import torch
from megatron.core import InferenceParams

from nemo_aligner.utils.deep_search.mcts.mcts import Node, State
from nemo_aligner.utils.deep_search.mcts.search_db import SearchDB, get_kv_cache


class TestHistoryTrackingNode:
    def setup(self):
        # root inference single batch
        self.root_inference_params = InferenceParams(max_batch_size=10, max_sequence_length=100)
        self.root_inference_params.sequence_len_offset = 0
        self.root_inference_params.key_value_memory_dict = {
            1: (torch.rand(100, 1, 10, 10), torch.rand(100, 1, 10, 10)),
            2: (torch.rand(100, 1, 10, 10), torch.rand(100, 1, 10, 10)),
        }

        # child inference batch size K=50
        K = 50
        self.K = K
        self.child_inference_params = InferenceParams(max_batch_size=10, max_sequence_length=100)
        self.child_inference_params.sequence_len_offset = 11
        self.child_inference_params.key_value_memory_dict = {
            1: (torch.rand(100, K, 10, 10), torch.rand(100, K, 10, 10)),
            2: (torch.rand(100, K, 10, 10), torch.rand(100, K, 10, 10)),
        }

    def test_root_node(self):
        search_db = SearchDB()
        state = State.get_state(self.root_inference_params, 0, 11, 0)
        assert state.depth == 0
        assert state.kv_cache[1][0].shape == (11, 10, 10)
        assert state.kv_cache[1][1].shape == (11, 10, 10)
        assert state.kv_cache[2][0].shape == (11, 10, 10)
        assert state.kv_cache[2][1].shape == (11, 10, 10)
        root_node = Node(state=state, parent=None, action=None, prior=0.0, visit_count=0, C=2.0)
        search_db.add("session1", 0, -1, root_node)

        full_actions = np.arange(0, 1000, 1)
        # shuffle
        np.random.shuffle(full_actions)
        mock_actions = full_actions[: self.K]
        for k in range(self.K):
            action = mock_actions[k]
            child_state = State.get_state(self.child_inference_params, 1, 11, 0)
            assert child_state.depth == 1
            assert child_state.kv_cache[1][0].shape == (1, 10, 10)
            child_node = Node(state=child_state, parent=root_node, action=action, prior=0.0, visit_count=0, C=2.0)
            root_node.children.append(child_node)
            search_db.add("session1", 1, action, child_node)

        # construction kv cache for selected actions
        K = 30
        selected_actions = mock_actions[:K]
        updated_kv_cache = get_kv_cache(selected_actions, search_db)
        assert len(updated_kv_cache) == 2
        assert updated_kv_cache[1][0].shape == (13, 2, 10, 10)
        assert updated_kv_cache[1][1].shape == (13, 2, 10, 10)
