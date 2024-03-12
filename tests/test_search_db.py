import numpy as np
import torch
from megatron.core import InferenceParams

from nemo_aligner.utils.deep_search.mcts.mcts import Node, get_state
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
        context_id = tuple(range(11))
        state = get_state(self.root_inference_params, True, 11, 0)
        assert state[1][0].shape == (11, 10, 10)
        assert state[1][1].shape == (11, 10, 10)
        assert state[2][0].shape == (11, 10, 10)
        assert state[2][1].shape == (11, 10, 10)
        root_node = Node(state=state, parent=None, action=list(context_id), prior=0.0, visit_count=0)
        search_db.add_root("session1", context_id, root_node)

        full_actions1 = np.arange(1, 1001, 1)
        # shuffle
        np.random.shuffle(full_actions1)
        mock_actions1 = full_actions1[: self.K]
        for k in range(self.K):
            action = mock_actions1[k]
            child_state = get_state(self.child_inference_params, False, 12, 0)
            assert child_state[1][0].shape == (1, 10, 10)
            child_node = Node(state=child_state, parent=root_node, action=action, prior=0.0, visit_count=0)
            root_node.children[action] = child_node
        depth1_child = child_node

        self.child_inference_params.sequence_len_offset += 1
        full_actions2 = np.arange(1, 1001, 1)
        # shuffle
        np.random.shuffle(full_actions2)
        mock_actions2 = full_actions2[: self.K]
        child_context_id = context_id + (depth1_child.action,)
        for k in range(self.K):
            action = mock_actions2[k]
            child_state = get_state(self.child_inference_params, False, 13, 0)
            assert child_state[1][0].shape == (1, 10, 10)
            child_node = Node(state=child_state, parent=depth1_child, action=action, prior=0.0, visit_count=0)
            depth1_child.children[action] = child_node

        # construction kv cache for selected actions
        K = 30
        selected_actions = np.concatenate([mock_actions1[: K // 2], mock_actions2[: K // 2]], axis=0)
        context_ids = [context_id] * (K // 2) + [child_context_id] * (K // 2)
        selected_actions = torch.tensor(selected_actions).cuda().unsqueeze(1)
        updated_kv_cache, tokens = get_kv_cache(selected_actions, "session1", context_ids, search_db)
        assert len(updated_kv_cache) == 2
        assert updated_kv_cache[1][0].shape == (14, 30, 10, 10)
        assert updated_kv_cache[1][1].shape == (14, 30, 10, 10)
        assert tokens.shape == (K, 2)
