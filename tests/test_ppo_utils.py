# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn.functional as F

from nemo_aligner.utils.ppo_utils import calculate_advantages_and_returns, calculate_entropy, calculate_ppo_rewards


class TestCalculateEntropy:
    def test_calculate_entropy_with_no_mask(self):
        log_probs = F.log_softmax(torch.randn(4, 2048, 4096, dtype=torch.float32), dim=-1)
        entropy_mine = calculate_entropy(log_probs, mask=None)

        entropy_py = torch.distributions.categorical.Categorical(logits=log_probs).entropy().mean()
        assert torch.allclose(
            entropy_py, entropy_mine
        ), f"expected entropy without mask to be {entropy_py} but got {entropy_mine}"

    def test_calculate_entropy_with_mask(self):
        log_probs = F.log_softmax(torch.randn(4, 2048, 4096, dtype=torch.float32), dim=-1)
        mask = torch.randint(low=0, high=2, size=(4, 2048), dtype=torch.float32)

        entropy_mine_with_mask = calculate_entropy(log_probs, mask=mask)
        entropy_py_with_mask = (
            (torch.distributions.categorical.Categorical(logits=log_probs).entropy() * mask).sum() / mask.sum()
        ).to(entropy_mine_with_mask.dtype)

        assert torch.allclose(
            entropy_py_with_mask, entropy_mine_with_mask
        ), f"expected entropy with mask to be {entropy_py_with_mask} but got {entropy_mine_with_mask}"

    def test_calculate_entropy_small_example(self):
        log_probs = torch.as_tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).view(1, 1, -1).log()

        # output is in nats, convert it to bits
        output = (calculate_entropy(log_probs) * math.log2(math.e)).item()

        assert output == 2, f"expected output to be 2 bits but got {output}"


class TestCalculatePPORewards:
    """Note: the way this is calculated for say 5 tokens is by doing
    [0 0 0 R 0] instead of [0 0 0 0 R] to align with the transformer
    outputs. Since tranformers tries to predict the *next* token
    """

    def test_calculate_ppo_rewards_with_no_kl_penalty(self):
        rewards = torch.as_tensor([1, 2], dtype=torch.float32)
        sentence_lengths = torch.as_tensor([5, 8], dtype=torch.long)
        init_policy_kl = torch.randn(2, 10)

        ppo_rewards = calculate_ppo_rewards(init_policy_kl, rewards, sentence_lengths, init_policy_kl, 0)

        reward_0 = ppo_rewards[0][sentence_lengths[0] - 2]
        reward_1 = ppo_rewards[1][sentence_lengths[1] - 2]

        assert reward_0 == rewards[0], f"expected reward without kl at idx 0 to be {reward_0} but got {rewards[0]}"
        assert reward_1 == rewards[1], f"expected reward without kl at idx 1 to be {reward_1} but got {rewards[1]}"

    def test_calculate_ppo_rewards_with_kl_penalty(self):
        kl_penalty_factor = 0.01

        rewards = torch.as_tensor([1, 2], dtype=torch.float32)
        sentence_lengths = torch.as_tensor([5, 8], dtype=torch.long)
        init_policy_kl = torch.randn(2, 10)

        ppo_rewards = calculate_ppo_rewards(
            init_policy_kl, rewards, sentence_lengths, init_policy_kl, kl_penalty_factor
        )

        reward_0 = ppo_rewards[0][sentence_lengths[0] - 2]
        reward_1 = ppo_rewards[1][sentence_lengths[1] - 2]
        target_reward_0 = rewards[0] - (kl_penalty_factor * init_policy_kl[0, sentence_lengths[0] - 2])
        target_reward_1 = rewards[1] - (kl_penalty_factor * init_policy_kl[1, sentence_lengths[1] - 2])

        assert (
            target_reward_0 == reward_0
        ), f"expected reward with kl at idx 0 to be {target_reward_0} but got {reward_0}"

        assert (
            target_reward_1 == reward_1
        ), f"expected reward with kl at idx 1 to be {target_reward_1} but got {reward_1}"

        # these are the locs with the rewards so we get rid of them
        # by swapping in the init kl
        B = ppo_rewards.size(0)
        ppo_rewards[torch.arange(B), sentence_lengths - 2] = (
            -kl_penalty_factor * init_policy_kl[torch.arange(B), sentence_lengths - 2]
        )

        assert torch.allclose(
            ppo_rewards, -kl_penalty_factor * init_policy_kl
        ), "ppo_rewards on is not aligned with the init policy kl on positions where rewards is 0"


class TestCalculateAdvantagesAndReturns:
    def test_calculate_advantage_and_returns_small_example(self):
        gae_lambda = 0
        discount_factor = 0.5
        values = torch.arange(4, dtype=torch.float32).view(1, 4)
        rewards = torch.as_tensor([0, 0, 1, 0], dtype=torch.float32).view(1, 4)

        advantages, returns = calculate_advantages_and_returns(values, rewards, discount_factor, gae_lambda)

        gt_advantages = torch.as_tensor([0.5, 0, 0.5, -3], dtype=torch.float32)

        assert torch.allclose(advantages, gt_advantages), "computed advantage is not the same as hand example"
        assert torch.allclose(returns, gt_advantages + values), "computed returns is not the same as hand example"
