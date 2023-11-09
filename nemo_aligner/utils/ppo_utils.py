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

"""helper functions for PPO training"""

import torch
from nemo_aligner.utils.utils import masked_mean


def calculate_advantages_and_returns(values, rewards, discount_factor, gae_lambda, mask=None):
    """calculate the per token advantages and returns for the entire sequence

    Args:
        values, rewards (torch.Tensor): shape of B x (S-1)
    """
    if mask is not None:
        # need the masking here because our sentence might not span the entire sequence length
        values = values * mask
        rewards = rewards * mask

    last_gae_lam = 0
    advantages = torch.zeros_like(rewards)
    max_seq_len = values.size(-1)

    for i in reversed(range(max_seq_len)):
        if i == max_seq_len - 1:
            next_values = 0.0  # Last element has next_value==0.0
        else:
            next_values = values[:, i + 1]  # Get value from next position.
        delta = rewards[:, i] + discount_factor * next_values - values[:, i]
        last_gae_lam = delta + discount_factor * gae_lambda * last_gae_lam
        advantages[:, i] = last_gae_lam

    returns = advantages + values
    return advantages, returns


def calculate_entropy(log_probs, mask=None):
    """calculate the entropy, with an optional mask

    Args:
        log_probs (torch.Tensor): Tensor of log probs with shape [B x S x V]
        mask (torch.Tensor): Tensor of masks on the sequence length with shape B x S
    """
    entropy_unmasked = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    return entropy_unmasked.mean() if mask is None else masked_mean(entropy_unmasked, mask)


def calculate_ppo_rewards(values, rewards, response_lengths, init_policy_kl, penalty_factor=0.0):
    """the reward should be defined on the last valid action"""
    rewards_sequence = torch.zeros_like(values)

    idx = (response_lengths - 2).clamp(min=0, max=None)
    rewards_sequence[torch.arange(rewards_sequence.size(0)), idx] = rewards.flatten()
    return rewards_sequence - penalty_factor * init_policy_kl


def calculate_kl_penalty(log_probs_a, log_probs_b, use_absolute_kl=True):
    """Calculates a per-token estimate of the KL Divergence between two log_probs.
    """
    init_policy_kl = log_probs_a - log_probs_b
    if use_absolute_kl:
        init_policy_kl = init_policy_kl.abs()

    return init_policy_kl


def create_mask(values, prompt_lengths, response_lengths):
    """Creates a mask to only keep the values in the sequence that are between prompt_lengths and sentence_lengths.
    This results in removing the prompt tokens, and removing the padding at the end of the sequence.
    """
    mask = torch.zeros_like(values)

    for i in range(mask.size(0)):
        # Do prompt_length - 1 to remove the first log prob. But keep sentence_length
        # as it is because we want to include one EOS token.
        mask[i, prompt_lengths[i] - 1 : response_lengths[i] - 1] = 1.0
    return mask
