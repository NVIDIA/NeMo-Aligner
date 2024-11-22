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

import operator

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


def select_topk(batch, num_select=1):
    """
    Function to select the topk responses for each unique prompt in a batch. 
    Please note that this function samples the same top response for each identical prompt.
    Duplicate prompts in the same batch may cause unexpected behavior.
    """
    unique_prompts = torch.unique(batch["prompt_tokens"], dim=0)
    selected_idx = []

    for i in range(len(unique_prompts)):
        is_matching_prompt = (batch["prompt_tokens"] == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(batch["prompt_tokens"]))[is_matching_prompt]
        sorted_idx = zip(prompt_idx, batch["rewards"][is_matching_prompt])
        sorted_idx = sorted(sorted_idx, key=operator.itemgetter(1))
        selected_idx += [x[0].item() for x in sorted_idx[-1 * num_select :]]

    selected_batch = {k: batch[k][selected_idx] for k in batch.keys()}
    return selected_batch


def calculate_rloo_baseline(prompts, reward, mask):
    """
    Function to select the RLOO baseline for each (prompt, response) pair in the batch. 
    The same baseline is calculated for each prompt. Masked samples are not included
    in the baseline calculation.
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(reward)
    reward_device = reward.get_device()
    if reward_device == -1:
        reward_device = "cpu"

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
        rloo_mat = (1 - torch.eye(len(prompt_idx))).to(reward_device)

        if mask[prompt_idx].sum() <= 1:
            # Ignore sample: set baseline equal to reward
            baseline[prompt_idx] = reward[prompt_idx]
        else:
            rloo = torch.matmul(rloo_mat, reward[prompt_idx] * mask[prompt_idx]) / (mask[prompt_idx].sum() - 1)
            baseline[prompt_idx] = rloo

    return baseline
