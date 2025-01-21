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


def calculate_kl_penalty_joschu2020(log_probs_policy, log_probs_reference):
    """Calculates a per-token estimate of the KL Divergence between two log_probs.
    From Schulman 2020, always positive.
    """
    r = log_probs_reference - log_probs_policy
    return torch.exp(r) - r - 1


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


def calculate_math_problem_wise_length_penalty(prompts, response_lengths, rewards, mask, penalty_amount):
    """
    Penalize responses that are longer than they need to be.
    Short responses that yield the correct answer for a problem set the baseline. The longest (correct) response
    recieves a penalty of penalty_amount. All other response length penalties are linearly interpolated from this.
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    reward_device = rewards.get_device()
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


def calculate_math_problem_wise_length_penalty(prompts, response_lengths, rewards, mask, penalty_amount):
    """
    Penalize responses that are longer than they need to be.
    Short responses that yield the correct answer for a problem set the baseline. The longest (correct) response
    receives a penalty of penalty_amount. All other response length penalties are linearly interpolated between 0 and penalty_amount.
    
    Args:
        prompts (torch.Tensor): Tensor of shape (batch_size, seq_len_prompt) 
                               indicating the tokenized prompts.
        response_lengths (torch.Tensor): Tensor of shape (batch_size, ) 
                                  indicating the tokenized response lengths
        rewards (torch.Tensor): Tensor of shape (batch_size,) 
                                indicating reward for each response. Typically 0/1 correctness or similar.
        mask (torch.Tensor): Tensor of shape (batch_size, ) 
                             that is 1/True for valid sequences 0/False otherwise. 
                             Summing across dim=1 gives the response length.
        penalty_amount (float): How large the penalty should be for the longest correct response. 
                                Other correct responses get a fraction of this penalty based on how their 
                                length compares to the shortest/longest correct responses.
    Returns:
        torch.Tensor: A new rewards tensor after applying the length-based penalty to correct responses.
    """

    # We will build a new rewards tensor rather than modify in-place.
    new_rewards = rewards.clone()
    reward_device = new_rewards.device

    # Identify each unique prompt
    unique_prompts = torch.unique(prompts, dim=0)

    for i in range(len(unique_prompts)):
        # Find all rows belonging to this prompt
        is_matching_prompt = (prompts == unique_prompts[i]).all(dim=1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]

        if mask[prompt_idx].sum() <= 1:
            continue

        # Compute response lengths for each example under this prompt
        # mask is assumed to be boolean or {0,1}, so sum(dim=1) gives the total length
        lengths = response_lengths[prompt_idx]

        # Identify which ones are correct
        # (Here, we assume reward > 0 => correct)
        correct_mask = new_rewards[prompt_idx] > 0.0
        correct_indices = torch.where(correct_mask)[0]

        # If there aren't at least two correct answers for this prompt, skip
        if len(correct_indices) < 2:
            continue

        # Among the correct ones, find min and max length
        correct_lengths = lengths[correct_indices]
        min_len = correct_lengths.min()
        max_len = correct_lengths.max()

        # If all correct responses have the same length, no penalty
        if min_len == max_len:
            continue

        # Penalize correct responses by how close their length is to max_len
        # - The shortest correct has penalty 0.
        # - The longest correct has penalty = penalty_amount.
        # - Everything in between is linearly scaled.
        length_range = max_len - min_len
        for ci in correct_indices:
            l = lengths[ci]
            penalty = penalty_amount * (l - min_len).float() / length_range.float()
            new_rewards[prompt_idx[ci]] -= penalty

    return new_rewards


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


def weight_by_correect(batch, weighting_mode="uniform"):
    """
    Function that returns positive scalars for correct answers and 0 for incorrect ones.
    With a weighting mode of 'balanced', we return a positive weight of 1 / (#correct answers 
    per identical prompt) for correct answers and 0 otherwise.
    With a weighting mode of 'uniform', we return 1.0 for correct and 0.0 for incorrect. 
    
    Returns (a torch tensor of weights, True if all responses are incorrect else False)
    """
    if weighting_mode == "uniform":
        return batch["rewards"], torch.sum(batch["rewards"]).item() == 0

    elif weighting_mode == "balanced":
        # unique_prompts = torch.unique(batch["prompt_tokens"], dim=0)
        # selected_idx = []

        # for i in range(len(unique_prompts)):
        # is_matching_prompt = (batch["prompt_tokens"] == unique_prompts[i]).all(1)
        # prompt_idx = torch.arange(len(batch["prompt_tokens"]))[is_matching_prompt]
        # sorted_idx = zip(prompt_idx, batch["rewards"][is_matching_prompt])
        # sorted_idx = sorted(sorted_idx, key=operator.itemgetter(1))
        # selected_idx += [x[0].item() for x in sorted_idx[-1 * num_select :]]

        # selected_batch = {k: batch[k][selected_idx] for k in batch.keys()}
        # return selected_batch
        return None, None

    else:
        raise NotImplementedError(f"weighting mode must be balanced or uniform. passed {weighting_mode}")


def online_prompt_filtering(rollout_batch, min_threshold=0.2, max_threshold=0.8):
    """
    Filters prompts based on accuracy thresholds, retaining only those within
    specified bounds. Calculates and returns a mask for valid sequences and
    corresponding accuracy metrics.

    Args:
        rollout_batch (dict): Contains 'prompt_tokens' and 'rewards' as torch.Tensors.
        min_threshold (float): Minimum accuracy required to retain a prompt.
        max_threshold (float): Maximum accuracy allowed to retain a prompt.

    Returns:
        tuple:
            - sequence_mask (torch.Tensor): Boolean mask for retained sequences.
            - accuracy_metrics (dict): Contains:
                - 'accuracy_min': Minimum accuracy.
                - 'accuracy_max': Maximum accuracy.
                - 'accuracy_avg': Average accuracy.
                - 'prompts_total': Total unique prompts.
                - 'prompts_dropped': Number dropped.
                - 'prompt_drop_rate': Drop rate.
    """

    prompt_tokens = rollout_batch["prompt_tokens"]
    rewards = rollout_batch["rewards"]

    # Find unique prompts
    unique_prompts = torch.unique(prompt_tokens, dim=0)

    # Initialize for storing statistics
    accuracy_stats = {"min": float("inf"), "max": float("-inf"), "avg": 0.0}

    total_dropped = 0
    # Create a mask for sequences to keep (initially all True)

    sequence_mask = torch.ones_like(rewards, dtype=torch.bool, device=rewards.device)

    # Calculate accuracy for each unique prompt
    for i in range(len(unique_prompts)):
        # Find all instances of this prompt
        is_matching_prompt = (prompt_tokens == unique_prompts[i]).all(dim=1)
        prompt_rewards = rewards[is_matching_prompt]

        # Calculate statistics
        total = prompt_rewards.size(0)
        correct = prompt_rewards.sum().item()
        accuracy = correct / total

        # Update accuracy stats
        accuracy_stats["min"] = min(accuracy_stats["min"], accuracy)
        accuracy_stats["max"] = max(accuracy_stats["max"], accuracy)
        accuracy_stats["avg"] += accuracy

        # Log prompt performance
        print(f"Prompt {unique_prompts[i]}: Accuracy={accuracy:.3f}, " f"Correct={correct}, Total={total}")

        # Keep track of prompts meeting threshold
        if not (min_threshold <= accuracy <= max_threshold):
            sequence_mask[is_matching_prompt] = False
            total_dropped += 1

    # Calculate final statistics
    num_unique_prompts = len(unique_prompts)
    accuracy_stats["avg"] /= num_unique_prompts
    prompts_kept = num_unique_prompts - total_dropped
    drop_rate = total_dropped / num_unique_prompts

    # Log summary statistics
    print(
        f"Accuracy Stats - Min: {accuracy_stats['min']:.3f}, "
        f"Max: {accuracy_stats['max']:.3f}, "
        f"Avg: {accuracy_stats['avg']:.3f}"
    )
    print(
        f"Prompt Stats - Total: {num_unique_prompts}, "
        f"Kept: {prompts_kept}, Dropped: {total_dropped}, "
        f"Drop Rate: {drop_rate:.3f}"
    )

    # Update metrics with accuracy and prompt statistics
    accuracy_metrics = {
        "accuracy_min": accuracy_stats["min"],
        "accuracy_max": accuracy_stats["max"],
        "prompts_total": num_unique_prompts,
        "prompts_kept": prompts_kept,
    }

    return sequence_mask, accuracy_metrics
