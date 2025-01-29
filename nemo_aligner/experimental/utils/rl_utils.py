import torch

def calculate_kl_penalty_joschu2020(log_probs_policy: torch.Tensor, log_probs_reference: torch.Tensor):
    """Calculates a per-token estimate of the KL Divergence between two log_probs.
    From Schulman 2020, always positive.
    
    log_probs_policy:    torch.Tensor (b, s)
    log_probs_reference: torch.Tensor (b, s)
    """
    r = log_probs_reference - log_probs_policy
    return torch.exp(r) - r - 1

# TODO @sahilj add unit test and normalization
def calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask, leave_one_out_baseline=True):
    """
    Function to compute a baseline for each (prompt, response) pair in the batch. 
    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask' 
    are not included in the baseline calculation. 

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute baseline by leaving out the sample that the baseline is for (from RLOO)
    
    returns
    tensor (b,) of baselines on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = "cpu"

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(reward_device)

        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            num_valid = valid_mask[prompt_idx].sum() - leave_one_out_baseline
            prompt_baseline = torch.matmul(baseline_mask_matrix, 
                                           rewards[prompt_idx] * valid_mask[prompt_idx]) / num_valid
            prompt_baseline_square = torch.matmul(baseline_mask_matrix, 
                                           (rewards[prompt_idx] ** 2) * valid_mask[prompt_idx]) / num_valid

            baseline[prompt_idx] = prompt_baseline
            sq_baseline[prompt_idx] = prompt_baseline_square

    std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
    return baseline, std


