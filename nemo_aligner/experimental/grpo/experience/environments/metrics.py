import torch


def calculate_pass_rate_per_prompt(prompts, is_correct):
    """
    Function to compute fraction of prompts that have at least one correct answer
    (reward > 0).

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    is_correct: tensor (b,)       bool-valued label. May be on any device
    
    returns
    pass rate: float
    """
    unique_prompts = torch.unique(prompts, dim=0)

    correct_prompt_ct = 0
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        if torch.any(is_correct[is_matching_prompt] > 0):
            correct_prompt_ct += 1

    return correct_prompt_ct / len(unique_prompts)
