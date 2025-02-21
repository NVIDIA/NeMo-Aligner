# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig
from concurrent import futures

import torch
from collections import Counter

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import masked_mean
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.servers.http_communicator import FlaskCommunicator
from nemo_aligner.experimental.grpo.experience.environments.format_checker import FormatChecker

class CodeEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)

        
        print(f"Started CodeEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata, is_end):
        """
        metadata: List[Dict]. Needs to contain either:
            - For IO tests: "unittests" and optionally "fn_name"
            - For assertion tests: "unittests"
            Also needs "test_type" field specifying "io_test" or "assertion"
        """
        if parallel_state.is_model_parallel_src_rank():
            prompts = [interaction[0] for interaction in interactions]
            responses = [''.join(interaction[1:]) for interaction in interactions]
            
            # Source rank calculates format metrics
            format_rewards = FormatChecker.calculate_format_metrics(prompts, responses, is_end)
            
            responses = [r.split("</think>")[-1].split("```python")[-1].split("```")[0].strip() for r in responses]

            # Prepare test data for each response
            test_data = []
            for meta in metadata:
                test_config = {
                    "test_type": meta["test_type"],
                }
                test_config["unittests"] = meta["unittests"]
                if "fn_name" in meta:
                    test_config["fn_name"] = meta["fn_name"]
                
                test_data.append(test_config)
            
            data = {
                "pred_responses": responses,
                "test_data": test_data,
                "format_rewards": format_rewards
            }
            print(prompts)
            print(data)
            return self.communicator.send_data_to_server("code_verifier", data)
        
        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = self.communicator.get_result(future, "rewards")

        th_rewards = torch.tensor(results).squeeze(1)
        print('th rewards shape', th_rewards.shape)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.
        """
        # Example response for debugging
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt": batch["prompt_sentences"][0],
            "response": batch["response_sentences"][0],
            "test_details": batch["extra_verifier_info"][0] if batch["extra_verifier_info"] else None
        }
        
        # Apply rewards only to properly ended sequences
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        # Group rewards by prompt for per-prompt metrics
        unique_prompts = {}
        for idx, prompt in enumerate(batch["prompt_sentences"]):
            if prompt not in unique_prompts:
                unique_prompts[prompt] = []
            unique_prompts[prompt].append(batch["rewards"][idx].item())
        
        # Calculate per-prompt metrics
        num_prompts = len(unique_prompts)
        prompts_with_perfect_solution = 0
        prompts_with_reward_signal = 0
        reward_range_per_prompt = []
        
        for prompt_rewards in unique_prompts.values():
            # Check for perfect solutions (reward = 1.0)
            if any(reward == 1.0 for reward in prompt_rewards):
                prompts_with_perfect_solution += 1
                
            # Check for reward signal (variance in rewards)
            if len(prompt_rewards) > 1 and not all(r == prompt_rewards[0] for r in prompt_rewards):
                prompts_with_reward_signal += 1
                
            # Calculate reward statistics per prompt
            if len(prompt_rewards) > 0:
                reward_range_per_prompt.append(max(prompt_rewards) - min(prompt_rewards))
        
        
        # Calculate average generation length for correct solutions
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0
        
        
        # Calculate format rewards for all prompt-response pairs
        format_rewards = FormatChecker.calculate_format_metrics(
            batch["prompt_sentences"],
            batch["response_sentences"],
            batch["is_end"]
        )
        
        metrics = {
            #"table": table,  # TODO: Implement table logging if needed
            
            # Overall accuracy metrics
            "accuracy": batch["rewards"].mean().item(),
            
            # RL-specific metrics
            "fraction_prompts_with_perfect_solution": prompts_with_perfect_solution / num_prompts,
            "avg_reward_range_per_prompt": sum(reward_range_per_prompt) / len(reward_range_per_prompt) if reward_range_per_prompt else 0,
            
            # Generation statistics
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            
            # Add format metrics
            "format_rewards": torch.tensor(format_rewards).float().mean().item(),
        }
        
        return batch, metrics