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

from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.experimental.grpo.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.servers.http_communicator import FlaskCommunicator
from nemo_aligner.utils.verifiers.math_grader import extract_answer
from nemo_aligner.experimental.grpo.experience.environments.format_checker import FormatChecker

class RewardModelEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started RewardModelEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata, is_end):
        """
        metadata: List[Dict] containing:
            - "prompt": The original problem prompt
            - "ground_truth": The ground truth answer (optional for reward model)
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            prompts = [interaction[0] for interaction in interactions]
            full_responses = [''.join(interaction[1:]) for interaction in interactions]
            
            # Source rank calculates format metrics
            format_rewards = FormatChecker.calculate_format_metrics(prompts, full_responses, is_end)
            
            responses = []
            for meta, interaction in zip(metadata, interactions):
                if meta.get("extract_box", False):
                    interaction = extract_answer(''.join(interaction[1:]))
                    if interaction is None:
                        interaction = ""
                else:
                    interaction = ''.join(interaction[1:])
                responses.append(interaction.split("</think>")[-1].strip())
            
            # Prepare data for verification
            data = {
                "prompts": prompts,
                "responses": responses,
                "format_rewards": format_rewards
            }
            print(f"Sending batch of {len(responses)} responses for reward model evaluation")
            return self.communicator.send_data_to_server("reward_model_judge", data)
        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = self.communicator.get_result(future, "rewards")

        th_rewards = torch.tensor(results).squeeze(1)
        print('Reward model rewards shape', th_rewards.shape)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed 
        calculations if you'd prefer for heavy metrics. 
        """
        # Sample data for debugging
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt_sentence": batch["prompt_sentences"][0],
            "response_sentence": batch["response_sentences"][0],
            "expected_answer": batch["extra_verifier_info"][0].get("ground_truth", "N/A")
        }
        print("### Sample:", table)
        
        # Set a reward of 0 for any incorrectly ended sequences
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        # Calculate average generation length for high-reward solutions (reward > 0.7)
        high_reward_mask = batch["rewards"] > 0.0
        if high_reward_mask.float().sum() > 0:
            high_reward_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[high_reward_mask].float().mean().item()
            )
        else:
            high_reward_generation_lengths = 0
        
        # Calculate format rewards for all prompt-response pairs
        format_rewards = FormatChecker.calculate_format_metrics(
            batch["prompt_sentences"],
            batch["response_sentences"],
            batch["is_end"]
        )
        
        # Calculate average reward
        avg_reward = batch["rewards"].mean().item()
        

        
        metrics = {
            "average_reward": avg_reward,
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "high_reward_generation_lengths": high_reward_generation_lengths,
            "format_rewards": torch.tensor(format_rewards).float().mean().item(),
        }
        
        return batch, metrics