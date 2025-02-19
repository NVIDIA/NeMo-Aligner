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

from nemo_aligner.utils import parallel_state
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.experimental.grpo.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.servers.http_communicator import FlaskCommunicator
from nemo_aligner.utils.verifiers.math_grader import extract_answer

class LLMJudgeEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started LLMJudgeEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata):
        """
        metadata: List[Dict] containing:
            - "prompt": The original problem prompt
            - "ground_truth": The ground truth answer
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            responses = []
            for meta, interaction in zip(metadata, interactions):
                if meta["extract_box"]:
                    interaction = extract_answer(interaction)
                    if interaction is None:
                        interaction = ""
                else:
                    interaction = ''.join(interaction[1:])
                responses.append(interaction)
            
            # Prepare data for verification
            data = {
                "prompts": [meta["prompt"] for meta in metadata],
                "responses": responses,
                "ground_truths": [meta["ground_truth"] for meta in metadata],
            }
            
            return self.communicator.send_data_to_server("llm_verifier", data)
        return None

    def finish_step(self, future):
        results = self.communicator.get_result(future, "rewards")
        
        th_rewards = torch.tensor(results["rewards"]).squeeze(1)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed 
        calculations if you'd prefer for heavy metrics. 
        """
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt_sentence": batch["prompt_sentences"][0],
            "response_sentence": batch["response_sentences"][0],
            "expected_answer": batch["extra_verifier_info"][0]["ground_truth"],
            "explanation": batch["extra_verifier_info"]["explanations"][0] if "explanations" in batch["extra_verifier_info"] else None,
        }
        
        # Set a reward of 0 for any incorrectly ended sequences
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        # Calculate average generation length for correct solutions
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0
        
        metrics = {
            #"table": table, # TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], batch["rewards"]),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        
        return batch, metrics