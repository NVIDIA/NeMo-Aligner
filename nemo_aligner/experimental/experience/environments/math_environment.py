from omegaconf import DictConfig
from concurrent import futures

import torch

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import masked_mean
from nemo_aligner.experimental.experience.interfaces import EnvironmentInterface
from nemo_aligner.experimental.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.servers.http_communicator import FlaskCommunicator

class MathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started MathEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata):
        """
        metadata: List[Dict]. Needs to contain a "ground_truth" key, which is what
                              the grader will use to evaluate correctness.
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            responses = [''.join(interaction[1:]) for interaction in interactions]
            ground_truths = [g["ground_truth"] for g in metadata]
            data = {
                "pred_responses": responses,
                "ground_truths": ground_truths,
            }
            return self.communicator.send_data_to_server("math_grader", data)
        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = self.communicator.get_result(future, "rewards")

        th_rewards = torch.tensor(results)
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
            "expected_answer": batch["ground_truths"][0],
        }
        correct_solution_generation_lengths = (
            (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
        )
        
        metrics = {
            #"table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], batch["rewards"]),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float.mean.item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        
        return batch, metrics