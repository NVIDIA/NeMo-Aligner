from omegaconf import DictConfig
from concurrent import futures

import torch

from nemo_aligner.utils import parallel_state
from nemo_aligner.experimental.experience.interfaces import EnvironmentInterface
from nemo_aligner.servers.http_communicator import FlaskCommunicator

class MathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started MathEnvironment client with {cfg.servers}")
        
    #def start_step(self, interactions, metadata):
    def start_step(self, rollout_batch):
        """
        metadata: List[Dict]. Needs to contain a "ground_truth" key, which is what
                              the grader will use to evaluate correctness.
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            # responses = [''.join(interaction[1:]) for interaction in interactions]
            response_sentences = rollout_batch["response_sentences"]
            ground_truths = rollout_batch["ground_truths"]
            data = {
                "pred_responses": response_sentences,
                "ground_truths": ground_truths,
            }
            return self.communicator.send_data_to_server("math_grader", data)
        return None

    def finish_step(self, future):
        results = self.communicator.get_result(future, "rewards")
        return None, None, results["rewards"], torch.ones(results.shape[0],)
    
    def global_post_process_and_metrics(self, rollout_batch):
        return rollout_batch, {}