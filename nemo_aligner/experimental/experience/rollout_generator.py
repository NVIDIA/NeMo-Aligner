import os
import json
import hashlib
from typing import Dict
from contextlib import nullcontext
from omegaconf import DictConfig

import torch

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import batch_repeat, batch_index_select, cpu_dict
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.parallel_state import is_trt_llm_reshard, trt_llm_reshard_region
from nemo_aligner.experimental.experience.interfaces import RolloutGeneratorInterface, EnvironmentInterface
from nemo_aligner.experimental.experience.rollout_batch import GPTRolloutBatch

class SequenceRewardRolloutGenerator(RolloutGeneratorInterface):
    def __init__(self, cfg: DictConfig, tasks_to_environments: Dict[str, EnvironmentInterface]):
        """
        tasks_to_environments: A mapping of task string names (as will be pulled from the 
                               sample "task_name" field) to an Environment object that can 
                               process that task.
        cfg:                   DictConfig that needs to define "samples_per_prompt" and 
                               "prompt_batch_size", and "rollout_mbs". 
                               prompt_batch_size must divide rollout_mbs and 
                               rollout_mbs // prompt_batch_size >= samples_per_prompt
        """
        self.tasks_to_environments = tasks_to_environments
        self.samples_per_prompt = cfg.samples_per_prompt
        self.prompt_batch_size = cfg.prompt_batch_size
        self.rollout_mbs = cfg.rollout_mbs
        self.val_samples_per_prompt = cfg.val_samples_per_prompt
        self.val_prompt_batch_size = cfg.val_prompt_batch_size
        self.rollout_batch_seq_length = cfg.rollout_batch_seq_length
        
        assert self.rollout_mbs % self.prompt_batch_size == 0, \
            (f"The inference microbatch size of the model ({self.rollout_mbs}) must be a ",
             f"clean multiple of the prompt_batch_size ({self.prompt_batch_size}) from the",
             f" dataloader.")
        assert self.rollout_mbs % self.val_prompt_batch_size == 0, \
            (f"The inference microbatch size of the model ({self.rollout_mbs}) must be a ",
             f"clean multiple of the val_prompt_batch_size ({self.val_prompt_batch_size}) from the",
             f" dataloader.")

        assert self.samples_per_prompt % (self.rollout_mbs // self.prompt_batch_size) == 0, \
            (f"The number of total samples per prompt ({self.samples_per_prompt}) must be a",
             f" multiple of the number of samples per prompt in each microbatch \n\n",
             f"Samples per prompt per microbatch = rollout_mbs / prompt_batch_size\n",
             f"{self.rollout_mbs} / {self.prompt_batch_size} = {self.rollout_mbs / self.prompt_batch_size}\n",
             f"{self.samples_per_prompt} % {self.rollout_mbs // self.prompt_batch_size} != 0.")
        assert self.val_samples_per_prompt % (self.rollout_mbs // self.val_prompt_batch_size) == 0, \
            (f"The number of total val samples per prompt ({self.val_samples_per_prompt}) must be a",
             f" multiple of the number of val samples per prompt in each microbatch \n\n",
             f"val samples per prompt per microbatch = rollout_mbs / val_prompt_batch_size\n",
             f"{self.rollout_mbs} / {self.val_prompt_batch_size} = {self.rollout_mbs / self.val_prompt_batch_size}\n",
             f"{self.val_samples_per_prompt} % {self.rollout_mbs // self.val_prompt_batch_size} != 0.")

    def generation_log(self, rollout_batch, savefile):
        """
        Logs all generations and related metadata to savefile. The savefile should be unique per rank.
        Runs on all ranks. We assume that the input "rollout_batch" is global here.
        """

        def dict_get(batch, idx):
            inst = {}
            for k in batch.keys():
                inst[k] = batch[k][idx]
            return inst

        batch_size = rollout_batch["prompt_tokens"].shape[0]
        per_rank_batch = batch_size // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        end_idx = (rank + 1) * per_rank_batch if rank < torch.distributed.get_world_size() - 1 else batch_size
        records = []
        for idx in range(rank * per_rank_batch, end_idx):
            # get individual element of batch
            instance = dict_get(rollout_batch, idx)
            response_tokens = instance["response_tokens"]
            response_length = instance["response_lengths"].item()
            prompt_length = instance["prompt_lengths"].item()
            ground_truth = instance["ground_truths"]
            is_end = instance["is_end"].item()
            reward = instance["rewards"].item()
            problem_str = self.model.tokenizer.ids_to_text(response_tokens[:prompt_length].tolist())
            response_str = self.model.tokenizer.ids_to_text(response_tokens[prompt_length:].tolist())
            problem_hash = hashlib.sha256(problem_str.encode("utf-8")).hexdigest()
            generator_rank = instance["generator_rank"].item()
            lps = instance["logprobs"][prompt_length:response_length].cpu().tolist()
            init_lps = instance["init_logprobs"][prompt_length:response_length].cpu().tolist()

            record = {
                "problem_hash": problem_hash,
                "idx_in_batch": idx,
                "problem": problem_str,
                "response": response_str,
                "response_length": response_length,
                "ground_truth": ground_truth,
                "rank": generator_rank,
                "ended_correctly": is_end,
                "step": self.reinforce_optimization_step,
                "reward": reward,
                "accuracy": instance["accuracy"].item(),
                "baseline": instance["baseline"].item(),
                "logprobs": lps,
                "init_logprobs": init_lps,
            }
            records.append(record)

        with open(savefile, "a", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

    def generate_rollouts(self, batch_iterator, policy_model, timer, is_validation=False):
        """
        Generate experience rollouts using the policy model and environments.
        Currently only supports single-turn (no feedback)
        """
        # Set up an inference environment with less sharding/parallelism to accelerate inference
        inference_reshard_context = trt_llm_reshard_region if self.trtllm_reshard else nullcontext

        rollout_batches, futures = [], []
        with inference_reshard_context():
            with timer("generate"):
                for batch in batch_iterator:
                    if is_validation:
                        num_repetitions = self.rollout_mbs // self.val_prompt_batch_size
                        num_rollout_batches_per_data_batch = self.val_samples_per_prompt // num_repetitions
                    else:
                        num_repetitions = self.rollout_mbs // self.prompt_batch_size
                        num_rollout_batches_per_data_batch = self.samples_per_prompt // num_repetitions

                    print(batch["problem"].shape)
                    batch = batch_repeat(batch, num_repetitions=num_repetitions)
                    for _ in range(num_rollout_batches_per_data_batch):
                        rollout_batch = policy_model.infer(batch)
                        rollout_batch["prompt_tokens"] = batch["prompt"]
                        rollout_batch["generator_rank"] = (
                            torch.ones(batch["problem"].shape[0]) * parallel_state.get_model_parallel_src_rank()
                        )

                        # iterate over tasks and call the environments to get rewards
                        microbatch_futures = []
                        for task in self.tasks_to_environments.keys():
                            indices = []
                            for idx, t in enumerate(rollout_batch["tasks"]):
                                if t == task:
                                    indices.append(idx)
                            task_batch = batch_index_select(rollout_batch, indices)
                            microbatch_futures.append((task, indices, self.tasks_to_environments[task].start_step(task_batch, None)))

                        rollout_batches.append(rollout_batch)
                        futures.append(microbatch_futures)

            # The batch_iterator may be a load-redistributing server, so batches may be jagged.
            # We gather everything so that we can rebalance it.
            unbalanced_local_batch = GPTRolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=policy_model.tokenizer.eos_id,
                rollout_batch_seq_length=self.cfg.rollout_batch_seq_length,
            )
            global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()

        padded_rollout_sequence_length = global_rollout_batch["response_tokens"].size(-1)

        # the chunking must be outside of the inference TRT-LLM context because we do 
        # logprob calculation in nemo with training sharding
        balanced_local_batch = global_rollout_batch.chunk(
            rank=parallel_state.get_data_parallel_rank(),
            split_size=parallel_state.get_data_parallel_world_size(),
        )

        # since we compute the logprobs in nemo we need to be outside the inference resharding context
        batched_response_tokens = balanced_local_batch["response_tokens"]

        with timer("logprobs"):
            rollout_logprobs = policy_model.get_inference_log_probs(batched_response_tokens)
            balanced_local_batch["logprobs"] = rollout_logprobs

        # compute init logprobs only if not in validation
        if not is_validation:
            with timer("init_logprobs"):
                rollout_init_logprobs = policy_model.get_init_policy_logprobs(batched_response_tokens)
                balanced_local_batch["init_logprobs"] = rollout_init_logprobs
        global_rollout_batch = balanced_local_batch.gather_and_balance_globally()

        # we send environment step requests in the sharded context, so we need to keep this sharding and then undo it
        with inference_reshard_context():
            with timer("environment_wait"):
                # TODO reassemble environment futures
                rm_rollout_batches = []
                for future in futures:
                    rewards = future.result().squeeze(1)
                    rm_rollout_batches.append({"rewards": rewards})

            unbalanced_rm_batch = GPTRolloutBatch.from_rollout_batches(
                rm_rollout_batches,
                eos_id=self.model.tokenizer.eos_id,
                rollout_batch_seq_length=padded_rollout_sequence_length,
            )
            global_rm_batch = unbalanced_rm_batch.gather_and_balance_globally()

            g_prompt_tokens = global_rollout_batch["prompt_tokens"]
            g_response_lengths = global_rollout_batch["response_lengths"]
            g_rewards = global_rm_batch["rewards"]
            g_is_end = global_rollout_batch["is_end"]
            global_rm_batch["accuracy"] = g_rewards

        # chunking needs to be outside of reshard region
        balanced_rm_batch = global_rm_batch.chunk(
            rank=parallel_state.get_data_parallel_rank(),
            split_size=parallel_state.get_data_parallel_world_size(),
        )
        balanced_local_batch.update(balanced_rm_batch)

        global_rollout_batch["mask"] = create_mask(values=global_rollout_batch["logprobs"], prompt_lengths=global_rollout_batch["prompt_lengths"], response_lengths=global_rollout_batch["response_lengths"])

        global_rollout_batch.update(global_rm_batch)
        rank = torch.distributed.get_rank()
        savefile = f"train_{rank}.jsonl" if not is_validation else f"validation_{rank}.jsonl"
        self.generation_log(global_rollout_batch, os.path.join(self.cfg.generation_save_dir, savefile))

        return balanced_local_batch, cpu_dict(self.compute_rollout_metrics(global_rollout_batch))