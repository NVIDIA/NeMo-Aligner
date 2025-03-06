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
import os
import json
import hashlib
from typing import Dict
from contextlib import nullcontext
from omegaconf import DictConfig

import torch

from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.utils.utils import batch_repeat, batch_index_select, reconstruct_split_batch, cpu_dict
from ..utils.parallel_state import is_inference_reshard, inference_reshard_region
from nemo_aligner.utils.distributed import ScopedTimer
from nemo_aligner.experimental.grpo.experience.interfaces import RolloutGeneratorInterface, EnvironmentInterface
from nemo_aligner.experimental.grpo.experience.rollout_batch import GPTRolloutBatch
from nemo_aligner.utils.utils import log_memory

class SuperSimpleRolloutGenerator(RolloutGeneratorInterface):
    def __init__(self, cfg: DictConfig, tasks_to_environments: Dict[str, EnvironmentInterface]):
        self.tasks_to_environments = tasks_to_environments
        self.samples_per_prompt = cfg.samples_per_prompt
        self.prompt_batch_size = cfg.prompt_batch_size
        self.rollout_batch_seq_length = cfg.rollout_batch_seq_length
    
    # TODO @sahil have a basic example 
    

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
        self.prompt_batch_size = cfg.prompt_micro_batch_size
        self.rollout_mbs = cfg.generation_rollout_mbs
        self.val_samples_per_prompt = cfg.val_samples_per_prompt
        self.val_prompt_batch_size = cfg.val_prompt_micro_batch_size
        self.reshard_weights_for_generation = cfg.inference_backend.enable and cfg.inference_backend.reshard
        self.generation_save_dir = cfg.generation_save_dir
        self.rollout_batch_seq_length = cfg.rollout_batch_seq_length
        self.timer = ScopedTimer()
        
        assert self.rollout_mbs % self.prompt_batch_size == 0, \
            (f"The generation microbatch size of the model ({self.rollout_mbs}) must be a ",
             f"multiple of the prompt_batch_size ({self.prompt_batch_size}) from the dataloader.")
        assert self.rollout_mbs % self.val_prompt_batch_size == 0, \
            (f"The generation microbatch size of the model ({self.rollout_mbs}) must be a ",
             f"multiple of the val_prompt_batch_size ({self.val_prompt_batch_size}) from the dataloader.")

        assert self.samples_per_prompt % (self.rollout_mbs // self.prompt_batch_size) == 0, \
            (f"The number of total samples per prompt ({self.samples_per_prompt}) must be a ",
             f"multiple of the number of samples per prompt in each microbatch \n\n",
             f"Samples per prompt per microbatch = rollout_mbs / prompt_batch_size\n",
             f"{self.rollout_mbs} / {self.prompt_batch_size} = {self.rollout_mbs / self.prompt_batch_size}\n",
             f"{self.samples_per_prompt} % {self.rollout_mbs // self.prompt_batch_size} != 0.")
        assert self.val_samples_per_prompt % (self.rollout_mbs // self.val_prompt_batch_size) == 0, \
            (f"The number of total val samples per prompt ({self.val_samples_per_prompt}) must be a",
             f" multiple of the number of val samples per prompt in each microbatch \n\n",
             f"val samples per prompt per microbatch = rollout_mbs / val_prompt_batch_size\n",
             f"{self.rollout_mbs} / {self.val_prompt_batch_size} = {self.rollout_mbs / self.val_prompt_batch_size}\n",
             f"{self.val_samples_per_prompt} % {self.rollout_mbs // self.val_prompt_batch_size} != 0.")

    def detokenize(self, policy_model, rollout_batch):
        response_tokens = rollout_batch["response_tokens"]
        response_lengths = rollout_batch["response_lengths"]
        prompt_tokens = rollout_batch["text"]
        prompt_lengths = rollout_batch["prompt_lengths"]

        prompt_sentences = [
            policy_model.tokenizer.ids_to_text(prompt_tokens[i][:prompt_lengths[i]].tolist())
            for i in range(prompt_lengths.shape[0])
        ]
        response_sentences = [
            policy_model.tokenizer.ids_to_text(response_tokens[i][prompt_lengths[i] : response_lengths[i]].tolist())
            for i in range(response_lengths.shape[0])
        ]
        
        rollout_batch["prompt_sentences"] = prompt_sentences
        rollout_batch["response_sentences"] = response_sentences
        return rollout_batch
    
    def prepare_env_state(self, rollout_batch):
        interactions = [[p, r] for p, r in zip(rollout_batch["prompt_sentences"], rollout_batch["response_sentences"])]
        metadata = rollout_batch["extra_verifier_info"]
        return interactions, metadata, rollout_batch["is_end"]
        
    def generate_rollouts(self,
                          batch_iterator,
                          policy_model,
                          is_validation=False,
                          greedy=False):
        log_memory("rollout_generator.py generate_rollouts START")
        """
        Generate experience rollouts using the policy model and environments.
        Currently only supports single-turn (no environment feedback).

        Returns a global rollout batch (sharded out later by the GRPO trainer)
        """
        # Set up a generation environment with less sharding/parallelism to accelerate it
        generation_reshard_context = inference_reshard_region if self.reshard_weights_for_generation else nullcontext

        # policy generation
        rollout_batches, futures = [], []
        with generation_reshard_context():
            with self.timer("generate"):
                for batch in batch_iterator:
                    if is_validation:
                        num_repetitions = self.rollout_mbs // self.val_prompt_batch_size
                        num_rollout_batches_per_data_batch = self.val_samples_per_prompt // num_repetitions
                    else:
                        num_repetitions = self.rollout_mbs // self.prompt_batch_size
                        num_rollout_batches_per_data_batch = self.samples_per_prompt // num_repetitions

                    print(batch["text"].shape)
                    print(f"batch idxs: {batch['idx']}", flush=True)
                    batch = batch_repeat(batch, num_repetitions=num_repetitions)
                    for _ in range(num_rollout_batches_per_data_batch):
                        rollout_batch = policy_model.infer(batch, use_greedy=greedy)
                        log_memory("rollout_generator.py generate_rollouts policy_model.infer")

                        rollout_batch = batch | rollout_batch
                        rollout_batch["generator_rank"] = (
                            torch.ones(batch["text"].shape[0]) * parallel_state.get_model_parallel_src_rank()
                        )
                        rollout_batch = self.detokenize(policy_model, rollout_batch)

                        # iterate over tasks and call the environments to get rewards
                        microbatch_futures = []
                        for task in sorted(self.tasks_to_environments.keys()):
                            indices = []
                            for idx, t in enumerate(rollout_batch["task_name"]):
                                if t == task:
                                    indices.append(idx)
                            if len(indices) > 0:
                                task_batch = batch_index_select(rollout_batch, indices)
                                interactions, metadata, is_end = self.prepare_env_state(task_batch)
                                microbatch_futures.append((task, indices, self.tasks_to_environments[task].start_step(interactions, metadata, is_end)))

                        rollout_batches.append(rollout_batch)
                        futures.append(microbatch_futures)
                        print(f"microbatch_futures: {microbatch_futures}", flush=True)

            # The batch_iterator may be a load-redistributing server, so batches may be jagged.
            # We gather everything so that we can rebalance it.
            unbalanced_local_batch = GPTRolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=policy_model.tokenizer.eos_id,
                rollout_batch_seq_length=self.rollout_batch_seq_length,
            )
            log_memory("rollout_generator.py generate_rollouts PTRolloutBatch.from_rollout_batches")

            global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()
            log_memory("rollout_generator.py generate_rollouts PTRolloutBatch.from_rollout_batches")

        padded_rollout_sequence_length = global_rollout_batch["response_tokens"].size(-1)

        balanced_local_batch = global_rollout_batch.chunk(
            rank=parallel_state.get_training_data_parallel_rank(),
            split_size=parallel_state.get_training_data_parallel_world_size(),
        )

        # logprob calculation
        # since we compute the logprobs in nemo we need to be outside the generation resharding context
        # we also do the logprob and init_logprob calculations here to overlap with async environment compute
        batched_response_tokens = balanced_local_batch["response_tokens"]

        policy_model.inference_backend.free()
# policy_model.inference_backend.shutdown()

        with self.timer("logprobs"):
            rollout_logprobs = policy_model.get_inference_log_probs(batched_response_tokens)
            balanced_local_batch["logprobs"] = rollout_logprobs
            log_memory("rollout_generator.py policy_model.get_inference_log_probs")

        # compute init logprobs only if not in validation
        if not is_validation:
            with self.timer("init_logprobs"):
                rollout_init_logprobs = policy_model.get_init_policy_logprobs(batched_response_tokens)
                balanced_local_batch["init_logprobs"] = rollout_init_logprobs
        global_rollout_batch = balanced_local_batch.gather_and_balance_globally()
        log_memory("rollout_generator.py balanced_local_batch.gather_and_balance_globally")

        # getting environment results
        # we send environment step requests in the sharded context, so we need to keep this sharding and then undo it
        with generation_reshard_context():
            with self.timer("environment_wait"):
                env_rollout_batches = []
                for batch_futures in futures:
                    all_task_indices, all_task_results = [], []
                    for task, batch_indices, task_future in batch_futures:
                        all_task_indices.append(batch_indices)
                        _, _, rewards, episode_complete = self.tasks_to_environments[task].finish_step(task_future)
                        # not touching episode_complete for now since this loop only supports single-turn
                        all_task_results.append({"rewards": rewards})
                    print(f"all_task_indices: {all_task_indices}, all_task_results: {all_task_results}", flush=True)
                    batch_rewards = reconstruct_split_batch(all_task_indices, all_task_results)
                    print("### TASK INDX", all_task_indices)
                    print("### TASK RESULTS", all_task_results)
                    env_rollout_batches.append(batch_rewards)

            unbalanced_env_batch = GPTRolloutBatch.from_rollout_batches(
                env_rollout_batches,
                eos_id=policy_model.tokenizer.eos_id,
                rollout_batch_seq_length=padded_rollout_sequence_length,
            )
            global_env_batch = unbalanced_env_batch.gather_and_balance_globally()

            global_rollout_batch.update(global_env_batch)
            with self.timer("env_postproc_and_metrics"):
                global_rollout_batch, metrics = self.post_process_and_compute_rollout_metrics(global_rollout_batch)

        # saving generations
        with self.timer("generation_save"):
            rank = torch.distributed.get_rank()
            savefile = f"train_{rank}.jsonl" if not is_validation else f"validation_{rank}.jsonl"
            self.generation_log(global_rollout_batch, os.path.join(self.generation_save_dir, savefile))
        log_memory("rollout_generator.py generate_rollouts DONE")
    
        return global_rollout_batch, cpu_dict(metrics), self.timer.consume_durations()
    
    def post_process_and_compute_rollout_metrics(self, global_rollout_batch):
        # iterate over tasks and call the environments to get metrics and finalized batches
        split_idxs, split_batches, metrics = [], [], {}
        for task in sorted(self.tasks_to_environments.keys()):
            indices = []
            for idx, t in enumerate(global_rollout_batch["task_name"]):
                if t == task:
                    indices.append(idx)
            if len(indices) > 0:
                task_batch = batch_index_select(global_rollout_batch, indices)
                task_batch, task_metrics = \
                    self.tasks_to_environments[task].global_post_process_and_metrics(task_batch)
                split_idxs.append(indices)
                split_batches.append(task_batch)
                metrics[task] = task_metrics#add_prefix(task_metrics, task)
        
        # recompose batches
        recomposed_batch = reconstruct_split_batch(split_idxs, split_batches)
        return recomposed_batch, {**metrics, **self.compute_overall_metrics(global_rollout_batch)}

    def compute_overall_metrics(self, rollout_batch):
        def dict_get(batch, idx):
            inst = {}
            for k in batch.keys():
                inst[k] = batch[k][idx]
            return inst

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        rewards = rollout_batch["rewards"]
        is_end = rollout_batch["is_end"]
        sum_trt_error = 0
        num_toks = 0
        for i in range(prompt_lengths.size(0)):
            instance = dict_get(rollout_batch, i)
            response_length = instance["response_lengths"].item()
            prompt_length = instance["prompt_lengths"].item()
            logprobs = instance["logprobs"][prompt_length-1:response_length-1]
            trt_lps = instance["response_trt_lps"][prompt_length:response_length]
            error = torch.sum(torch.exp(torch.abs(logprobs - trt_lps)))
            sum_trt_error += error.item()
            num_toks += response_length - prompt_length

        metrics = {
            "rollout_size": prompt_lengths.size(0),
            "prompt_lengths": prompt_lengths.float().mean().item(),
            "generation_length": (response_lengths - prompt_lengths).float().mean().item(),
            "rewards": rewards.mean().item(),
            "fraction_of_samples_properly_ended": is_end.float().mean().item(),
            "trt_logprob_error": sum_trt_error / num_toks,
        }

        return metrics

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

        batch_size = rollout_batch["text"].shape[0]
        per_rank_batch = batch_size // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        end_idx = (rank + 1) * per_rank_batch if rank < torch.distributed.get_world_size() - 1 else batch_size
        records = []
        for idx in range(rank * per_rank_batch, end_idx):
            # get individual element of batch
            instance = dict_get(rollout_batch, idx)
            response_length = instance["response_lengths"].item()
            prompt_length = instance["prompt_lengths"].item()
            prompt_str = instance["prompt_sentences"]
            prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
            lps = instance["logprobs"][prompt_length-1:response_length-1].cpu().tolist()
            if "init_logprobs" in instance.keys():
                init_lps = instance["init_logprobs"][prompt_length-1:response_length-1].cpu().tolist()
            else:
                init_lps = None
            generation_tokens = instance["response_tokens"][prompt_length:response_length].cpu().tolist()

            record = {
                "prompt_hash": prompt_hash,
                "idx_in_batch": idx,
                "prompt": instance["prompt_sentences"],
                "response": instance["response_sentences"],
                "response_length": response_length,
                "extra_verifier_info": instance.get("extra_verifier_info", None),
                "rank": instance["generator_rank"].item(),
                "ended_correctly": instance["is_end"].item(),
                "reward": instance["rewards"].item(),
                "logprobs": lps,
                "trt_lps": instance["response_trt_lps"][prompt_length:response_length].cpu().tolist(),
                "init_logprobs": init_lps,
                "generation_tokens": generation_tokens,
            }
            records.append(record)

        with open(savefile, "a", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

