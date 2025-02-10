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

import requests
import torch
import nemo_aligner.utils.parallel_state as parallel_state
from nemo_aligner.utils.distributed import broadcast_tensor 
import time

def print_group_ranks(group):
    """
    Gather and print the global ranks of all processes in the given group.
    
    Args:
        group: A torch.distributed process group.
        
    Returns:
        A list containing the global ranks that belong to the group.
    """
    # Create a tensor that holds the current global rank on each process.
    local_rank_tensor = torch.tensor([torch.distributed.get_rank()], device="cuda", dtype=torch.long)
    
    # Get the world size for this group.
    group_world_size = torch.distributed.get_world_size(group=group)
    
    # Prepare a list to gather the global ranks.
    gathered_tensors = [torch.empty_like(local_rank_tensor) for _ in range(group_world_size)]
    
    # All gather the ranks of each process within the group.
    torch.distributed.all_gather(gathered_tensors, local_rank_tensor, group=group)
    
    # Convert the gathered tensors to integers.
    group_ranks = [int(t.item()) for t in gathered_tensors]
    print(f"Group ranks: {group_ranks}", flush=True)
    return group_ranks

class VLLMClient:
    DEFAULT_PAD_ID = -42

    def __init__(self, cfg, tokenizer, checkpoint_path):
        self.base_url = f"http://{cfg.ip}:{cfg.port}"

        self.pad_id = VLLMClient.DEFAULT_PAD_ID
        self.eos_id = tokenizer.eos_id
        self.checkpoint_path = checkpoint_path
        self.cpu_mp_gloo_group = None
        
    def build_cpu_mp_gloo_group(self):
        if self.cpu_mp_gloo_group is not None:
            return

        # get ranks of all processes in the MP group
        world_size = torch.distributed.get_world_size()
        mp_world_size = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_pipeline_model_parallel_world_size()
        
        # get ranks of all processes in the current MP group
        local_mp_rank = torch.tensor([torch.distributed.get_rank()], device="cuda", dtype=torch.long)
        gathered_mp_ranks = [torch.empty_like(local_mp_rank) for _ in range(mp_world_size)]
        torch.distributed.all_gather(gathered_mp_ranks, local_mp_rank, group=parallel_state.get_model_parallel_group())

        # Gather all MP groups globally
        local_mp_ranks = torch.tensor(gathered_mp_ranks, device="cuda")
        all_mp_ranks = [torch.empty_like(local_mp_ranks) for _ in range(world_size // mp_world_size)]

        # All gather across data parallel groups to get all MP groups
        torch.distributed.all_gather(
            all_mp_ranks,
            local_mp_ranks,
            group=parallel_state.get_data_parallel_group()
        )
        
        # Convert gathered tensors to lists and deduplicate
        seen_groups = set()
        for mp_group in all_mp_ranks:
            group = tuple(mp_group.cpu().tolist())
            if group not in seen_groups:
                seen_groups.add(group)
        rank_groups = sorted(list(seen_groups))  # Deduplicate and sort final list
        
        # build the Gloo groups
        for rank_group in rank_groups:
            print(f"Rank {torch.distributed.get_rank()} Building Gloo group with ranks: {rank_group}",flush=True)
            group = torch.distributed.new_group(list(rank_group), backend="gloo")
            if int(torch.distributed.get_rank()) in list(rank_group):
                self.cpu_mp_gloo_group = group
                print(f"Rank {torch.distributed.get_rank()} local Gloo group built successfully",flush=True)

    def refit(self, model):
        """
        Start the remote vLLM inference server.
        """
        ret_val = None
        if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
            url = f"{self.base_url}/start"
            try:
                data = {
                    "checkpoint_path": self.checkpoint_path,
                    "tp": parallel_state.get_tensor_model_parallel_world_size(),
                    "tp_src_gpu_idx": torch.cuda.current_device(),
                }
                response = requests.post(url, json=data)
                response.raise_for_status()
                data = response.json()
                print(f"Start response: {data}")
                ret_val = data
            except requests.exceptions.RequestException as e:
                print(f"Error starting the server: {e}")
        torch.distributed.barrier(group=parallel_state.get_model_parallel_group())
        return ret_val

    def generate(self, batch_tokens: tuple[torch.Tensor, torch.Tensor], use_greedy: bool = False):
        """
        Perform generation on a batch of token lists.
        
        :param batch_tokens: List of lists of tokens (e.g., [[1,2,3], [4,5,6]])
        :return: A dictionary with generations and logprobs if successful.
        """
        self.build_cpu_mp_gloo_group() # will become a no-op if already built
        if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
            prompt_tokens, prompt_lengths = batch_tokens
            batch_input_ids = []
            for idx in range(prompt_tokens.shape[0]):
                batch_input_ids.append(prompt_tokens[idx][0 : prompt_lengths[idx]].cpu().tolist())

            url = f"{self.base_url}/generate"
            # retry sending requests until it works
            response_success = False
            retry_ctr=0
            while not response_success:
                try:
                    response = requests.post(url, json=batch_input_ids)
                    response.raise_for_status()
                    data = response.json()
                    response_success = True
                except requests.exceptions.RequestException as e:
                    print(f"Error during generation: {e}, retrying {retry_ctr}")
                    retry_ctr += 1

            # prefix add the prompt tokens to the response tokens
            response_tokens = data["response_tokens"]
            response_tokens = [prompt_tokens[idx][0:prompt_lengths[idx]].cpu().tolist() + response for idx, response in enumerate(response_tokens)]
            response_lengths = [len(x) for x in response_tokens]
            # prefix add fake prompt logprobs to the response logprobs
            response_logprobs = data["response_logprobs"]
            response_logprobs = [[0] * prompt_lengths[idx] + response for idx, response in enumerate(response_logprobs)]
            
            # convert to tensor and pad
            output_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x).long() for x in response_tokens], batch_first=True, padding_value=self.eos_id)
            response_logprobs = torch.nn.utils.rnn.pad_sequence([torch.tensor(x).float() for x in response_logprobs], batch_first=True, padding_value=0.0)
            output_ids[output_ids == self.pad_id] = self.eos_id
            tensors = {
                "response_tokens": output_ids,
                "response_lengths": torch.tensor(response_lengths, device="cuda").long(),
                "response_logprobs_trt": response_logprobs,
            }
        else:
            tensors = {
                "response_tokens": None,
                "response_lengths": None, 
                "response_logprobs_trt": None
            }

        src_rank = parallel_state.get_model_parallel_src_rank()
        mp_group = parallel_state.get_model_parallel_group()
        
        # torch.distributed.barrier(group=parallel_state.get_model_parallel_group()) # wait for src process to get generation results
        assert self.cpu_mp_gloo_group is not None # cpu gloo group must be built before calling this function
        torch.distributed.barrier(group=self.cpu_mp_gloo_group)
        print(f"Tensors: {tensors}", flush=True)
        for k in sorted(tensors.keys()):
            print(k, flush=True)
            print(f"Broadcasting {k} rank {torch.distributed.get_rank()} src_rank {src_rank}")
            tensors[k] = broadcast_tensor(tensors[k], src_rank, mp_group)
            torch.distributed.barrier(group=parallel_state.get_model_parallel_group())
        print(f"Inference response: {tensors}")
        return tensors

    def free(self):
        """Shutdown the vLLM inference server."""
        data = None
        if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
            url = f"{self.base_url}/shutdown"
            try:
                response = requests.post(url)
                response.raise_for_status()
                data = response.json()
                print(f"Shutdown response: {data}")
            except requests.exceptions.RequestException as e:
                print(f"Error shutting down the server: {e}")
        torch.distributed.barrier(group=parallel_state.get_model_parallel_group())
        return data
