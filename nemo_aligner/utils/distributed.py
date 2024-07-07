# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""distributed utils for communicating between different ranks"""

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel

from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from nemo.utils.timers import NamedTimer
from nemo_aligner.utils.ppo_utils import calculate_entropy


def gather_tensor(tensor, dst, group, dtype=torch.float32):
    """Gather any 2d tensor to the dst rank from every other rank in the given group.
    All the ranks that send or receive data must call this function."""
    tensor = tensor.to(device=torch.cuda.current_device(), dtype=dtype)
    if torch.distributed.get_rank() == dst:
        gather_list = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    else:
        gather_list = None

    torch.distributed.gather(tensor, gather_list=gather_list, dst=dst, group=group)
    return gather_list


def broadcast_2d_tensor(tensor, src, group, dtype=torch.float32):
    """Broadcast any 2d tensor from the src rank to every other rank in the given group.
    All the ranks that send or receive data must call this function."""
    if torch.distributed.get_rank() == src:
        assert tensor.ndim == 2, f"tensor dims is not 2 but is {tensor.ndim} with shape {tensor.shape}"
        tensor = tensor.cuda().to(dtype)

        input_info = [
            tensor.size(0),
            tensor.size(1),
        ]
        input_info_tensor = torch.cuda.FloatTensor(input_info)

        torch.distributed.broadcast(input_info_tensor, src, group)
        torch.distributed.broadcast(tensor, src, group)
    else:
        input_info_tensor = torch.empty(2, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(input_info_tensor, src, group)

        dim1 = int(input_info_tensor[0].item())
        dim2 = int(input_info_tensor[1].item())

        tensor = torch.empty(dim1, dim2, dtype=dtype, device=torch.cuda.current_device())
        torch.distributed.broadcast(tensor, src, group)
    return tensor


def broadcast_2d_tensor_within_mp(tensor, dtype=torch.float32):
    """helper function to broadcast within the model parallel group
    """
    group = parallel_state.get_model_parallel_group()

    if torch.distributed.get_world_size(group) > 1:
        return broadcast_2d_tensor(tensor, get_model_parallel_src_rank(), group, dtype=dtype)

    return tensor


def broadcast_2d_tensor_within_pp(tensor, dtype=torch.float32):
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        return broadcast_2d_tensor(
            tensor,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            dtype=dtype,
        )

    return tensor


def run_if_model_parallel_src(fn, *fn_args, **fn_kwargs):
    """This function is meant to wrap an arbitary function to only call the function
    if it's the model parallel src. So if we have DP=2, this function will be called
    only twice."""
    src_rank = get_model_parallel_src_rank()

    output = None
    if torch.distributed.get_rank() == src_rank:
        output = fn(*fn_args, **fn_kwargs)

    return output


def normalize_tensor(tensor, mask, group=None):
    """normalizes a tensor using global mean and std
    """
    tensor = tensor.to(device=torch.cuda.current_device())
    mask = mask.to(device=torch.cuda.current_device())

    tensor_global_mean, tensor_global_var = masked_global_mean_var(tensor, mask, group=group)
    tensor = (tensor - tensor_global_mean) * torch.rsqrt(tensor_global_var + 1e-8)
    return tensor


def masked_global_mean_var(values, mask, group=None):
    """computes the global mean and var when there is a mask

    NOTE: the variance here is uncorrected

    mask and values must have same shape, with mask being {0,1} with 1 being the values we want to keep
    """
    assert values.shape == mask.shape, (values.shape, mask.shape)
    values = values.to(device=torch.cuda.current_device())
    mask = mask.to(device=torch.cuda.current_device())

    values = values * mask

    # Get global sum and count and calculate the global mean and variance
    sum_and_count = torch.tensor([values.sum(), mask.sum()], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(sum_and_count, group=group)
    global_sum, global_count = sum_and_count
    global_mean = global_sum / global_count
    variance_summed = (
        (((values - global_mean) ** 2) * mask).sum().to(device=torch.cuda.current_device(), dtype=torch.float32)
    )

    torch.distributed.all_reduce(variance_summed, group=group)

    return global_mean, variance_summed / global_count


@torch.no_grad()
def _compute_distributed_softmax(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
        return shape B x S x V//TP but softmaxed across the V dimension
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    exp_logits = vocab_parallel_logits.exp_()

    sum_exp_logits = exp_logits.sum(-1)

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
    )
    # exp_logits becomes the softmax output
    exp_logits.div_(sum_exp_logits.unsqueeze(-1))

    return exp_logits


@torch.no_grad()
def _compute_distributed_log_softmax(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
        return shape B x S x V//TP but softmaxed across the V dimension. More stable than just computing softmax
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


def _compute_distributed_top_k_p(logits, k, p, rank, world_size):
    """Expects a size B x S x V//TP tensor, computes a distributed top_k and top_p - setting all other logits to -Inf.
        return shape B x S x V//TP where only global top-k-p values (across the V dimension) are not filtered (i.e. set to -Inf)
    """
    src_rank = get_model_parallel_src_rank()
    get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = logits.size(-1)
    vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

    local_topk_values, local_topk_indices = torch.topk(logits, k=k, dim=-1)  # [B,S,k]
    local_topk_indices += vocab_start_index
    # Prepare containers for the gathered top-k values and indices from all GPUs
    if rank == src_rank:
        gathered_values = [torch.zeros_like(local_topk_values) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(local_topk_indices) for _ in range(world_size)]
    else:
        gathered_values = None
        gathered_indices = None

    # Gather top-k values and indices from all GPUs
    torch.distributed.gather(local_topk_values, gathered_values, dst=src_rank)
    torch.distributed.gather(local_topk_indices, gathered_indices, dst=src_rank)

    if rank == src_rank:  # only rank 0 will do the computation and scatter the outcome
        # Concatenate the gathered values and indices along a new dimension
        all_values = torch.cat(gathered_values, dim=-1)  # [B,S,world_size*k]
        all_indices = torch.cat(gathered_indices, dim=-1)

        # Perform a global top-k operation to find the global top-k values and indices
        global_topk_values, topk_indices = torch.topk(all_values, k=k, dim=-1)  # [B,S,k]
        global_topk_indices = torch.gather(all_indices, -1, topk_indices)

        # perform top_p
        if 0.0 < p < 1.0:
            # perform top_p and save in global_top_k_p_indices and spread to all ranks
            sorted_logits, sorted_indices = torch.sort(global_topk_values, descending=True, dim=-1)  # [B,S,k] for each
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # [B,S,k]
            global_top_k_p_indices = torch.gather(global_topk_indices, -1, sorted_indices)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove = sorted_indices_to_remove.roll(shifts=1, dims=-1)
            sorted_indices_to_remove[..., 0] = False
            global_top_k_p_indices = torch.where(
                sorted_indices_to_remove, torch.tensor(-1, dtype=torch.long), global_top_k_p_indices
            )  # the top_p are kept, the rest are set to -1 (so will be filtered in the following lines)
    else:
        global_top_k_p_indices = torch.empty_like(local_topk_indices)  # [B,S,k]

    torch.distributed.broadcast(global_top_k_p_indices, src=src_rank)

    # generate a mask according to the rank
    # filter indices within the current rank's segment
    mask_top_k_p_indices = (global_top_k_p_indices >= vocab_start_index) & (
        global_top_k_p_indices < vocab_end_index
    )  # [B,S,k] where only indices that are within the scope of current rank are True

    # adjust indices to local index space
    local_top_k_p_indices = global_top_k_p_indices
    local_top_k_p_indices -= vocab_start_index
    local_top_k_p_indices = torch.where(
        mask_top_k_p_indices, local_top_k_p_indices, torch.tensor(-1, dtype=torch.long)
    )  # [B,S,k] - the global top_k_p indices are localized to the rank, or -1 if they are not in this rank's segment

    valid_logits = torch.zeros_like(logits, dtype=torch.bool)
    batch_indices, sequence_indices = torch.where(
        mask_top_k_p_indices.any(dim=-1)
    )  # collect all b,s indices where there is a valid index in [b,s,:]
    local_vocab_indices = local_top_k_p_indices[
        batch_indices, sequence_indices
    ]  # collect the v indices per each [b,s]. should be up to k valid indices (not valid is -1)
    valid_local_indx_mask = local_vocab_indices != -1
    valid_local_batch_idx = batch_indices.unsqueeze(1).expand_as(valid_local_indx_mask)[valid_local_indx_mask]
    valid_local_sequence_idx = sequence_indices.unsqueeze(1).expand_as(valid_local_indx_mask)[valid_local_indx_mask]
    valid_local_vocab_idx = local_vocab_indices[valid_local_indx_mask]
    valid_logits[valid_local_batch_idx, valid_local_sequence_idx, valid_local_vocab_idx] = True
    logits[~valid_logits] = -torch.inf
    # return updated_logits
    return logits


def _distributed_apply_sampling_params(logits, context_lengths, sampling_params, rank, world_size):
    # apply the sampling params to the logits - focusing only on the generated tokens.
    if sampling_params.get("use_greedy", False):
        return logits
    if sampling_params.get("repetition_penalty", 1.0) != 1.0:
        raise NotImplementedError("not supporting repetition penalty when applying sampling params to logprobs")

    context_length = context_lengths.min().item()
    resp_logits = logits[:, context_length - 1 :]
    # divide by temp
    if sampling_params["temperature"] != 1.0:
        resp_logits /= sampling_params["temperature"]
    top_k = sampling_params["top_k"]
    top_p = sampling_params["top_p"]
    if top_k > 0:
        # Note : currently assuming that top_p is applied only if top_k>0.
        resp_logits = _compute_distributed_top_k_p(resp_logits, top_k, top_p, rank, world_size)
    elif 0.0 < top_p < 1.0:
        raise NotImplementedError(
            "Currently not supporting 0 < top_p < 1 with top_k=0 when applying sampling params to log probs"
        )

    return logits


def _distributed_apply_sampling_params(logits, context_lengths, sampling_params, rank, world_size):
    # apply the sampling params to the logits - focusing only on the generated tokens.
    if sampling_params.get("use_greedy", False):
        return logits

    context_length = context_lengths.min().item()
    resp_logits = logits[:, context_length - 1 :]
    # divide by temp
    if sampling_params["temperature"] != 1.0:
        resp_logits /= sampling_params["temperature"]
    top_k = sampling_params["top_k"]
    top_p = sampling_params["top_p"]
    if top_k > 0:
        # Note : currently assuming that top_p is applied only if top_k>0.
        resp_logits = _compute_distributed_top_k_p(resp_logits, top_k, top_p, rank, world_size)

    return logits


class DistributedLogprob(torch.autograd.Function):
    """Function to get logprobs out and differentiate through it
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        target,
        inference_only=False,
        higher_stability=False,
        sampling_params=None,
        prompt_lengths=None,
    ):
        get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        # if sampling_params should be applied, apply them to the vocab_parallel_logits
        if sampling_params is not None:
            if prompt_lengths is None:
                raise ValueError("prompt_lengths must be provided to apply sampling params to log ptobs")
            vocab_parallel_logits = _distributed_apply_sampling_params(
                vocab_parallel_logits, prompt_lengths, sampling_params, rank, world_size
            )

        # higher stability uses a more numerically stable distributed log_softmax instead of softmax
        # however, it uses more VRAM because there is an unavoidable exp() OP on the entire logits tensor
        # some models (like DPO) will get -inf in the resulting logprobs unless you set higher_stability=True
        if higher_stability:
            log_softmax_output = _compute_distributed_log_softmax(vocab_parallel_logits)
            log_probs = log_softmax_output.clone()
            softmax_output = log_softmax_output.exp_()
        else:
            softmax_output = _compute_distributed_softmax(vocab_parallel_logits)
            # if we only do inference, then do the log in place
            log_probs = softmax_output.log_() if inference_only else softmax_output.log()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # 1 if it's the chosen log prob, 0 otherwise
        is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target, num_classes=partition_vocab_size
        )

        grad_input = is_chosen.float().sub_(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None


def calculate_distributed_entropy(vocab_parallel_logits, mask=None):
    # TODO(geshen): this is memory intensive
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(vocab_parallel_logits)
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)[:, :-1, :].contiguous()

    return calculate_entropy(full_log_probs, mask)


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits,
    target,
    inference_only=False,
    higher_stability=False,
    sampling_params=None,
    prompt_lengths=None,
):
    """get log probs out of a B x S x V//TP tensor
        NOTE: this function shifts the target, which means you must give it the unmodified targets

    Returns a B x S-1 tensor
    """
    target = target.roll(shifts=-1, dims=-1)
    return DistributedLogprob.apply(
        vocab_parallel_logits, target, inference_only, higher_stability, sampling_params, prompt_lengths
    )[:, :-1].contiguous()


def pad_tensors_to_max_global_seq_len(list_of_tensors, pad_value, group, sequence_length_to_pad_to=None):
    """pad a list of tensors to the global sequence length across the specified group
    """
    # compute the local padding
    tensors_padded = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

    # find global max seq length
    max_seq_length = torch.tensor([tensors_padded.size(-1)], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(max_seq_length, op=torch.distributed.ReduceOp.MAX, group=group)
    max_seq_length = int(max_seq_length)

    if sequence_length_to_pad_to is not None:
        if max_seq_length > sequence_length_to_pad_to:
            warnings.warn(
                f"{max_seq_length=} is bigger than the provided {sequence_length_to_pad_to=}, overwriting the padding"
                f" to {max_seq_length}"
            )
        # pad to sequence length or max seq length, whichever is bigger
        max_seq_length = max(sequence_length_to_pad_to, max_seq_length)

    return torch.nn.functional.pad(tensors_padded, (0, max_seq_length - tensors_padded.size(-1)), value=pad_value)


class SyncTimer(NamedTimer):
    """Wrapper around NamedTimer to sync across DP ranks
        for more precise timing
    """

    def __init__(self, *args, **kwargs):
        self.reduce_op = kwargs.pop("reduce_op", torch.distributed.ReduceOp.MAX)
        super().__init__(*args, **kwargs)
        self.stored_results = defaultdict(list)

    def sync_time(self, list_to_sync):
        output_tensor = torch.tensor(list_to_sync, dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(output_tensor, op=self.reduce_op, group=parallel_state.get_data_parallel_group())

        return output_tensor

    def get_synced(self, *args, **kwargs):
        # time unsynced
        output = self.get(*args, **kwargs)

        # sync the time
        return self.sync_time([output]).item()

    def store(self, name=""):
        """instead of immediately syncing the timing, we'll store it
            for a sync later on
        """
        # store the last recorded timing, rather than a reduction of it
        output = self.get(name=name)
        self.stored_results[name].append(output)

    def sync_and_consume_over_stored_time(self, name=""):
        """get the timings we stored, sync them and iterates over them
            this function consumes the timings (i.e remove them after iterating)
        """
        if name not in self.stored_results:
            return

        output_list = self.sync_time(self.stored_results[name]).tolist()
        yield from output_list

        del self.stored_results[name]


@dataclass
class Timer:
    """Timer to tell us when the time limit is reached
    """

    duration: Optional[str]

    def __post_init__(self):
        self._duration = float("inf")

        if self.duration is not None:
            days, hours, mins, seconds = map(int, self.duration.strip().split(":"))
            self._duration = timedelta(days=days, hours=hours, minutes=mins, seconds=seconds).total_seconds()

    def start_time(self):
        self._start_time = time.monotonic()

    def get_time_elapsed(self):
        return time.monotonic() - self._start_time

    def get_time_remaining(self):
        return self._duration - self.get_time_elapsed()

    def is_finished(self):
        time_left = self.get_time_remaining()

        is_finished = time_left <= 0
        is_finished_tensor = torch.tensor([is_finished], dtype=torch.bool, device="cuda")

        # only respect rank 0 timing
        torch.distributed.broadcast(is_finished_tensor, 0)
        return is_finished_tensor.item()
