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

import functools
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.parts import utils_funcs
from nemo.utils.timers import NamedTimer
from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.utils.ppo_utils import calculate_entropy
from nemo_aligner.utils.utils import deprecated_in_version


def rebalance_nd_tensor(tensor, group):
    """
    Takes tensors with variable leading sizes (at dim=0) and then stack them into a single tensor.
    
    NOTE: assumes all other (i.e., non-zero) dimensions are equal.
    """
    num_samples = torch.as_tensor(tensor.size(0), dtype=torch.int64, device=torch.cuda.current_device())
    batch_num_per_rank = torch.zeros(
        torch.distributed.get_world_size(group), dtype=torch.int64, device=torch.cuda.current_device()
    )
    torch.distributed.all_gather_into_tensor(batch_num_per_rank, num_samples, group=group)

    B = batch_num_per_rank.sum()
    other_dims = tensor.shape[1:]

    indices = batch_num_per_rank.cumsum(dim=0)
    output_tensor = torch.zeros(B, *other_dims, dtype=tensor.dtype, device=torch.cuda.current_device())

    # tensor_split is a view we can copy into
    output_tensor.tensor_split(indices[0:-1].cpu())[torch.distributed.get_rank(group=group)].copy_(tensor)
    torch.distributed.all_reduce(output_tensor, group=group)
    return output_tensor


@deprecated_in_version("0.7.0", "Please use broadcast_tensor(tensor, src, group, dtype)")
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


def broadcast_tensor(tensor: torch.Tensor | None, src, group, dtype: torch.dtype | None = None):
    """
    Broadcast a tensor from the source rank to every other rank in the given group.
    All the ranks that send or receive data must call this function.
    
    Parameters:
    - tensor: The tensor to be broadcasted (or None for non source ranks).
    - src: The rank of the source tensor.
    - group: The process group to use for the broadcast.
    - dtype: (Optional) The desired data type to cast the tensor before broadcasting.

    Returns:
    - The broadcasted tensor.
    """
    if torch.distributed.get_rank() == src:
        tensor = tensor.cuda()
        if dtype:
            tensor = tensor.to(dtype)

       # Convert dtype and shape to plain Python types.
        metadata = [str(tensor.dtype), list(tensor.size())]
        print(f"Broadcasting metadata: {metadata}")
        torch.distributed.broadcast_object_list(metadata, src, group)
        torch.distributed.broadcast(tensor, src, group)
    else:
        metadata = ["", []]
        print(f"Pre-broadcast metadata: {metadata}", flush=True)
        torch.distributed.broadcast_object_list(metadata, src, group)
        print(f"Received metadata: {metadata}", flush=True)
        dtype_str, input_shape = metadata
        dtype = getattr(torch, dtype_str.split('.')[-1])
        tensor = torch.empty(input_shape, dtype=dtype, device=torch.cuda.current_device())
        torch.distributed.broadcast(tensor, src, group)
    return tensor


def broadcast_2d_tensor_within_mp(tensor, dtype=torch.float32):
    """helper function to broadcast within the model parallel group
    """
    group = parallel_state.get_model_parallel_group()

    if torch.distributed.get_world_size(group) > 1:
        return broadcast_2d_tensor(tensor, parallel_state.get_model_parallel_src_rank(), group, dtype=dtype)

    return tensor


@deprecated_in_version("0.7.0", "Please use broadcast_tensor_within_pp(tensor, dtype)")
def broadcast_2d_tensor_within_pp(tensor, dtype=torch.float32, from_last: bool = True):
    """
    from_last: True=broadcast from the last PP rank and False=broadcast from first PP rank (default=True)
    """
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        return broadcast_2d_tensor(
            tensor,
            parallel_state.get_pipeline_model_parallel_last_rank()
            if from_last
            else parallel_state.get_pipeline_model_parallel_first_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            dtype=dtype,
        )

    return tensor


def broadcast_tensor_within_pp(tensor: torch.Tensor | None, dtype: torch.dtype = None, from_last: bool = True):
    """
    tensor: Should be a valid tensor on src rank and None elsewhere
    dtype: no dtype means that the dtype is inferred
    from_last: True=broadcast from the last PP rank and False=broadcast from first PP rank (default=True)
    """
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        return broadcast_tensor(
            tensor,
            parallel_state.get_pipeline_model_parallel_last_rank()
            if from_last
            else parallel_state.get_pipeline_model_parallel_first_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            dtype=dtype,
        )

    return tensor


def gather_tensor(tensor, dst, group, dtype=None):
    """Gather any tensor to the dst rank from every other rank in the given group.
    All the ranks that send or receive data must call this function."""
    tensor = tensor.to(device=torch.cuda.current_device(), dtype=dtype)
    if torch.distributed.get_rank() == dst:
        gather_list = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    else:
        gather_list = None

    torch.distributed.gather(tensor, gather_list=gather_list, dst=dst, group=group)
    return gather_list

def gather_jagged_object_lists(local_objects: list, group=None):
    """
    Gathers jagged lists of picklable objects from all ranks.
    WARNING: synchronous
    
    Args:
        local_objects: List of objects to gather from current rank
        group: Optional process group
        
    Returns:
        Flattened list of all objects from all ranks in order [rank0, rank1, ...]
    """
    # Gather all lists across ranks
    world_size = torch.distributed.get_world_size(group=group)
    gathered_lists = [None] * world_size
    torch.distributed.all_gather_object(gathered_lists, local_objects, group=group)
    
    # Flatten into single list while preserving order
    return [obj for sublist in gathered_lists for obj in sublist]

def run_if_model_parallel_src(fn, *fn_args, **fn_kwargs):
    """This function is meant to wrap an arbitary function to only call the function
    if it's the model parallel src. So if we have DP=2, this function will be called
    only twice."""
    src_rank = parallel_state.get_model_parallel_src_rank()

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


class DistributedLogprob(torch.autograd.Function):
    """Function to get logprobs out and differentiate through it
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, inference_only=False, higher_stability=False):
        get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

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
        return grad_input, None, None, None


def calculate_distributed_entropy(vocab_parallel_logits, mask=None):
    # TODO(geshen): this is memory intensive
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(vocab_parallel_logits)
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)[:, :-1, :].contiguous()

    return calculate_entropy(full_log_probs, mask)


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits, target, inference_only=False, higher_stability=False, ignore_last=True
):
    """get log probs out of a B x S x V//TP tensor
        NOTE: this function shifts the target, which means you must give it the unmodified targets

    Returns a B x S-1 tensor
    """

    if ignore_last:
        target = target.roll(shifts=-1, dims=-1)
    probs = DistributedLogprob.apply(vocab_parallel_logits, target, inference_only, higher_stability).contiguous()
    ### ignore_last should be true if labels are not shifted as a data preparation step
    if ignore_last:
        return probs[:, :-1]
    else:
        return probs


def all_reduce_dict(dictionary, dtype=torch.float32, group=None, op=torch.distributed.ReduceOp.SUM):
    keys = sorted(dictionary)
    tensor = torch.as_tensor([dictionary[k] for k in keys], dtype=dtype, device=torch.cuda.current_device())
    torch.distributed.all_reduce(tensor, op=op, group=group)
    return dict(zip(keys, tensor.tolist()))


@torch.no_grad()
def compute_topk_logits_in_batched_sequence(
    model: MCoreGPTModel,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int,
    precision: str,
    forward_micro_batch_size: int = 1,  ## TODO: make this match with model micro_batch_size
):
    """
    Compute the topk predictive logits of the model at each token in a batch of sequences.
    
    model: GPTModel in megatron_core.
    tokens: torch.Tensor.
    position_ids: torch.Tensor.
    attention_mask: torch.Tensor.
    top_k: Int. 
    precision: The precision of the model.
    forward_micro_batch_size: Int. The micro batch size in forwarding.
    """

    def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
        batch = next(dataloader_iter)

        tokens = batch["tokens"]

        # this is necessary if MBS > 1 with the new GBS padding logic, as you may get batch dim > 1 in some configs
        # these two lines ensure your position_ids and attn_mask are always B=1
        # position_ids = batch["position_ids"][0:1]
        attention_mask = batch["attention_mask"][0:1]
        position_ids = batch["position_ids"]

        output_tensor = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask)

        # in this nemo version the model and autocast dtypes are not synced
        # so we need to explicitly cast it
        if not parallel_state.is_pipeline_last_stage():
            output_tensor = output_tensor.to(dtype=utils_funcs.torch_dtype_from_precision(precision))

        def fake_loss_func(output_tensor, non_loss_data=True):
            # gather the output_tensor across tensor parallel ranks.
            # The resulting tensor is [batch_size, sequence_length, n_vocab_size]
            output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

            # subtract the logits with its maximum, to ensure stability when computing log probs.
            logits_max = torch.amax(output_tensor, dim=-1, keepdim=True)
            output_tensor = output_tensor - logits_max

            # compute the log sum exp logits
            log_sum_exp_logits = output_tensor.exp().sum(-1, keepdim=False).log().float()

            # compute the top_k logits and topk token_ids
            topk_logits, topk_token_ids = torch.topk(output_tensor, top_k)

            dp_world_size = parallel_state.get_data_parallel_world_size()
            group = parallel_state.get_data_parallel_group()
            rank = torch.distributed.get_rank(group=group)

            def gather_tensors(in_tensor):
                out_tensor = torch.zeros(
                    (in_tensor.shape[0] * dp_world_size, *in_tensor.shape[1:]),
                    dtype=in_tensor.dtype,
                    device=torch.cuda.current_device(),
                )
                torch.distributed.all_gather_into_tensor(out_tensor, in_tensor, group=group)
                return out_tensor

            topk_logits_gathered = gather_tensors(topk_logits)
            topk_token_ids_gathered = gather_tensors(topk_token_ids)
            log_sum_exp_logits_gathered = gather_tensors(log_sum_exp_logits)

            return {
                "topk_logits": topk_logits_gathered,
                "topk_token_ids": topk_token_ids_gathered,
                "log_sum_exp_logits": log_sum_exp_logits_gathered,
            }

        return output_tensor, fake_loss_func

    seq_length = tokens.shape[1]
    assert (
        tokens.shape[0] % forward_micro_batch_size == 0
    ), f"batch size {tokens.shape[0]} / forward_micro_batch_size {forward_micro_batch_size} is not divisible."
    num_microbatches = int(tokens.shape[0] / forward_micro_batch_size)
    data_iter = get_iterator_k_split(
        {"tokens": tokens, "position_ids": position_ids, "attention_mask": attention_mask}, num_microbatches
    )
    fwd_bwd_function = get_forward_backward_func()
    losses_reduced_per_micro_batch = fwd_bwd_function(
        forward_step_func=fwd_output_and_loss_func,
        data_iterator=data_iter,
        model=model,
        num_microbatches=num_microbatches,
        forward_only=True,
        seq_length=seq_length,
        micro_batch_size=forward_micro_batch_size,
        collect_non_loss_data=True,
    )

    if parallel_state.is_pipeline_last_stage():
        topk_logits = torch.cat([item["topk_logits"] for item in losses_reduced_per_micro_batch], dim=0)
        topk_token_ids = torch.cat([item["topk_token_ids"] for item in losses_reduced_per_micro_batch], dim=0)
        log_sum_exp_logits = torch.cat([item["log_sum_exp_logits"] for item in losses_reduced_per_micro_batch], dim=0)
    else:
        topk_logits = None
        topk_token_ids = None
        log_sum_exp_logits = None

    # broadcast it from last PP stage to everything else
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        topk_logits = broadcast_tensor(
            topk_logits,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
        )
        topk_token_ids = broadcast_tensor(
            topk_token_ids,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
        )
        log_sum_exp_logits = broadcast_tensor(
            log_sum_exp_logits,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
        )

    return topk_logits, topk_token_ids, log_sum_exp_logits


class _TopKLogitsCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        target_logits,
        target_token_ids,
        labels,
        kd_loss_weight=1.0,
        sft_loss_weight=0,
        bwd_kl=False,
        cross_tokenizer=False,
    ):

        # vocab_parallel_logits: logits
        # target_logits: Tensor of [B, seq_len, K]. Logits values of the target tokens.
        # target_token_ids: Tensor of [B, seq_len, K]. Token ids of the target tokens.
        # target_log_sum_exp_logits: Union[None, Tensor[B, seq_len], 1]. If not None, logsumexp of target logits over the whole vocab.
        # labels: Tensor of [B, seq_len]. True labels.
        # bwd_kl: Whether to use backward KL divergence instead of foward KL
        # cross_tokenizer: Whether to use the student's top-k logits from the loss calculation.
        #    rather than using the logits corresponding to the teacher's top-k logits.
        #    can be used when the teacher and student use different tokenizers.
        ## NOTE: "target" refers to the teacher top-k logits while "label" refers to the true label for the given example

        ## variables for SFT loss
        exp_logits, label_mask, masked_labels_1d, vocab_size = None, None, None, 0.0

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indices
        get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
        vocab_size = partition_vocab_size * world_size
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)

        K = target_logits.size()[-1]

        if cross_tokenizer:  ## naively grab the top-k logits from the student
            predicted_logits_full = tensor_parallel.gather_from_tensor_model_parallel_region(vocab_parallel_logits)
            ## shape [bs, sl, K], ids in [0, VS)
            predicted_logits_topk, predicted_topk_ids = torch.topk(predicted_logits_full, K)

            predicted_logits_topk = predicted_logits_topk.view_as(target_logits)  ## TODO: is this necessary?
            # Create a mask of valid vocab ids for this rank (1 means it needs to be masked).
            target_mask = (predicted_topk_ids < vocab_start_index) | (predicted_topk_ids >= vocab_end_index)
            predicted_topk_ids = predicted_topk_ids.clone() - vocab_start_index
            ids_for_loss = predicted_topk_ids.view(-1)

        else:
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            target_mask = (target_token_ids < vocab_start_index) | (target_token_ids >= vocab_end_index)
            masked_target_token_ids = target_token_ids.clone() - vocab_start_index
            masked_target_token_ids[target_mask] = 0

            ## Get the logits for the top-K teacher ids
            ## use repeat_interleave as we want to select K tokens per example
            arange_1d_topk = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device).repeat_interleave(
                K
            )

            target_token_ids_1d = masked_target_token_ids.view(-1)  ## B * seq_length * K
            predicted_logits_1d_topk = logits_2d[arange_1d_topk, target_token_ids_1d]
            predicted_logits_1d_topk = predicted_logits_1d_topk.clone().contiguous()
            predicted_logits_topk = predicted_logits_1d_topk.view_as(target_logits)
            predicted_logits_topk[target_mask] = 0.0
            # All reduce is needed to get the chunks from other GPUs.
            torch.distributed.all_reduce(
                predicted_logits_topk,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_tensor_model_parallel_group(),
            )

            ids_for_loss = target_token_ids_1d

        ## get target probabilities for top-K classes
        target_exp_logits = target_logits.exp()
        target_sum_exp_logits = target_exp_logits.sum(-1, keepdims=True)
        target_probs = target_exp_logits / target_sum_exp_logits
        del target_exp_logits
        log_sum_exp = target_sum_exp_logits.log()

        # compute the logsumexp of all K predicted logits
        exp_logits_topk = predicted_logits_topk.exp()
        sum_exp_logits_topk = exp_logits_topk.sum(dim=-1, keepdims=True)
        log_sum_exp_logits_topk = sum_exp_logits_topk.log()

        ## compute the predicted probabilitites
        probs = exp_logits_topk / sum_exp_logits_topk

        if bwd_kl:
            # The objective is
            # target_prob_k = exp(target_logits_k) / sum_{k=1}^K exp(target_logits_k)
            # prob_k = exp(logits_k) / sum_{k=1}^K exp(logits_k)
            # loss = sum_{k=1}^{K} prob_k * (log prob_{k}  - log target_prob_k)
            kd_loss = (probs * (predicted_logits_topk - log_sum_exp_logits_topk - target_logits + log_sum_exp)).sum(-1)

        else:  ## forward_kl
            # The objective is
            # target_prob_k = exp(target_logits_k) / sum_{k=1}^K exp(target_logits_k)
            # prob_k = exp(logits_k) / sum_{k=1}^K exp(logits_k)
            # neg_loss = sum_{k=1}^{K} target_prob_k * log prob_{k} - sum_{k=1}^{K} target_prob_k * log target_prob_k
            #          = sum_{k=1}^{K} target_prob_k * (logits_k - log sum_{k=1}^K exp(logits_k)) - const
            #          = (sum_{k=1}^{K} target_prob_k * logits_k) - log sum_{k=1}^K exp(logits_k) - const
            # neg_loss will be Tensor of shape [B, seq_len]
            neg_loss = (target_probs * predicted_logits_topk).sum(-1, keepdims=False)

            const = (target_probs * (target_logits - log_sum_exp)).sum(-1)
            kd_loss = -neg_loss + log_sum_exp_logits_topk.squeeze() + const

        sft_loss = torch.zeros_like(kd_loss)
        if sft_loss_weight > 0:
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            label_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            masked_labels = labels.clone() - vocab_start_index
            masked_labels[label_mask] = 0

            masked_labels_1d = masked_labels.view(-1)
            arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
            predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
            predicted_logits_1d = predicted_logits_1d.clone().contiguous()
            predicted_logits = predicted_logits_1d.view_as(labels)
            predicted_logits[label_mask] = 0.0
            # All reduce is needed to get the chunks from other GPUs.
            torch.distributed.all_reduce(
                predicted_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_tensor_model_parallel_group(),
            )

            # Sum of exponential of logits along vocab dimension across all GPUs.
            exp_logits = vocab_parallel_logits.exp()
            sum_exp_logits = exp_logits.sum(dim=-1)
            torch.distributed.all_reduce(
                sum_exp_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_tensor_model_parallel_group(),
            )

            # Loss = log(sum(exp(logits))) - predicted-logit.
            sft_loss = torch.log(sum_exp_logits) - predicted_logits

            # Store softmax, target-mask and masked-target for backward pass.
            exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        loss = kd_loss_weight * kd_loss + sft_loss_weight * sft_loss

        ctx.save_for_backward(
            probs,
            target_probs,
            target_mask,
            ids_for_loss,  ## either student or teacher top-k indices depending on whether cross_tokenizer is True
            torch.LongTensor([partition_vocab_size]),
            torch.Tensor([bwd_kl]),
            torch.Tensor([kd_loss_weight, sft_loss_weight]),
            exp_logits,
            label_mask,
            masked_labels_1d,
        )

        return loss, kd_loss, sft_loss

    @staticmethod
    def backward(ctx, grad_output, *_):

        # Retreive tensors from the forward path.
        (
            probs,
            target_probs,
            target_mask,
            target_token_ids_1d,
            partition_vocab_size,
            bwd_kl,
            loss_weights,
            full_softmax,
            label_mask,
            masked_labels_1d,
        ) = ctx.saved_tensors

        K = probs.size()[-1]

        grad_input = torch.zeros((*probs.size()[:-1], partition_vocab_size), device=probs.device)
        # For simplicity, work with the 2D gradient.
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Used access the the indices of the gradient that are nonzero
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device).repeat_interleave(K)

        inv_target_mask = ~target_mask

        bwd_kl = bwd_kl.item()
        ## these values are only needed when computing bwd KL
        if bwd_kl:
            log_prob_ratio = probs.log() - target_probs.log()  ## shape = (bs, sl)
            bwd_kl_const = (probs * log_prob_ratio).sum(-1, keepdims=True).repeat_interleave(K, -1)

            ## get only the values on this TP rank
            bwd_kl_const = bwd_kl_const[inv_target_mask]
            log_prob_ratio = log_prob_ratio[inv_target_mask]

        ## grab only the probs on this TP rank
        probs = probs[inv_target_mask]
        target_probs = target_probs[inv_target_mask]
        arange_1d = arange_1d[inv_target_mask.view(-1)]
        target_token_ids_1d = target_token_ids_1d[inv_target_mask.view(-1)]

        kd_loss_weight = loss_weights[0]
        sft_loss_weight = loss_weights[1]

        if bwd_kl:
            nonzero_grad = probs * (log_prob_ratio - bwd_kl_const)
            grad_2d[arange_1d, target_token_ids_1d] = nonzero_grad.view(-1)  ## slot in the nonzero gradients

        else:  ## forward KL
            grad_2d[arange_1d, target_token_ids_1d] = probs.view(-1)
            softmax_update = torch.zeros_like(grad_2d)
            softmax_update[arange_1d, target_token_ids_1d] = target_probs.view(-1)  ## slot in the nonzero gradients
            grad_2d -= softmax_update

        grad_2d *= kd_loss_weight

        if sft_loss_weight > 0:

            # All the inputs have softmax as thier gradient.
            grad_input_sft = full_softmax
            # For simplicity, work with the 2D gradient.
            grad_2d_sft = grad_input_sft.view(-1, partition_vocab_size)

            # Add the gradient from matching classes.
            arange_1d_label = torch.arange(start=0, end=grad_2d_sft.size()[0], device=grad_2d_sft.device)

            label_mask = label_mask.view(-1).float()
            label_mask -= 1.0
            grad_2d_sft[arange_1d_label, masked_labels_1d] += label_mask
            grad_2d_sft *= sft_loss_weight

            grad_2d += grad_2d_sft

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None, None, None, None, None


@deprecated_in_version("0.7.0", "Consider using ScopedTimer")
class SyncTimer(NamedTimer):
    """Wrapper around NamedTimer to sync across DP ranks
        for more precise timing
    """

    def __init__(self, *args, **kwargs):
        # TODO: double check can delete
        self.reduce_op = kwargs.pop("reduce_op", torch.distributed.ReduceOp.MAX)
        super().__init__(*args, **kwargs)
        # TODO: double check can delete
        self.stored_results = defaultdict(list)

    # TODO: double check can delete
    def sync_time(self, list_to_sync):
        output_tensor = torch.tensor(list_to_sync, dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(output_tensor, op=self.reduce_op, group=parallel_state.get_data_parallel_group())

        return output_tensor

    # TODO: double check can delete
    def get_synced(self, *args, **kwargs):
        # time unsynced
        output = self.get(*args, **kwargs)

        # sync the time
        return self.sync_time([output]).item()

    def stop_and_get_time(self, name=""):
        self.stop(name=name)
        return self.get(name=name)

    # TODO: double check can delete
    def store(self, name=""):
        """instead of immediately syncing the timing, we'll store it
            for a sync later on
        """
        # store the last recorded timing, rather than a reduction of it
        output = self.get(name=name)
        self.stored_results[name].append(output)

    # TODO: double check can delete
    def sync_and_consume_over_stored_time(self, name=""):
        """get the timings we stored, sync them and iterates over them
            this function consumes the timings (i.e remove them after iterating)
        """
        if name not in self.stored_results:
            return

        output_list = self.sync_time(self.stored_results[name]).tolist()
        yield from output_list

        del self.stored_results[name]


class ScopedTimer:
    """
    A thin adapter over the NamedTimer class to help time sections of code 
    using a context manager.

    This class is useful for tracking timings automatically so you don't need 
    to manually collect them. You only need to pass the timer around and can 
    collect the durations in one place, instead of returning and mutating 
    dictionaries throughout your code.

    The ScopedTimer ensures that durations are logged and consumed properly, 
    preventing accidental overwriting of previous measurements.

    Usage:
        timer = ScopedTimer()

        # All durations are logged in the timer
        with timer("step_time"):
            with timer("fwd"):
                model.fwd()
            with timer("bwd"):
                model.bwd()

        # Consume all durations and reset internal store
        durations = timer.consume_durations()

        # Durations that are not consumed will raise a ValueError
        with timer("fwd"):
            model.fwd()
        with timer("fwd"):
            model.fwd()  # <-- This will raise an error as timer.consume_durations()
                         # is not called, meaning the previous measurement is 
                         # still stored.

    Methods:
        consume_durations() -> dict[str, float]:
            Returns a dictionary of all logged durations and resets the internal log.

        __call__(name: str):
            Context manager for timing a section of code. Raises a ValueError if
            durations are not consumed before starting a new measurement for the 
            same name.

    Raises:
        ValueError: If attempting to start a new timing section for a name that
                    already has a recorded duration without consuming the previous
                    measurement using consume_durations().
    """

    def __init__(self, *args, **kwargs):
        self._timer = NamedTimer(*args, **kwargs)
        self._duration_log = {}

    def consume_durations(self) -> dict[str, float]:
        durations = self._duration_log
        self._duration_log = {}
        return durations

    @contextmanager
    def __call__(self, name: str):
        try:
            self._timer.start(name=name)
            yield
        finally:
            self._timer.stop(name=name)
            if name in self._duration_log:
                raise ValueError(
                    f"Attempted to store new duration for {name=} before consuming last measurement. Call consume_durations() to consume the last set of measurements."
                )
            self._duration_log[name] = self._timer.get(name=name)


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


def pad_list(tensor_list, pad_value):
    """
    Pad list of tensors to max seq len
    """
    max_N = max(tensor.size(1) for tensor in tensor_list)
    padded_tensors = [torch.nn.functional.pad(t, (0, max_N - t.size(1))) for t in tensor_list]

    return padded_tensors


def run_distributed_inference(inputs=None, infer_fn=None):
    tokens, lengths = None, None
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        tokens = torch.as_tensor(inputs["inputs"], dtype=torch.long, device=torch.cuda.current_device())
        lengths = torch.as_tensor(inputs["sequence_length"], dtype=torch.long, device=torch.cuda.current_device())

    tokens = broadcast_2d_tensor(tokens, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank]
    lengths = broadcast_2d_tensor(lengths, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank].squeeze(-1)

    outputs = infer_fn(inputs=(tokens, lengths))

    if isinstance(outputs, tuple):
        # rm and critic are combined in this case
        rewards, values = outputs
        rewards = rebalance_nd_tensor(rewards, group=parallel_state.get_data_parallel_group()).squeeze(1).cpu().numpy()
        values = rebalance_nd_tensor(values, group=parallel_state.get_data_parallel_group()).cpu().numpy()

        return rewards, values

    return rebalance_nd_tensor(outputs, group=parallel_state.get_data_parallel_group()).cpu().numpy()


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

def gather_and_sort_lists_of_lists(my_list_of_lists, group):
    # gather the list across all ranks using object gather
    gathered_lists = [None for _ in range(torch.distributed.get_world_size(group=group))]
    torch.distributed.all_gather_object(gathered_lists, my_list_of_lists, group=group)
    # Flatten the gathered lists (each from a different process) into a single list, then sort it.
    flattened_list = []
    for sublist in gathered_lists:
        flattened_list.extend(sublist)
    flattened_list.sort()
    return flattened_list