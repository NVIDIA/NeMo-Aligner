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
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from nemo.collections.nlp.parts import utils_funcs
from nemo.utils.timers import NamedTimer
from nemo.collections.nlp.modules.common.megatron.utils import (
    get_iterator_k_split,
)
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


def broadcast_tensor(tensor, src, group, dtype=torch.float32, ndim=2):
    """Broadcast any 2d tensor from the src rank to every other rank in the given group.
    All the ranks that send or receive data must call this function."""
    if torch.distributed.get_rank() == src:
        assert tensor.ndim == ndim, f"tensor dims is not {ndim} but is {tensor.ndim} with shape {tensor.shape}"
        tensor = tensor.cuda().to(dtype)

        input_info = [tensor.size(i) for i in range(ndim)]
        input_info_tensor = torch.cuda.FloatTensor(input_info)

        torch.distributed.broadcast(input_info_tensor, src, group)
        torch.distributed.broadcast(tensor, src, group)
    else:
        input_info_tensor = torch.empty(ndim, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(input_info_tensor, src, group)

        dims = [int(input_info_tensor[i].item()) for i in range(ndim)]
        tensor = torch.empty(*dims, dtype=dtype, device=torch.cuda.current_device())
        torch.distributed.broadcast(tensor, src, group)
    return tensor


def broadcast_2d_tensor(tensor, src, group, dtype=torch.float32):
    return broadcast_tensor(tensor, src, group, dtype=dtype, ndim=2)


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


def subtract_distributed_logits_with_max(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, return the tensor which is substracted by the maximum value along its last dimension.
    This allows for more stable computation in softmax.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )
    # Subtract the maximum value.
    return vocab_parallel_logits - logits_max


def compute_distributed_log_sum_exp_logits(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, return shape B x S which is the logsumexp of the logits along the 
    last dimension across all TPs.
    """
    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=False).float()

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
    )
    return sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


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


def from_parallel_logits_to_logprobs(vocab_parallel_logits, target, inference_only=False, higher_stability=False):
    """get log probs out of a B x S x V//TP tensor
        NOTE: this function shifts the target, which means you must give it the unmodified targets

    Returns a B x S-1 tensor
    """
    target = target.roll(shifts=-1, dims=-1)
    return DistributedLogprob.apply(vocab_parallel_logits, target, inference_only, higher_stability)[
        :, :-1
    ].contiguous()


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


@torch.no_grad()
def compute_topk_logits_in_batched_sequence(
    model: MCoreGPTModel, tokens: torch.Tensor, position_ids: torch.Tensor, 
    attention_mask: torch.Tensor, top_k: int, precision: str, forward_micro_batch_size: int = 1):
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
            
            return {"topk_logits": topk_logits, "topk_token_ids": topk_token_ids, "log_sum_exp_logits": log_sum_exp_logits}
        return output_tensor, fake_loss_func

    seq_length = tokens.shape[1]
    assert tokens.shape[0] % forward_micro_batch_size == 0, f"batch size {tokens.shape[0]} / forward_micro_batch_size {forward_micro_batch_size} is not divisible."
    num_microbatches = int(tokens.shape[0] // forward_micro_batch_size)
    data_iter = get_iterator_k_split(
        {"tokens":  tokens, "position_ids": position_ids, "attention_mask": attention_mask},
        num_microbatches)
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
            ndim=3
        )
        topk_token_ids = broadcast_tensor(
            topk_token_ids,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            ndim=3
        )
        log_sum_exp_logits = broadcast_tensor(
            log_sum_exp_logits,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            ndim=2
        )

    return topk_logits, topk_token_ids, log_sum_exp_logits


# class _TopKLogitsCrossEntropy(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, vocab_parallel_logits, target_logits, target_token_ids, target_log_sum_exp_logits):
#         # vocab_parallel_logits: logits
#         # target_logits: Tensor of [B, seq_len, K]. Logits values of the target tokens.
#         # target_token_ids: Tensor of [B, seq_len, K]. Token ids of the target tokens.
#         # target_log_sum_exp_logits: Union[None, Tensor[B, seq_len], 1]. If not None, logsumexp of target logits over the whole vocab.
        
#         # Maximum value along vocab dimension across all GPUs.
#         logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
#         torch.distributed.all_reduce(
#             logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
#         )
#         # Subtract the maximum value.
#         vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        
#         # Get the partition's vocab indecies
#         get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
#         partition_vocab_size = vocab_parallel_logits.size()[-1]
#         rank = get_tensor_model_parallel_rank()
#         world_size = get_tensor_model_parallel_world_size()
#         vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

#         # Create a mask of valid vocab ids (1 means it needs to be masked).
#         target_mask = (target_token_ids < vocab_start_index) | (target_token_ids >= vocab_end_index)
#         masked_target_token_ids = target_token_ids.clone() - vocab_start_index
#         masked_target_token_ids[target_mask] = 0

#         # Get predicted-logits = logits[target]. Shape is [B, seq_len, K]. 
#         # Some of them (based on target_mask) needs to be masked.
#         predicted_logits = torch.gather(vocab_parallel_logits, dim=-1, index=masked_target_token_ids)
        
#         # When target_log_sum_exp_logits is None. The objective is 
#         # target_prob_k = exp(target_logits_k) / sum_{k=1}^K exp(target_logits_k)
#         # prob_k = exp(logits_k) / sum_{k=1}^K exp(logits_k)
#         # neg_loss = sum_{k=1}^{K} target_prob_k * log prob_{k} 
#         #          = sum_{k=1}^{K} target_prob_k * logits_k - sum_{k=1}^{K} target_prob_k * log sum_{k=1}^K exp(logits_k)
#         #          = sum_{k=1}^{K} target_prob_k * logits_k - log sum_{k=1}^K exp(logits_k)
#         # neg_loss will be Tensor of shape [B, seq_len]
#         if target_log_sum_exp_logits is None:
#             target_probs = target_logits.exp()
#             target_probs = target_probs / target_probs.sum(-1, keepdims=True)
#             neg_loss = (target_probs * predicted_logits).sum(-1, keepdims=False)
#             neg_loss[target_mask] = 0.0
            
            
#             # All reduce is needed to get the chunks from other GPUs.
#             torch.distributed.all_reduce(
#                 neg_loss, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
#             )
        
#             # compute the logsumexp of all K logits
#             exp_logits = predicted_logits.exp()
#             exp_logits[target_mask] = 0.0
#             sum_exp_logits = exp_logits.sum(dim=-1, keepdims=False)
#             torch.distributed.all_reduce(
#                 sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
#             )
#             log_sum_exp_logits = sum_exp_logits.log()
            
#             loss = - neg_loss + log_sum_exp_logits
        
#             # Store softmax, target-softmax, target-mask and masked-target for backward pass.
#             probs = exp_logits / sum_exp_logits.unsqueeze(dim=-1)
#             vocab_size = exp_logits.size(-1)
#             ctx.save_for_backward(
#                 probs, None, target_probs, None, target_mask, masked_target_token_ids, torch.LongTensor([vocab_size])
#             )
        
#         # When target_log_sum_exp_logits is not None. The objective is
#         # target_prob_k = exp(target_logits_k) / exp(target_log_sum_exp_logits), k=1,..., K
#         # target_prob_{K+1} = 1 - sum_{k=1}^K target_prob_k
#         # prob_k = exp(logits_k) / sum_{v=1}^V exp(logits_v), k=1,..., K
#         # prob_{K+1} = 1 - sum_{k=1}^K prob_k
#         # neg_loss = sum_{k=1}^{K+1} target_prob_k * log prob_{k}
#         # neg_loss will be Tensor of shape [B, seq_len]
#         else:
#             raise NotImplementedError
#             target_probs = (target_logits - target_log_sum_exp_logits).exp()
#             target_prob_K_add_1 = 1 - target_probs.sum(-1, keepdims=False)
            
#             # compute the logsumexp of the logits over the whole Vocab. Tensor of shape [B, seq_len, 1]
#             log_sum_exp_logits = compute_distributed_log_sum_exp_logits(vocab_parallel_logits).unsqueeze(-1)
#             predicted_logprobs = predicted_logits - log_sum_exp_logits
            
#             neg_loss = (target_probs * predicted_logprobs).sum(-1, keepdims=False)
#             neg_loss[target_mask] = 0.0
        
#             # All reduce is needed to get the chunks from other GPUs.
#             torch.distributed.all_reduce(
#                 neg_loss, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
#             )
        
#             # this should be computed on one TP only. We do this in rank=0
#             sum_predicted_probs = predicted_logprobs.exp().sum(dim=-1, keepdims=False)
#             torch.distributed.all_reduce(
#                 sum_predicted_probs, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
#             )
#             logprob_K_add_1 = (1 - sum_predicted_probs).log()
#             loss = - neg_loss - target_prob_K_add_1 * logprob_K_add_1
    
#             # Store softmax, target-mask and masked-target for backward pass.
#             probs = predicted_logprobs.exp()
#             prob_K_add_1 = logprob_K_add_1.exp()
#             vocab_size = exp_logits.size(-1)
#             ctx.save_for_backward(
#                 probs, prob_K_add_1, target_probs, target_prob_K_add_1,
#                 target_mask, masked_target_token_ids, torch.LongTensor([vocab_size])
#             )

#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):

#         # Retreive tensors from the forward path.
#         (probs, prob_K_add_1, target_probs, target_prob_K_add_1, target_mask, 
#          masked_target_token_ids, vocab_size) = ctx.saved_tensors

#         probs = probs * (2 * probs - 1)

#         vocab_size = vocab_size.item()

#         # All the inputs have softmax as thier gradient.
#         grad_input = probs
#         # For simplicity, work with the 2D gradient.
#         partition_vocab_size = probs.size()[-1]
#         grad_2d = grad_input.view(-1, partition_vocab_size)

#         # Add the gradient from matching classes.
#         arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

#         softmax_update = 1.0 - target_mask.view(-1).float()
  
#         grad_2d[arange_1d, masked_target_token_ids.view(-1)] -= softmax_update

#         # Finally elementwise multiplication with the output gradients.
#         grad_input.mul_(grad_output.unsqueeze(dim=-1))

#         return grad_input, None, None, None
    

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
