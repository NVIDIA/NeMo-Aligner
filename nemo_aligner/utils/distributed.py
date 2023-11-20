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

import warnings
from collections import defaultdict

import torch
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


def broadcast_2d_tensor_within_mp(tensor):
    """helper function to broadcast within the model parallel group
    """
    group = parallel_state.get_model_parallel_group()

    if torch.distributed.get_world_size(group) > 1:
        return broadcast_2d_tensor(tensor, get_model_parallel_src_rank(), group)

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
