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

import megatron.core.parallel_state as mcore_parallel_state
import numpy as np
import pytest
import torch
import torch.distributed
from megatron.core import tensor_parallel

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    _TopKLogitsCrossEntropy,
    broadcast_tensor_within_pp,
    calculate_distributed_entropy,
    from_parallel_logits_to_logprobs,
    masked_global_mean_var,
)
from nemo_aligner.utils.ppo_utils import calculate_entropy

"""A file to test the core distributed function calls in RLHF"""


def slow_from_parallel_logits_to_logprobs(parallel_logits, tokens, ignore_last=True):
    """a slow but very safe way of computing logits -> logprobs. Uses a lot of memory but good for testing"""
    # Gather logits across all TP ranks for testing
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(parallel_logits)

    # Convert from logits to log-probs.
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    if ignore_last:
        full_log_probs = full_log_probs[:, :-1, :]
        indices = tokens[:, 1:].unsqueeze(-1)
    else:
        indices = tokens.unsqueeze(-1)
    full_log_probs = full_log_probs.contiguous()
    log_probs = torch.gather(input=full_log_probs, dim=2, index=indices).squeeze(dim=-1).contiguous()
    return log_probs


def calculate_entropy_full(logits):
    # Convert from logits to log-probs.
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    full_log_probs = full_log_probs[:, :-1, :].contiguous()
    return calculate_entropy(full_log_probs)


### naive reference implementation
def naive_top_k_loss_function(
    output_tensor,
    target_topk_logits,
    target_topk_token_ids,
    target_log_sum_exp_logits,
    loss_mask,
    labels,
    kd_loss_weight=1,
    sft_loss_weight=0,
    kd_loss="fwd_kl",
    cross_tokenizer=False,
):
    def loss_func(
        logits, target_logits, mask, kd_loss="fwd_kl", logits_scale=1.0, target_logits_scale=1.0,
    ):

        logprobs = torch.nn.functional.log_softmax(logits_scale * logits, dim=-1)
        target_logprobs = torch.nn.functional.log_softmax(target_logits_scale * target_logits, dim=-1)

        if kd_loss == "fwd_kl":
            loss = torch.sum(target_logprobs.exp() * (target_logprobs - logprobs), dim=-1)
        elif kd_loss == "bwd_kl":
            loss = torch.sum(logprobs.exp() * (logprobs - target_logprobs), dim=-1)
        else:
            raise ValueError(f"kd_loss {kd_loss} is not supported.")
        return torch.sum(loss * mask) / torch.sum(mask).clamp(min=1.0)

    output_tensor_max = torch.max(output_tensor, dim=-1)[0]
    torch.distributed.all_reduce(
        output_tensor_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )
    output_tensor = output_tensor - output_tensor_max.unsqueeze(dim=-1).detach()
    output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

    if cross_tokenizer:
        topk_logits, _ = torch.topk(output_tensor, target_topk_token_ids.shape[-1])
    else:
        # compute the knowledge distillation loss against the ground-truth logits
        topk_logits = torch.gather(output_tensor, dim=-1, index=target_topk_token_ids)

    target_topk_logits_in_loss = target_topk_logits

    kd_loss = loss_func(topk_logits, target_topk_logits_in_loss, mask=loss_mask, kd_loss=kd_loss)

    # compute the sft loss against the ground-truth labels
    sft_loss = torch.zeros_like(kd_loss)
    if sft_loss_weight != 0:
        target_label_logits = torch.gather(output_tensor, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        log_sum_exp_logits = torch.logsumexp(output_tensor, dim=-1)
        target_label_logprobs = target_label_logits - log_sum_exp_logits
        sft_loss = -torch.sum(target_label_logprobs * loss_mask) / torch.sum(loss_mask).clamp(min=1.0)

    loss = kd_loss_weight * kd_loss + sft_loss_weight * sft_loss

    return loss


@pytest.mark.run_only_on("GPU")
def test_distributed_masked_global_mean_var(init_model_parallel):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    device = torch.cuda.current_device()

    # global values and mask
    values = [
        torch.randn(4, 8, generator=torch.Generator(device).manual_seed(i), device=device)
        for i in range(torch.distributed.get_world_size())
    ]
    masks = [
        (torch.randn(4, 8, generator=torch.Generator(device).manual_seed(i + 1), device=device) > 0).float()
        for i in range(torch.distributed.get_world_size())
    ]

    values_catted = torch.cat(values)
    masks_catted = torch.cat(masks)

    global_var_pt, global_mean_pt = torch.var_mean(
        values_catted.flatten()[masks_catted.bool().flatten()], correction=0
    )

    rank = torch.distributed.get_rank()

    values = values[rank]
    mask = masks[rank]
    global_mean, global_var = masked_global_mean_var(values, mask, None)

    torch.testing.assert_close(
        global_mean_pt, global_mean, msg=f"expected global mean {global_mean_pt} but got {global_mean}"
    )
    torch.testing.assert_close(
        global_var_pt, global_var, msg=f"expected global var {global_var_pt} but we got {global_var}"
    )


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "batch_size,seed,dtype,atol,rtol,higher_stability,ignore_last",
    [
        (1, 9999, torch.float32, 1e-08, 1e-05, False, True),
        (4, 100, torch.float32, 1e-08, 1e-05, False, True),
        (8, 1234, torch.float32, 1e-08, 1e-05, False, True),
        (1, 9999, torch.float32, 1e-08, 1e-05, True, True),
        (4, 100, torch.float32, 1e-08, 1e-05, True, True),
        (8, 1234, torch.float32, 1e-08, 1e-05, True, True),
        (1, 746, torch.bfloat16, 0.005, 0.01, False, True),
        (4, 334, torch.bfloat16, 0.005, 0.01, False, True),
        (8, 123456, torch.bfloat16, 0.005, 0.01, False, True),
        (1, 746, torch.bfloat16, 0.005, 0.01, True, True),
        (4, 334, torch.bfloat16, 0.005, 0.01, True, True),
        (8, 123456, torch.bfloat16, 0.005, 0.01, True, True),
        (1, 9999, torch.float32, 1e-08, 1e-05, True, False),
        (8, 1234, torch.float32, 1e-08, 1e-05, True, False),
    ],
)
def test_distributed_log_probs(
    init_model_parallel, batch_size, seed, dtype, atol, rtol, higher_stability, ignore_last
):
    """This function is used to test our custom log prob function, we compare it against
        the more memory intensive naive implementation in the fwd and bwd pass
    """
    init_model_parallel(tensor_model_parallel_size=torch.cuda.device_count())
    device = torch.cuda.current_device()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    B, S, V_total = batch_size, 2048, 512 * world_size

    assert parallel_state.get_tensor_model_parallel_world_size() == world_size

    generator = torch.Generator(device).manual_seed((seed * 10) + rank)

    # pretend this is a col parallel output, B x S x V//TP
    fake_output = torch.randn(
        B,
        S,
        V_total // parallel_state.get_tensor_model_parallel_world_size(),
        device=device,
        requires_grad=True,
        generator=generator,
        dtype=dtype,
    )

    # target across TP must be the same
    generator = torch.Generator(device).manual_seed(seed)
    target = torch.randint(0, V_total, size=(B, S), device=device, generator=generator)

    with torch.no_grad():
        log_probs_fast = from_parallel_logits_to_logprobs(
            fake_output, target, higher_stability=higher_stability, ignore_last=ignore_last
        )
        log_probs_slow = slow_from_parallel_logits_to_logprobs(fake_output, target, ignore_last=ignore_last)

        log_probs_slow_inf_only = from_parallel_logits_to_logprobs(
            fake_output, target, inference_only=True, higher_stability=higher_stability, ignore_last=ignore_last,
        )

        torch.testing.assert_close(
            log_probs_fast,
            log_probs_slow,
            atol=atol,
            rtol=rtol,
            msg="forward pass between fast, slow and log prob calculation is not the same!",
        )
        torch.testing.assert_close(
            log_probs_slow_inf_only,
            log_probs_fast,
            atol=atol,
            rtol=rtol,
            msg="forward pass between fast, slow and log prob calculation is not the same!",
        )

    slow_from_parallel_logits_to_logprobs(fake_output, target, ignore_last=ignore_last).sum().backward()

    fake_output_grad_slow = fake_output.grad.detach().clone()

    fake_output.grad = None
    from_parallel_logits_to_logprobs(
        fake_output, target, higher_stability=higher_stability, ignore_last=ignore_last
    ).sum().backward()
    fake_output_grad_fast = fake_output.grad.detach().clone()

    torch.testing.assert_close(
        fake_output_grad_fast,
        fake_output_grad_slow,
        atol=atol,
        rtol=rtol,
        msg="backward pass between fast and slow log prob calculation is not the same!",
    )


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("batch_size,seed", [(1, 5555), (4, 6666)])
def test_distributed_entropy(init_model_parallel, batch_size, seed):
    """Test entropy against just using doing it on a single GPU
    """
    init_model_parallel(tensor_model_parallel_size=torch.cuda.device_count())
    world_size = torch.distributed.get_world_size()
    assert parallel_state.get_tensor_model_parallel_world_size() == world_size
    device = torch.cuda.current_device()

    rank = torch.distributed.get_rank()

    B, S, V_total = batch_size, 2048, 512 * world_size

    generator = torch.Generator(device).manual_seed(seed)
    full_logits = torch.randn(B, S, V_total, device=device, generator=generator, requires_grad=True)

    fake_parallel_logits = full_logits.chunk(world_size, dim=-1)[rank].detach().clone().requires_grad_()

    with torch.no_grad():
        entropy_distributed = calculate_distributed_entropy(fake_parallel_logits)
        entropy_full = calculate_entropy_full(full_logits)

        torch.testing.assert_close(
            entropy_distributed, entropy_full, msg="entropy between distributed and full path are different!"
        )

    calculate_entropy_full(full_logits).sum().backward()
    grad_full_slice = full_logits.grad.chunk(world_size, dim=-1)[rank]

    full_logits.grad = None

    calculate_distributed_entropy(fake_parallel_logits).sum().backward()
    grad_distributed = fake_parallel_logits.grad

    torch.testing.assert_close(
        grad_full_slice, grad_distributed, msg="grad of entropy between distributed and full path are different!"
    )


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "tp_size, pp_size, from_last, shape, dtype, override_dtype",
    [
        (1, 2, True, (2,), torch.float32, False),
        (1, 2, False, (2, 3), torch.bfloat16, True),
        (1, 2, True, (2, 3, 4), torch.float16, False),
        (1, 2, False, (2,), torch.int32, True),
    ],
)
def test_broadcast_within_pp(init_model_parallel, tp_size, pp_size, from_last, shape, dtype, override_dtype):
    init_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size)
    num_el = np.product(shape)
    expected = torch.arange(num_el, dtype=dtype).reshape(shape)

    tensor = None
    if (
        from_last
        and mcore_parallel_state.get_pipeline_model_parallel_rank()
        == mcore_parallel_state.get_pipeline_model_parallel_last_rank()
    ):
        tensor = expected
    elif (
        not from_last
        and mcore_parallel_state.get_pipeline_model_parallel_rank()
        == mcore_parallel_state.get_pipeline_model_parallel_first_rank()
    ):
        tensor = expected

    if override_dtype:
        # For posterity, this was default behavior when type wasn't set
        out_tensor = broadcast_tensor_within_pp(tensor, from_last=from_last, dtype=torch.float32)
        assert out_tensor.dtype == torch.float32
        torch.testing.assert_close(out_tensor.to("cpu"), expected.to("cpu").type(torch.float32))
    else:
        out_tensor = broadcast_tensor_within_pp(tensor, from_last=from_last)
        assert out_tensor.dtype == dtype
        torch.testing.assert_close(out_tensor.to("cpu"), expected.to("cpu"))


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "K,batch_size,seq_len,partition_vocab_size,sft_loss_weight,kd_loss_weight,bwd_kl,cross_tokenizer",
    [
        (3, 4, 8, 16, 0.5, 0.5, False, False),
        (3, 2, 8, 16, 0, 1.0, False, False),
        (3, 2, 8, 32, 1.0, 0, False, False),
        (3, 4, 8, 16, 0.5, 0.5, True, False),
        (3, 2, 8, 16, 0, 1.0, True, False),
        (3, 4, 8, 16, 0.5, 0.5, False, True),
    ],
)
def test_top_k_logits(
    init_model_parallel,
    K,
    batch_size,
    seq_len,
    partition_vocab_size,
    sft_loss_weight,
    kd_loss_weight,
    bwd_kl,
    cross_tokenizer,
):
    init_model_parallel(tensor_model_parallel_size=torch.cuda.device_count())
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    torch.manual_seed(0)

    true_logits = (
        torch.randint(low=0, high=100, size=(batch_size, seq_len, partition_vocab_size * world_size)) / 5
    ).to(torch.cuda.current_device())
    target_logits, target_token_ids = torch.topk(true_logits, K)
    target_log_sum_exp_logits = true_logits.exp().sum(-1).log()
    loss_mask = torch.ones(target_logits.size()[:-1]).to(torch.cuda.current_device())
    labels = torch.randint(low=0, high=partition_vocab_size * world_size, size=(batch_size, seq_len)).to(
        torch.cuda.current_device()
    )

    torch.manual_seed(torch.cuda.current_device() + 10)
    vocab_parallel_logits = torch.autograd.Variable(
        (torch.randint(low=0, high=100, size=(batch_size, seq_len, partition_vocab_size)) / 5).to(
            torch.cuda.current_device()
        ),
        requires_grad=True,
    )

    ## test loss function
    # test forward
    ctx = torch.autograd.function.FunctionCtx()

    naive_loss = naive_top_k_loss_function(
        vocab_parallel_logits,
        target_logits,
        target_token_ids,
        target_log_sum_exp_logits,
        loss_mask,
        labels,
        kd_loss_weight,
        sft_loss_weight,
        "bwd_kl" if bwd_kl else "fwd_kl",
        cross_tokenizer,
    )

    efficient_loss, kd, sft = _TopKLogitsCrossEntropy.forward(
        ctx,
        vocab_parallel_logits,
        target_logits,
        target_token_ids,
        labels,
        kd_loss_weight,
        sft_loss_weight,
        bwd_kl,
        cross_tokenizer,
    )

    ## sum p(x)logp(x) - p(x) logq(x)
    efficient_loss = torch.mean(efficient_loss)

    torch.testing.assert_close(naive_loss, efficient_loss)

    ctx.saved_tensors = ctx.to_save  ## WAR for "AttributeError: 'FunctionCtx' object has no attribute 'saved_tensors'"

    # test backward
    naive_loss.backward()
    naive_grad = vocab_parallel_logits.grad
    new_grad = _TopKLogitsCrossEntropy.backward(
        ctx, 1.0 / (batch_size * seq_len) * torch.ones(batch_size, seq_len).to(torch.cuda.current_device())
    )[0]

    torch.testing.assert_close(naive_grad, new_grad)
