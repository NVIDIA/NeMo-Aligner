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

import pytest
import torch
from megatron.core import parallel_state, tensor_parallel

from nemo_aligner.utils.distributed import (
    calculate_distributed_entropy,
    from_parallel_logits_to_logprobs,
    masked_global_mean_var,
)
from nemo_aligner.utils.ppo_utils import calculate_entropy

"""A file to test the core distributed function calls in RLHF"""


def slow_from_parallel_logits_to_logprobs(parallel_logits, tokens):
    """a slow but very safe way of computing logits -> logprobs. Uses a lot of memory but good for testing"""
    # Gather logits across all TP ranks for testing
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(parallel_logits)

    # Convert from logits to log-probs.
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    full_log_probs = full_log_probs[:, :-1, :].contiguous()
    indices = tokens[:, 1:].unsqueeze(-1)
    log_probs = torch.gather(input=full_log_probs, dim=2, index=indices).squeeze(dim=-1).contiguous()
    return log_probs


def calculate_entropy_full(logits):
    # Convert from logits to log-probs.
    full_log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    full_log_probs = full_log_probs[:, :-1, :].contiguous()
    return calculate_entropy(full_log_probs)


class TestDistributedFunctions:
    def _init_distributed(self, local_rank, main_address, main_port, nprocs):
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl" if torch.cuda.is_available() else "gloo",
                rank=local_rank,
                world_size=nprocs,
                init_method=f"tcp://{main_address}:{main_port}",
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

    def _run_test(self, func, *args):
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        for nproc in range(nprocs):
            torch.multiprocessing.spawn(func, args=("localhost", 1234, nproc, *args), nprocs=nproc, join=True)

    def _test_masked_global_mean_var(self, *args, **kwargs):
        self._init_distributed(*args, **kwargs)
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

        assert torch.allclose(
            global_mean_pt, global_mean
        ), f"expected global mean {global_mean_pt} but got {global_mean}"
        assert torch.allclose(
            global_var_pt, global_var
        ), f"expected global var {global_var_pt} but we got {global_var}"

    def _test_distributed_log_probs(
        self, local_rank, main_address, main_port, nprocs, batch_size, seed, dtype, atol, rtol, higher_stability
    ):
        """This function is used to test our custom log prob function, we compare it against
            the more memory intensive naive implementation in the fwd and bwd pass
        """
        self._init_distributed(local_rank, main_address, main_port, nprocs)
        device = torch.cuda.current_device()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        B, S, V_total = batch_size, 2048, 512 * world_size

        # pretend initalize the tensor model_parallel so the util function works
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
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
            log_probs_fast = from_parallel_logits_to_logprobs(fake_output, target, higher_stability=higher_stability)
            log_probs_slow = slow_from_parallel_logits_to_logprobs(fake_output, target)

            log_probs_slow_inf_only = from_parallel_logits_to_logprobs(
                fake_output, target, inference_only=True, higher_stability=higher_stability
            )

            assert torch.allclose(log_probs_fast, log_probs_slow, atol=atol, rtol=rtol) and torch.allclose(
                log_probs_slow_inf_only, log_probs_fast, atol=atol, rtol=rtol
            ), "forward pass between fast, slow and log prob calculation is not the same!"

        slow_from_parallel_logits_to_logprobs(fake_output, target).sum().backward()

        fake_output_grad_slow = fake_output.grad.detach().clone()

        fake_output.grad = None
        from_parallel_logits_to_logprobs(fake_output, target, higher_stability=higher_stability).sum().backward()
        fake_output_grad_fast = fake_output.grad.detach().clone()

        assert torch.allclose(
            fake_output_grad_fast, fake_output_grad_slow, atol=atol, rtol=rtol
        ), "backward pass between fast and slow log prob calculation is not the same!"

    def _test_distributed_entropy(self, local_rank, main_address, main_port, nprocs, batch_size, seed):
        """Test entropy against just using doing it on a single GPU
        """
        self._init_distributed(local_rank, main_address, main_port, nprocs)
        device = torch.cuda.current_device()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        B, S, V_total = batch_size, 2048, 512 * world_size

        # pretend initalize the tensor model_parallel so the util function works
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        assert parallel_state.get_tensor_model_parallel_world_size() == world_size

        generator = torch.Generator(device).manual_seed(seed)
        full_logits = torch.randn(B, S, V_total, device=device, generator=generator, requires_grad=True)

        fake_parallel_logits = full_logits.chunk(world_size, dim=-1)[rank].detach().clone().requires_grad_()

        with torch.no_grad():
            entropy_distributed = calculate_distributed_entropy(fake_parallel_logits)
            entropy_full = calculate_entropy_full(full_logits)

            assert torch.allclose(
                entropy_distributed, entropy_full
            ), "entropy between distributed and full path are different!"

        calculate_entropy_full(full_logits).sum().backward()
        grad_full_slice = full_logits.grad.chunk(world_size, dim=-1)[rank]

        full_logits.grad = None

        calculate_distributed_entropy(fake_parallel_logits).sum().backward()
        grad_distributed = fake_parallel_logits.grad

        assert torch.allclose(
            grad_full_slice, grad_distributed
        ), "grad of entropy between distributed and full path are different!"

    @pytest.mark.run_only_on("GPU")
    def test_distributed_masked_global_mean_var(self):
        self._run_test(self._test_masked_global_mean_var)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "batch_size,seed,dtype,atol,rtol,higher_stability",
        [
            (1, 9999, torch.float32, 1e-08, 1e-05, False),
            (4, 100, torch.float32, 1e-08, 1e-05, False),
            (8, 1234, torch.float32, 1e-08, 1e-05, False),
            (1, 9999, torch.float32, 1e-08, 1e-05, True),
            (4, 100, torch.float32, 1e-08, 1e-05, True),
            (8, 1234, torch.float32, 1e-08, 1e-05, True),
            (1, 746, torch.bfloat16, 0.005, 0.01, False),
            (4, 334, torch.bfloat16, 0.005, 0.01, False),
            (8, 123456, torch.bfloat16, 0.005, 0.01, False),
            (1, 746, torch.bfloat16, 0.005, 0.01, True),
            (4, 334, torch.bfloat16, 0.005, 0.01, True),
            (8, 123456, torch.bfloat16, 0.005, 0.01, True),
        ],
    )
    def test_distributed_log_probs(self, batch_size, seed, dtype, atol, rtol, higher_stability):
        self._run_test(self._test_distributed_log_probs, batch_size, seed, dtype, atol, rtol, higher_stability)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("batch_size,seed", [(1, 5555), (4, 6666)])
    def test_distributed_entropy(self, batch_size, seed):
        self._run_test(self._test_distributed_entropy, batch_size, seed)
