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

# NOTE: Copied from https://github.com/NVIDIA/Megatron-LM/blob/main/tests/unit_tests/test_utilities.py
# to avoid having to rely on megatron repo for tests when all we need is this file

import os
from datetime import timedelta

import megatron.core.parallel_state as ps
import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous
from torch.distributed.distributed_c10d import _store_based_barrier


class TestModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, bias: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)])


class Utils:

    inited = False
    store = None

    @classmethod
    @property
    def world_size(cls):
        """Lazily grab device count"""
        return torch.cuda.device_count()

    @classmethod
    @property
    def rank(cls):
        """Lazily grab rank"""
        return int(os.environ["LOCAL_RANK"])

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(f"Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}")
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += master_ip + ":" + master_port
            rendezvous_iterator = rendezvous(init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1))
            store, _, _ = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend="nccl", world_size=Utils.world_size, rank=Utils.rank, store=store
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if torch.distributed.is_initialized() and Utils.world_size != torch.distributed.get_world_size():
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ["LOCAL_RANK"])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        if not Utils.inited:
            return
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, **kwargs,
        )
        Utils.inited = True
