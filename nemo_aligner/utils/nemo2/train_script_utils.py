import os
import torch

from nemo.lightning import _strategy_lib
from nemo.lightning.pytorch.strategies.megatron_strategy import ParallelismConfig
from nemo.utils import AppState

from megatron.core.num_microbatches_calculator import (
    ConstantNumMicroBatchesCalculator,
    get_current_global_batch_size,
    get_micro_batch_size,
    get_num_microbatches,
    init_num_microbatches_calculator,
)

## utils from PTL
def _setup_distributed() -> None:
    _process_group_backend = "nccl"
    #assert self.cluster_environment is not None

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(_process_group_backend, rank=global_rank, world_size=world_size)

def _init_mb_calculator(
    global_batch_size,
    micro_batch_size,
    global_rank,
):
    if global_batch_size and micro_batch_size is not None:
        # TODO: add rampup_batch_size here when we have it implemented
        from megatron.core.num_microbatches_calculator import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

        app_state = AppState()

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
            init_num_microbatches_calculator(
                rank=global_rank,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_batch_size,
                data_parallel_size=app_state.data_parallel_size,
                rampup_batch_size=None, #rampup_batch_size,
                decrease_batch_size_if_needed=False,
            )
        else:
            if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatchesCalculator):
                assert get_current_global_batch_size() == global_batch_size
                assert get_micro_batch_size() == micro_batch_size
                assert get_num_microbatches() == global_batch_size // (
                    micro_batch_size * app_state.data_parallel_size
                )
            else:
                raise Exception("Microbatch calculator already initialized.")

def setup_distributed(parallelism, data_config):

    _setup_distributed()

    ## calls initialize_model_parallel_for_nemo
    _strategy_lib.init_parallel_ranks(
        world_size=int(os.environ["WORLD_SIZE"]),
        global_rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        parallel_config=parallelism,
    )

    ## calls parallel_state.initialize_model_parallel
    ## uses app state to get model parallel sizes
    ## app state should be setup above 
    _strategy_lib.init_model_parallel()

    _init_mb_calculator(
        data_config.global_batch_size,
        data_config.micro_batch_size,
        int(os.environ["RANK"]),
    )
