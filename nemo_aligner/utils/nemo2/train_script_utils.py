from nemo.lightning import _strategy_lib
from nemo.lightning.pytorch.strategies.megatron_strategy import ParallelismConfig

def setup_distributed():

    ## TODO: make this configurable 
    parallelism = ParallelismConfig(
        tensor_model_parallel_size=self.tensor_model_parallel_size,
        pipeline_model_parallel_size=self.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
        microbatch_group_size_per_vp_stage=self.microbatch_group_size_per_vp_stage,
        context_parallel_size=self.context_parallel_size,
        sequence_parallel=self.sequence_parallel,
        expert_model_parallel_size=self.expert_model_parallel_size,
        moe_extended_tp=self.moe_extended_tp,
        pipeline_dtype=self.pipeline_dtype,
    )

    ## TODO: grab these from torch
    ## calls initialize_model_parallel_for_nemo
    _strategy_lib.init_parallel_ranks(
        world_size=self.cluster_environment.world_size(),
        global_rank=self.cluster_environment.global_rank(),
        local_rank=self.cluster_environment.local_rank(),
        parallel_config=parallelism,
    )

    ## calls parallel_state.initialize_model_parallel
    ## uses app state to get model parallel sizes
    ## app state should be setup above 
    _strategy_lib.init_model_parallel()

