import nemo_run as run

DEFAULT_ENV_VARS = {
    "TRANSFORMERS_OFFLINE": "1",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
}

# use this one with run.cli.main
@run.cli.factory
def local_executor_torchrun() -> run.Config[run.Executor]:
    # TODO: need mpi one
    return run.Config(run.LocalExecutor, ntasks_per_node=1, launcher="torchrun", env_vars=DEFAULT_ENV_VARS)


@run.cli.factory
def slurm_executor() -> run.Config[run.Executor]:
    return run.Config(
        run.SlurmExecutor,
        packager=run.Config(run.GitArchivePackager, include_pattern="*.py",),
        account="<<ACCOUNT>>",
        partition="<<PARTITION>>",
        time="04:00:00",
        retries=1,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=1,
        mem="0",
        exclusive=True,
        container_image="nvcr.io/nvidia/nemo:dev",
        heterogeneous=False,
        memory_measure=False,
        tunnel=run.Config(run.SSHTunnel, job_dir="/path/to/remote/job/dir", host="<<HOST>>", user="<<USER>>",),
        container_mounts=[],
        env_vars={},
    )
