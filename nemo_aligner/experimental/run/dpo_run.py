import os
from typing import Callable

import fiddle as fdl
import nemo_run as run

# Import executors module to register the executor factories to the CLI
import nemo_aligner.experimental.run.executors
from nemo_aligner.data.nlp.config import DPODataConfig
from nemo_aligner.experimental.run.dpo_loop import dpo_loop
from nemo_aligner.experimental.run.executors import local_executor_torchrun
from nemo_aligner.experimental.run.gpt_dpo import default_dpo_data_config, megatron_adam_optimizer
from nemo_aligner.utils.nemo2.optim import MegatronOptimizer


@run.cli.entrypoint(namespace="TODO", type="experiment", name="DPO Post-Training")
def experiment(
    restore_from_path: str,
    optimizer: MegatronOptimizer = megatron_adam_optimizer(),
    data_config: DPODataConfig = default_dpo_data_config(),
    tp: int = 1,
    pp: int = 1,
    vp: int = None,
    # Experiment args
    executor: run.Executor = None,
):
    """
    TODO: help docstring
    """
    # TODO: executor passed into Experiment must be instance and not a config, we we build before we enter
    # breakpoint()
    # executor = fdl.build(executor)
    # breakpoint()

    if executor is None:
        raise ValueError(
            "executor must be provided, consider either executor=local_executor_torchrun or executor=slurm_executor"
        )

    with run.Experiment("dpo_experiment", executor=executor, log_level="WARN") as exp:

        dpo_loop_fn = run.Partial(
            dpo_loop,
            #
            restore_from_path=restore_from_path,
            optimizer=optimizer,
            data_config=data_config,
            tp=tp,
            pp=pp,
            vp=vp,
        )

        exp.add(dpo_loop_fn, name="TODO", tail_logs=True)
        exp.run(detach=False)
        # ctx.launch(experiment=exp)


if __name__ == "__main__":

    run.cli.main(experiment)
