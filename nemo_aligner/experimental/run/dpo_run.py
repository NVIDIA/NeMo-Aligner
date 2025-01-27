import os
from typing import Callable

import fiddle as fdl
import nemo_run as run

# Import executors module to register the executor factories to the CLI
import nemo_aligner.experimental.run.executors
from nemo_aligner.experimental.run.dpo_loop import dpo_loop


@run.cli.entrypoint(namespace="TODO", type="experiment")
def experiment(
    restore_from_path: str,
    tp: int = 1,
    pp: int = 1,
    vp: int = None,
    # Experiment args
    executor: run.Executor = None,  # = local_executor_torchrun(),
):
    # TODO: executor passed into Experiment must be instance and not a config, we we build before we enter
    # breakpoint()
    # executor = fdl.build(executor)
    # breakpoint()
    with run.Experiment("dpo_experiment", executor=executor, log_level="WARN") as exp:

        dpo_loop_fn = run.Partial(dpo_loop, restore_from_path=restore_from_path, tp=tp, pp=pp, vp=vp)

        exp.add(dpo_loop_fn, name="TODO", tail_logs=True)
        exp.run(detach=False)
        # ctx.launch(experiment=exp)


if __name__ == "__main__":

    run.cli.main(experiment)
