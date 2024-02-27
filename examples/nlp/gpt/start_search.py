import functools
import os
import pathlib
import sys
from typing import Any, Callable, Optional

import torch
from datasets import load_dataset
from hydra import TaskFunction
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

from tasks import get_search_for_batch


def hydra_runner(
    config_path: Optional[str] = ".", config_name: Optional[str] = None, schema: Optional[Any] = None
) -> Callable[[TaskFunction], Any]:
    """
    Decorator used for passing the Config paths to main function.
    Optionally registers a schema used for validation/providing default values.

    Args:
        config_path: Optional path that will be added to config search directory.
            NOTE: The default value of `config_path` has changed between Hydra 1.0 and Hydra 1.1+.
            Please refer to https://hydra.cc/docs/next/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path/
            for details.
        config_name: Pathname of the config file.
        schema: Structured config  type representing the schema used for validation/providing default values.
    """

    def decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # Check it config was passed.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()

                # Parse arguments in order to retrieve overrides
                parsed_args = args.parse_args()  # type: argparse.Namespace

                # Get overriding args in dot string format
                overrides = parsed_args.overrides  # type: list

                # Disable the creation of .hydra subdir
                # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory
                overrides.append("hydra.output_subdir=null")
                # Hydra logging outputs only to stdout (no log file).
                # https://hydra.cc/docs/configure_hydra/logging
                overrides.append("hydra/job_logging=stdout")

                # Set run.dir ONLY for ExpManager "compatibility" - to be removed.
                overrides.append("hydra.run.dir=.")

                # Check if user set the schema.
                if schema is not None:
                    # Create config store.
                    cs = ConfigStore.instance()

                    # Get the correct ConfigStore "path name" to "inject" the schema.
                    if parsed_args.config_name is not None:
                        path, name = os.path.split(parsed_args.config_name)
                        # Make sure the path is not set - as this will disable validation scheme.
                        if path != "":
                            sys.stderr.write(
                                f"ERROR Cannot set config file path using `--config-name` when "
                                "using schema. Please set path using `--config-path` and file name using "
                                "`--config-name` separately.\n"
                            )
                            sys.exit(1)
                    else:
                        name = config_name

                    # Register the configuration as a node under the name in the group.
                    cs.store(name=name, node=schema)  # group=group,

                # Wrap a callable object with name `parse_args`
                # This is to mimic the ArgParser.parse_args() API.
                def parse_args(self, args=None, namespace=None):
                    return parsed_args

                parsed_args.parse_args = parse_args

                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                # argparse_wrapper = _argparse_wrapper(args)
                argparse_wrapper = parsed_args

                _run_hydra(
                    args=argparse_wrapper,
                    args_parser=args,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                )

        return wrapper

    return decorator


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg):
    dataset = load_dataset("gsm8k", "main")

    total = len(dataset["train"])
    save_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")

    existing_files = pathlib.Path(save_dir).rglob("*.pt")
    ids = set()
    for file_name in existing_files:
        id_str = file_name.name.split("_")[0].split("-")
        ids.update([int(id) for id in id_str])

    all_ids = set(range(total))
    non_processed_ids = all_ids - ids

    data_id_list = torch.tensor(list(non_processed_ids)).split(
        cfg.model.mcts.rollout_micro_batch_size
    )  # list of data to process

    search_for_batch = get_search_for_batch(cfg.server_url)

    results = [search_for_batch.delay(data.tolist()) for data in data_id_list]
    global_pbar = tqdm(total=total, desc="Search Global Progress")
    while len(results) > 0:
        for subtask in results:  # Iterate over a copy of the list
            try:
                if subtask.ready():
                    args = subtask.get()
                    global_pbar.update(cfg.model.mcts.rollout_micro_batch_size)
                    global_pbar.write(f"Finished {args}")
                    results.remove(subtask)
                    # results.children.remove(subtask)  # Remove the subtask from the list
            except TimeoutError:
                pass


if __name__ == "__main__":
    main()
