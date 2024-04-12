import functools
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Optional

import amqp
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


class EvalMTBenchmark(object):
    def __init__(self, output_file, data_file):
        self.output_file = output_file + ".jsonl"
        self.output_tmp_file = output_file + ".jsonl.tmp"
        self.data_file = data_file

    def get_first_turn_template(self):
        return """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{text}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""

    def get_next_turn_template(self):
        return """<extra_id_1>User
{text}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""

    def get_input(self, obj):
        return self.get_first_turn_template().format(**obj)

    def run(self):
        all = []

        finished_sample_map = {}
        # test if the output_tmp_file exists
        if os.path.exists(self.output_tmp_file):
            print(f"{self.output_tmp_file} exists, loading from it")
            with open(self.output_tmp_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    id = obj["question_id"]
                    finished_sample_map[id] = obj
            print(f"loaded {len(finished_sample_map)} elements")

        with open(self.output_tmp_file, "a", encoding="utf-8") as tf:
            with open(self.data_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    # load it from the cache
                    if obj["question_id"] in finished_sample_map:
                        print(f"found {obj['question_id']}, load it")
                        all.append(finished_sample_map[obj["question_id"]])
                        continue
                    print(f"calculate {obj['question_id']}")
                    input = ""
                    obj["choices"] = [{"index": 0, "turns": []}]
                    obj["model_id"] = self.output_file
                    obj["tstamp"] = time.time()
                    for i in range(len(obj["turns"])):
                        if i == 0:
                            input = self.get_input({"text": obj["turns"][i]})
                        else:
                            input = input + self.get_next_turn_template().format(**{"text": obj["turns"][i]})
                        text = get_response([input])
                        text = text[0]
                        if text.find("<extra_id_0>") < 0:
                            # hack due to the problem that huggingface's tokenizer strips out the <extra_id_x> token
                            input = (
                                input.replace("<extra_id_0>", "")
                                .replace("<extra_id_1>", "")
                                .replace("<extra_id_2>", "")
                            )
                        ans = text[len(input) :]
                        if ans.endswith("\x11"):
                            ans = ans[: -len("\x11")]
                        if ans.endswith("<extra_id_1>"):
                            ans = ans[: -len("<extra_id_1>")]
                        print(ans)
                        obj["choices"][0]["turns"].append(ans.strip())
                        input = input + ans.strip() + "\n"
                    all.append(obj)
                    # save the record to the cache too
                    print(f"save {obj['question_id']} to cache")
                    tf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    tf.flush()

        with open(self.output_file, "w", encoding="utf-8") as f:
            for obj in all:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_answer(data):
    full_text = data["full_text"]
    context = data["context"]
    ans = full_text[len(context) :]
    if ans.endswith("\x11"):
        ans = ans[: -len("\x11")]
    if ans.endswith("<extra_id_1>"):
        ans = ans[: -len("<extra_id_1>")]
    return ans.strip()


class FakeResult:
    def __init__(self, obj) -> None:
        self.obj = obj

    def get(self):
        return [[self.obj]]

    def ready(self):
        return True


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg):
    data_path = os.path.join(cfg.exp_manager.explicit_log_dir, "mcts_cache")
    files = pathlib.Path(data_path).rglob("*.pt")

    finished = {}
    for file in files:
        data = torch.load(file)
        outputs = data["mcts_outputs"][0]
        for item in outputs:
            finished[item["data_id"]] = item

    search_for_batch = get_search_for_batch(cfg.server_url, cfg.backend_url)

    mt_bench_file = os.path.join(data_path, "mt_bench.jsonl")
    eval = EvalMTBenchmark("output", mt_bench_file)
    first_turn_inputs = {}
    with open(eval.data_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            input = eval.get_input({"text": obj["turns"][0]})
            first_turn_inputs[obj["question_id"]] = {"turn": 0, "question": input}
    data_points = [
        {"question": v["question"], "answer": "", "data_id": k}
        for k, v in first_turn_inputs.items()
        if k not in finished
    ]
    chunked_data_points = chunks(data_points, cfg.model.mcts.rollout_micro_batch_size)
    results = [search_for_batch.delay(data) for data in chunked_data_points]
    finished_reuslts = [FakeResult(finished[k]) for k, v in first_turn_inputs.items() if k in finished]
    results.extend(finished_reuslts)

    turn_2_offset = 1000
    global_pbar = tqdm(total=len(data_points) * 2, desc="Search Global Progress")
    while len(results) > 0:
        for subtask in results:  # Iterate over a copy of the list
            try:
                if subtask.ready():
                    output = subtask.get()
                    output = output[0]
                    args = [i["data_id"] for i in output]
                    for item in output:
                        finished[item["data_id"]] = item

                    # start to submit the next turn job
                    second_turn_inputs = {}
                    with open(eval.data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line)
                            question_id = obj["question_id"]
                            if question_id not in args:
                                continue

                            for i in range(len(obj["turns"])):
                                if i == 0:
                                    input = eval.get_input({"text": obj["turns"][i]})
                                else:
                                    input = input + eval.get_next_turn_template().format(**{"text": obj["turns"][i]})
                                    continue
                                if question_id in finished:
                                    ans = get_answer(finished[question_id])
                                else:
                                    raise ValueError("The first turn should be finished")
                                input = input + ans + "\n"
                            second_turn_inputs[obj["question_id"] + turn_2_offset] = {"turn": 1, "question": input}
                    data_points = [
                        {"question": v["question"], "answer": "", "data_id": k}
                        for k, v in second_turn_inputs.items()
                        if k not in finished
                    ]

                    chunked_data_points = chunks(data_points, cfg.model.mcts.rollout_micro_batch_size)
                    results.extend([search_for_batch.delay(data) for data in chunked_data_points])
                    global_pbar.update(cfg.model.mcts.rollout_micro_batch_size)
                    global_pbar.write(f"Finished {args}")
                    results.remove(subtask)
                    # results.children.remove(subtask)  # Remove the subtask from the list
            except TimeoutError:
                pass
            except amqp.exceptions.PreconditionFailed:
                global_pbar.write("RabbitMQ connection failed")

    all = []
    # save the final result
    with open(eval.data_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["choices"] = [{"index": 0, "turns": []}]
            question_id = obj["question_id"]
            for i in range(len(obj["turns"])):
                if i == 0:
                    ans = get_answer(finished[question_id])
                    obj["choices"][0]["turns"].append(ans)
                else:
                    ans = get_answer(finished[question_id + turn_2_offset])
                    obj["choices"][0]["turns"].append(ans)
            all.append(obj)

    with open("output.jsonl", "w", encoding="utf-8") as f:
        for obj in all:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
