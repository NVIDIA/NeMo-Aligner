import os
import pathlib

import torch
from datasets import load_dataset
from tqdm import tqdm

from nemo.core.config import hydra_runner
from tasks import get_search_for_batch


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
