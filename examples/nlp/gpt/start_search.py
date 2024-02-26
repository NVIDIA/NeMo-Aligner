from tasks import get_search_for_batch
from celery.result import allow_join_result
from celery import group
import torch
from datasets import load_dataset
from nemo.core.config import hydra_runner
from tqdm import tqdm



@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg):
    dataset = load_dataset("gsm8k", "main")

    total = len(dataset)
    total = 4

    data_id_list = torch.tensor(range(total)).split(cfg.model.mcts.rollout_micro_batch_size)  # list of data to process
    search_for_batch = get_search_for_batch(cfg.server_url)


    results  = [search_for_batch.delay(data.tolist()) for data in data_id_list]
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



    # job = group(search_for_batch.s(data.tolist()) for data in data_id_list)

    # result = job.apply_async()

    # global_pbar = tqdm(total, desc="Search Global Progress")
    # while not result.ready():
    #     for subtask in list(result.children):  # Iterate over a copy of the list
    #         try:
    #             if subtask.ready():
    #                 args = subtask.get()
    #                 global_pbar.update(cfg.model.mcts.rollout_micro_batch_size)
    #                 global_pbar.write(f"Finished {args}")
    #                 result.children.remove(subtask)  # Remove the subtask from the list
    #         except TimeoutError:
    #             pass

if __name__ == "__main__":
    main()