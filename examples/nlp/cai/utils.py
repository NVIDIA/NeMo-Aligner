import os
from collections import defaultdict
from multiprocessing import Pool

from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def _pool_process_item(item_index: int, max_seq_length: int):
    global g_dataset

    item = g_dataset[item_index]
    item_mask = item["mask"]
    item_mask_len = item_mask.shape[0]
    need_to_remove = item_mask[: max_seq_length + 1].sum().item() == 0
    return item_index, item_mask_len, need_to_remove


def remove_long_dialogs(
    input_file_path: str,
    max_seq_length: int,
    tokenizer_model: str,
    tokenizer_library: str,
    output_dir: str,
    use_pool: bool,
):
    assert os.path.isfile(input_file_path)
    input_file_name, input_file_extension = os.path.splitext(os.path.basename(input_file_path))

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = os.path.join(
        output_dir, f"{input_file_name}_remove_long_dialogs_max_seq_{max_seq_length}{input_file_extension}"
    )

    # load tokenizer model
    tokenizer = get_nmt_tokenizer(library=tokenizer_library, tokenizer_model=tokenizer_model,)

    # create dataset object
    dataset = GPTSFTChatDataset(
        file_path=input_file_path, tokenizer=tokenizer, max_seq_length=max_seq_length, min_seq_length=1
    )

    removed_ids = set()
    length_statistics = defaultdict(int)

    if use_pool:

        def init_worker(shared_queue):
            # declare scope of a new global variable
            global g_dataset

            # store argument in the global variable for this process
            g_dataset = shared_queue

        with Pool(initializer=init_worker, initargs=(dataset,)) as pool:
            tasks = [pool.apply_async(_pool_process_item, (i, max_seq_length)) for i in range(len(dataset))]
            for task in tqdm(tasks):
                item_index, item_mask_len, need_to_remove = task.get()

                if need_to_remove:
                    removed_ids.add(item_index)
                length_statistics[item_mask_len] += 1
    else:
        for i in tqdm(range(len(dataset))):
            item_mask = dataset[i]["mask"]
            item_mask_len = item_mask.shape[0]
            need_to_remove = item_mask[: max_seq_length + 1].sum().item() == 0

            if need_to_remove:
                removed_ids.add(i)
            length_statistics[item_mask_len] += 1

    print(f"removed {(len(removed_ids) / len(tasks)) * 100:.2f}%")

    # note: we assume each sample is a single line.
    with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_name, "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            if i in removed_ids:
                continue
            o.write(line)

    return dict(output_file=output_file_name, removed_ids=removed_ids, length_statistics=length_statistics)
