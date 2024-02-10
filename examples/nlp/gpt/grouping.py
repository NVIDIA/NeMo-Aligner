from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CACHE_DIR = "/home/geshen/mcts/NeMo-Aligner/examples/nlp/gpt/cache_dir"

GLOBAL_CONTEXT_LENGTH_DICT = {}

grouper = defaultdict(list)

all_idx = []


values = []
policies = []


def process_single_sample(list_of_samples):
    # has to be a list for a single data id
    sample_all = defaultdict(list)
    data_id = list_of_samples[0]["data_id"]
    context_length = GLOBAL_CONTEXT_LENGTH_DICT[data_id]

    tokens = [x["tokens"] for x in list_of_samples]
    assert min(len(x) for x in tokens) == context_length
    max_length = max(len(x) for x in tokens)

    sample_all["context_length"] = context_length
    sample_all["response_length"] = max_length

    tokens_to_use = [x for x in tokens if len(x) == max_length][0]

    def check_func(x):
        return x == tokens_to_use[: len(x)]

    assert all(map(check_func, tokens))

    lengths = [len(x["tokens"]) for x in list_of_samples]

    assert lengths

    indices = np.argsort(lengths)

    sorted_index = np.sort(indices)

    sample_all["tokens"] = tokens_to_use
    sample_all["data_id"] = data_id

    # lengths should be sorted
    assert np.allclose(sorted_index, indices)

    keys_to_stack = {"action_probs", "reward", "actions"}

    for idx in indices:
        item = list_of_samples[idx]

        for key in keys_to_stack:
            sample_all[key].append(item[key])

    for k in keys_to_stack:
        sample_all[k] = np.stack(sample_all[k])

    return sample_all


def batch_policy_memory(output):
    batched_dict = defaultdict(list)

    for output_sample in output:
        batched_dict[output_sample["data_id"]].append(output_sample)

    return list(map(process_single_sample, batched_dict.values()))


# no padding so keep everything in a list
def batch_value_memory(output_value):
    batches = []

    for value in output_value:
        batch = defaultdict(list)

        for toks, reward in value["value_memory"]:
            batch["tokens"].append(list(toks))
            batch["reward"].append(reward)

        batch["context_length"] = len(value["backup_root_states"])
        batch["data_id"] = value["data_id"]

        # will be provided by policy soon
        GLOBAL_CONTEXT_LENGTH_DICT[value["data_id"]] = len(value["backup_root_states"])

        batches.append(batch)

    return batches


for p in tqdm(sorted(Path(CACHE_DIR).glob("*.pt"))):
    print(p)
    x = torch.load(p)["mcts_outputs"]

    assert len(x[1::2]) == len(x[::2])

    for output_policy, output_value in zip(x[::2], x[1::2]):
        values.extend(batch_value_memory(output_value))
        policies.extend(batch_policy_memory(output_policy))

torch.save({"policies": policies, "values": values}, "data.pt")
