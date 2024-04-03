import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CACHE_DIR = sys.argv[1]
DATA_FILE = "solar_no_value_head_iter0.pt"

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


total_data_ids = set()
for p in tqdm(sorted(Path(CACHE_DIR).glob("*.pt"))):
    print(p)
    save_file = torch.load(p)
    data_ids = save_file["data_ids"]
    total_data_ids.update(data_ids)
    x = save_file["mcts_outputs"]

    assert len(x[1::2]) == len(x[::2])

    for output_policy, output_value in zip(x[::2], x[1::2]):
        values.extend(batch_value_memory(output_value))
        policies.extend(batch_policy_memory(output_policy))

print("### FILTERING OUT empty lists")

print("length of value before filtering", len(values))
values = [v for v in values if len(v["tokens"]) > 0]
print("length of value after filtering", len(values))

print("length of policies before filtering", len(policies))
policies = [p for p in policies if len(p["tokens"]) > 0]
print("length of policies after filtering", len(policies))

print("total data ids", len(total_data_ids))

# TODO(geshen): should we shuffle the data?
policy_data, value_data = policies, values

num_questions_correct = 0

for p in policy_data:
    if all(x > 0 for x in p["reward"]):
        num_questions_correct += 1

data_metrics = {
    "num_questions_correct": num_questions_correct,
    "num_questions": len(policy_data),
    "accuracy": num_questions_correct / len(policy_data),
}

print(data_metrics)

print("filtering out the negative reward cases in the policy")
before_filter_len = len(policies)
policies = [p for p in policies if all(r > 0 for r in p["reward"])]
print("questions before filter {} questions after filter {}".format(before_filter_len, len(policies)))

torch.save({"policies": policies, "values": values}, DATA_FILE)
