import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CACHE_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
# if OUTPUT_DIR does not exist, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
VALUE_DATA_FILE = "value_data_{data_id}.pt"
POLICY_DATA_FILE = "policy_data_{data_id}.pt"

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

        diff = 1
        if idx + 1 in indices:
            diff = len(list_of_samples[idx + 1]["tokens"]) - len(item["tokens"])
        if diff > 1:
            for key in keys_to_stack:
                if key == "actions":
                    list_obj = [isinstance(element, list) for element in item[key]]
                    # has to have list in actions
                    assert any(list_obj)
                    fixed = [element[0] if isinstance(element, list) else element for element in item[key]]
                    item[key] = fixed
                sample_all[key].append(item[key])
                for _ in range(diff - 1):
                    if key == "actions":
                        sample_all[key].append([-1] * len(item[key]))
                    elif key == "reward":
                        sample_all[key].append(item[key])
                    elif key == "action_probs":
                        sample_all[key].append(np.zeros_like(item[key]))
        else:
            for key in keys_to_stack:
                if key == "actions":
                    list_obj = [isinstance(element, list) for element in item[key]]
                    if any(list_obj):
                        fixed = [element[0] if isinstance(element, list) else element for element in item[key]]
                        item[key] = fixed
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

        for toks, token_values, reward in value["value_memory"]:
            batch["tokens"].append(list(toks))
            batch["token_values"].append(list(token_values))
            batch["reward"].append(reward)

        batch["context_length"] = len(value["backup_root_states"])
        batch["data_id"] = value["data_id"]

        # will be provided by policy soon
        GLOBAL_CONTEXT_LENGTH_DICT[value["data_id"]] = len(value["backup_root_states"])

        batches.append(batch)

    return batches


total_data_ids = set()
length_of_value_before_filter = 0
length_of_policy_before_filter = 0
length_of_value_after_filter = 0
length_of_policy_after_filter = 0
length_of_policy_after_wrong_question_filter = 0
finished_ids = set()
num_questions_correct = 0
for p in tqdm(sorted(Path(CACHE_DIR).glob("*.pt"))):
    print(p)
    save_file = torch.load(p)
    data_ids = save_file["data_ids"]
    total_data_ids.update(data_ids)
    x = save_file["mcts_outputs"]

    assert len(x[1::2]) == len(x[::2])

    for output_policy, output_value in zip(x[::2], x[1::2]):
        # values.extend(batch_value_memory(output_value))
        # policies.extend(batch_policy_memory(output_policy))
        values = batch_value_memory(output_value)
        length_of_value_before_filter += len(values)
        values = [v for v in values if len(v["tokens"]) > 0]
        length_of_value_after_filter += len(values)

        policies = batch_policy_memory(output_policy)
        length_of_policy_before_filter += len(policies)
        policies = [p for p in policies if len(p["tokens"]) > 0]
        length_of_policy_after_filter += len(policies)
        policies = [p for p in policies if all(r > 0 for r in p["reward"])]
        length_of_policy_after_wrong_question_filter += len(policies)

        finished_ids.update([i["data_id"] for i in policies])

        for policy in policies:
            if all(x == 1 for x in policy["reward"]):
                num_questions_correct += 1

            filename = os.path.join(OUTPUT_DIR, POLICY_DATA_FILE.format(data_id=policy["data_id"]))
            torch.save(policy, filename)

        for value in values:
            filename = os.path.join(OUTPUT_DIR, VALUE_DATA_FILE.format(data_id=value["data_id"]))
            torch.save(value, filename)


print("### FILTERING OUT empty lists")

print("length of value before filtering", length_of_value_before_filter)
print("length of value after filtering", length_of_value_after_filter)

print("length of policies before filtering", length_of_policy_before_filter)
print("length of policies after filtering", length_of_policy_after_filter)

print("not finished ids", total_data_ids - finished_ids)

print("total data ids", len(total_data_ids))

data_metrics = {
    "num_questions_correct": num_questions_correct,
    "num_questions": length_of_policy_after_filter,
    "accuracy": num_questions_correct / length_of_policy_after_filter,
}

print(data_metrics)

print(
    "questions before filter {} questions after filter {}".format(
        length_of_policy_after_filter, length_of_policy_after_wrong_question_filter
    )
)
