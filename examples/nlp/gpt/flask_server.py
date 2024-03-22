from pathlib import Path
import argparse
import random

from flask import Flask, request

app = Flask(__name__)

DATASET_IDX = None

def find_completed_ids(path):
    completed_ids = set()
    if path is None or not Path(path).exists() or not Path(path).is_dir():
        return completed_ids

    for p in Path(path).glob("*.pt"):
        ids = p.name.split("-")
        ids[-1] = ids[-1].split("_")[0]

        ids = list(map(int, ids))

        completed_ids = completed_ids.union(set(ids))

    return completed_ids

@app.route("/get_idx", methods=["PUT"])
def get_http_idx():
    global DATASET_IDX

    batch_size = request.json["batch_size"]
    batches = []

    while len(DATASET_IDX) > 0 and len(batches) < batch_size:
        batches.append(DATASET_IDX.pop())

    print("AFTER", len(DATASET_IDX))

    return list(reversed(batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-length", type=int, required=True,
    )
    parser.add_argument(
        "--port", type=int, default=9999,
    )
    parser.add_argument(
        "--cache-path", type=str, required=False, default=None
    )

    args = parser.parse_args()
    completed_ids = find_completed_ids(args.cache_path)
    print("## FOUND COMPLETED IDS", len(completed_ids))

    random.seed(98)
    idx_list = list(range(args.dataset_length))
    random.shuffle(idx_list)
    idx_list = [idx for idx in idx_list if idx not in completed_ids]

    DATASET_IDX = idx_list
    print("### ALL DATASET IDX", len(DATASET_IDX))


    app.run(host="localhost", port=args.port, use_reloader=False, threaded=False)
