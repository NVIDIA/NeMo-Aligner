import argparse
import random

from flask import Flask, request

app = Flask(__name__)

DATASET_IDX = None


@app.route("/get_idx", methods=["PUT"])
def get_http_idx():
    global DATASET_IDX

    batch_size = request.json["batch_size"]
    batches = []

    while len(DATASET_IDX) > 0 and len(batches) < batch_size:
        batches.append(DATASET_IDX.pop())

    print("AFTER", DATASET_IDX)

    return list(reversed(batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-length", type=int, required=True,
    )
    parser.add_argument(
        "--port", type=int, default=9999,
    )
    args = parser.parse_args()
    random.seed(98)

    idx_list = list(range(args.dataset_length))
    random.shuffle(idx_list)
    DATASET_IDX = idx_list
    print("### ALL DATASET IDX", DATASET_IDX)

    app.run(host="localhost", port=args.port, use_reloader=False, threaded=False)
