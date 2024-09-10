import time
from pprint import pprint

import numpy as np
import torch
from pytriton.client import ModelClient
from transformers import AutoTokenizer

all = torch.load("/rlhf/batched.pt")

host = "localhost"
port = 5555

model_name = "reward_model"

# inputs = {
#    "tokens": np.array([[1,2,3]]),
#   "sequence_lengths": np.array([[3]]),
# }
# for x in all:
#   inputs = {
#     "tokens": x['tokens'].numpy(),
#     "sequence_lengths": torch.arange(x['tokens'].size(1), dtype=torch.long)[-1].add(1).view(1, -1).numpy(),
#   }

#   with ModelClient(f"{host}:{port}", model_name) as client:
#       print("#### SEND INFER AT", time.time())
#       result_dict = client.infer_batch(**inputs)
#       print("#### got INFER AT", time.time())

#   print((result_dict['rewards'][:, :, 4] - x['pred'].numpy()).mean())
#   print({k:v.shape for k,v in result_dict.items()})


def get_value(tokens):
    inputs = {
        "tokens": tokens.numpy(),
        "sequence_lengths": torch.arange(tokens.size(1), dtype=torch.long)[-1].add(1).view(1, -1).numpy(),
    }

    with ModelClient(f"{host}:{port}", model_name) as client:
        result_dict = client.infer_batch(**inputs)

    return result_dict["rewards"]


def _str_list2numpy(str_list) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def get_value_text(text):
    inputs = {"sentences": _str_list2numpy(text)}

    with ModelClient(f"{host}:{port}", model_name) as client:
        result_dict = client.infer_batch(**inputs)

    return result_dict["rewards"]


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B", use_fast=True)

for batch in all:
    # values = get_value(batch['tokens'])
    # text = tokenizer.decode(batch['tokens'].tolist()[0])
    # values = get_value_text([text])
    # print((values[:, :batch['pred'].size(1), 4] - batch['pred'].numpy()).mean())
    # continue
    for i in range(len(batch["tokens"][0])):
        if not batch["mask"][0][i].item():
            continue
        sub_token = batch["tokens"][0][: i + 1]
        text = tokenizer.decode(sub_token, skip_special_tokens=True)
        values = get_value_text([text])[:, :, 4]
        pred = values[0][i]
        batch_pred = batch["pred"][0][i]
        batch_gt = batch["scores"][0][i]
        result = {
            "pos": i,
            "gt": batch_gt,
            "trn_pred": batch_pred,
            "infer_value": float(pred),
            "diff_against_train_pred": float(pred) - float(batch_pred),
            "diff_again_gt": float(pred) - float(batch_gt),
        }
        pprint(result)
