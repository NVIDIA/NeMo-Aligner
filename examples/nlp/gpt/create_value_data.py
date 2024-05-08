import json
import pathlib
import sys

import fire
import numpy as np
import sentencepiece as spm
import torch
import tqdm


def main(
    data_path,
    type="value",
    tokenizer_path="/datasets/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model",
):
    tokenizer = spm.SentencePieceProcessor(tokenizer_path)
    path = pathlib.Path(data_path)
    if type == "value":
        files = path.glob("value_data_*.pt")
    elif type == "policy":
        files = path.glob("policy_data_*.pt")
    else:
        raise ValueError("type should be either value or policy")

    with open(f"{data_path}/dpo_data.json", "w", encoding="utf-8") as f:
        progress = tqdm.tqdm(files)

        for file in progress:
            progress.set_description(f"Processing {file.name}")
            data = torch.load(file)

            all_data = []
            for reward, tokens in zip(data["reward"], data["tokens"]):
                full_text = tokenizer.decode(tokens)
                context_len = data["context_length"]

                response = tokenizer.decode(tokens[context_len:])
                prompt = tokenizer.decode(tokens[:context_len])
                user_prompt = prompt.split("<extra_id_1>")[1][5:].strip()
                index = response.find("<extra_id")
                if index != -1:
                    response = response[:index].strip()
                data_record = {
                    "conversations": [
                        {"from": "User", "value": user_prompt},
                        {"from": "Assistant", "value": response},
                    ],
                    "data_id": data["data_id"],
                    "system": "",
                    "mask": "User",
                    "reward": reward,
                }
                all_data.append(data_record)
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
