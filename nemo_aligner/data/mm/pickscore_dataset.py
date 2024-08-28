# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
from glob import glob
from typing import Any

import numpy as np
import torch

# load files
from datasets import Dataset as Dataset_hf
from datasets import concatenate_datasets
from PIL import Image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, Normalize, ToTensor

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.utils import logging

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_valid_datasets(
    model_cfg, consumed_samples, tokenizer=None, seed=None, return_test_data=False,
):
    train_data = PickScoreDataset(
        model_cfg, tokenizer=tokenizer, consumed_samples=consumed_samples, split="train", seed=seed,
    )

    val_data = PickScoreDataset(
        model_cfg, tokenizer=tokenizer, consumed_samples=consumed_samples, split="val", seed=seed,
    )

    if return_test_data:
        test_data = PickScoreDataset(
            model_cfg, tokenizer=tokenizer, consumed_samples=consumed_samples, split="test", seed=seed,
        )
        return train_data, val_data, test_data

    # TODO: Add test data
    return train_data, val_data


class PickScoreDataset(Dataset):
    def __init__(
        self,
        model_cfg,
        tokenizer=None,
        stop_idx=None,
        consumed_samples=0,
        path=None,
        seed: int = 42,
        split: str = "train",
    ):
        super().__init__()
        self.model_cfg = model_cfg
        # check path
        self.tokenizer = tokenizer
        assert split in ("train", "val", "test")
        # self.split = split
        self.split_path = {"train": "train", "val": "validation_unique", "test": "test_unique"}[split]
        self.path = path or model_cfg.data.get("data_path")
        self.filter_ties = model_cfg.data.get(split, {}).get("filter_ties", False)  # do we want to filter ties
        # lazy load all datasets
        from os import path as osp

        datasets = sorted(list(glob(osp.join(self.path, self.split_path, "*.arrow"))))
        datasets = [Dataset_hf.from_file(x) for x in datasets]
        datasets = concatenate_datasets(datasets)
        num_rows = datasets.num_rows
        # set shuffled indices to the list and then filter if asked to
        self.shuffled_indices = np.arange(num_rows).astype(np.int32)
        if self.filter_ties:
            logging.info(f"Filtering ties from {split} split.")
            det_label = np.logical_or(np.array(datasets["label_0"]) == 1, np.array(datasets["label_1"]) == 1)
            det_label = np.where(det_label)[0]
            self.shuffled_indices = det_label

        self.df = datasets
        print(f"*********** Loading {split} dataset containing {len(self.shuffled_indices)} entries ***********")
        # shuffle the indices if given
        if seed is not None:
            np_rng = np.random.RandomState(seed=seed)
            np_rng.shuffle(self.shuffled_indices)

    def __len__(self) -> int:
        return len(self.shuffled_indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        true_idx = int(self.shuffled_indices[i])
        df = self.df[true_idx]
        img_0 = Image.open(io.BytesIO(df["jpg_0"])).convert("RGB")
        img_1 = Image.open(io.BytesIO(df["jpg_1"])).convert("RGB")
        label = torch.FloatTensor([df["label_0"], df["label_1"]])  # preference label
        text = df["caption"]

        img_0, img_1 = torch.FloatTensor(np.array(img_0)), torch.FloatTensor(np.array(img_1))

        output = {"img_0": img_0, "img_1": img_1, "label": label, "prompt": text, "time_step": torch.tensor([0.0])}
        return output


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    cfg = {
        "vision": {"img_w": 224, "img_h": 224, "img_mean": OPENAI_DATASET_MEAN, "img_std": OPENAI_DATASET_STD},
        "text": {"max_position_embeddings": 77,},
        "data": {"data_path": args.data_path,},
    }
    cfg = OmegaConf.create(cfg)
    dataset = PickScoreDataset(cfg, tokenizer=None, split="val")
    print(len(dataset))
    for i in range(10):
        batch = dataset[i]
        for k, v in batch.items():
            if k == "prompt":
                print(k, v)
            else:
                print(k, v.shape)
                if "img" in k:
                    print(v.max(), v.min())
        print()
