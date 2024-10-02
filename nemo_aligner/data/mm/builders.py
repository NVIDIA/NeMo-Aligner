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
import json
from functools import partial

import numpy as np
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_
from nemo.utils import logging
from nemo_aligner.data.mm.datasets import MultimodalChatDataset, MultimodalDPOModelDataset

def build_mm_dataset_generic(cls, cfg, data_prefix, data_impl, num_samples, seq_length, seed, tokenizer, name, image_processor):
    def _build_dataset(current_data_prefix, current_num_samples):
        if data_impl == "mmap":
            data_payload = get_indexed_dataset_(current_data_prefix, data_impl, cfg.data.get("skip_warmup", True))
        elif data_impl.startswith("json"):
            with open(current_data_prefix, "r", encoding="utf_8") as fr:
                data_payload = [json.loads(line.strip()) for line in fr]
        else:
            raise RuntimeError(f"data.data_impl must be either mmap or json or jsonl, but got {data_impl}")
        total_num_of_documents = len(data_payload)

        # Print stats about the splits.
        logging.info(" > dataset split:")
        logging.info("     Total {} documents is : {} ".format(name, total_num_of_documents))

        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)

        dataset = cls(
            cfg=cfg,
            tokenizer=tokenizer,
            name=name,
            data_prefix=current_data_prefix,
            documents=np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            data=data_payload,
            seq_length=seq_length,
            seed=seed,
            image_processor=image_processor,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)
    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        data_prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(data_prefixes)):
            dataset = _build_dataset(data_prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples)

def build_mm_train_valid_test_datasets(
    cls, cfg, data_prefix, data_impl, splits_string, train_valid_test_num_samples, seq_length, seed, tokenizer, image_processor,
):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data path is of type dictionary.")
        train_ds = build_mm_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["train"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[0]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="train",
            image_processor=image_processor,
        )
        validation_ds = build_mm_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["validation"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[0]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="validation",
            image_processor=image_processor,
        )
        test_ds = build_mm_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["test"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[0]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="test",
            image_processor=image_processor,
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_mm_train_valid_test_datasets(
                cls=cls,
                cfg=cfg,
                data_prefix=data_prefix[0],
                data_impl=data_impl,
                splits_string=splits_string,
                train_valid_test_num_samples=train_valid_test_num_samples,
                seq_length=seq_length,
                seed=seed,
                tokenizer=tokenizer,
                image_processor=image_processor,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        data_prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(data_prefixes)):
            train_ds, valid_ds, test_ds = _build_mm_train_valid_test_datasets(
                cls=cls,
                cfg=cfg,
                data_prefix=data_prefixes[i],
                data_impl=data_impl,
                splits_string=splits_string,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seq_length=seq_length,
                seed=seed,
                tokenizer=tokenizer,
                image_processor=image_processor,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_mm_train_valid_test_datasets(
    cls, cfg, data_prefix, data_impl, splits_string, train_valid_test_num_samples, seq_length, seed, tokenizer, image_processor,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset or jsonl
    if data_impl == "mmap":
        data_payload = get_indexed_dataset_(data_prefix, data_impl, cfg.data.get("skip_warmup", True))
    elif data_impl.startswith("json"):
        with open(data_prefix, "r", encoding="utf_8") as fr:
            data_payload = [json.loads(line.strip()) for line in fr]
    else:
        raise RuntimeError(f"data.data_impl must be either mmap or json or jsonl, but got {data_impl}")
    total_num_of_documents = len(data_payload)
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(" > dataset split:")

    def print_split_stats(name, index):
        logging.info("    {}:".format(name))
        logging.info(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "validation":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = cls(
                cfg=cfg,
                tokenizer=tokenizer,
                name=name,
                data_prefix=data_prefix,
                documents=documents,
                data=data_payload,
                num_samples=train_valid_test_num_samples[index],
                seq_length=seq_length,
                seed=seed,
                image_processor=image_processor,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "validation")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)

build_train_valid_test_dpo_datasets = partial(build_mm_train_valid_test_datasets, MultimodalDPOModelDataset)

def build_mm_sft_dataset(model_cfg, data_cfg, mm_cfg, tokenizer, image_processor, special_tokens=None):
    dataset = MultimodalChatDataset(
        data_cfg=data_cfg,
        mm_cfg=mm_cfg,
        tokenizer=tokenizer,
        image_processor=image_processor,
        media_type=model_cfg.data.get("media_type","image"),
        image_folder=model_cfg.data.get("image_folder", None),
        video_folder=model_cfg.data.get("video_folder", None),
        image_aspect_ratio=model_cfg.data.get("image_aspect_ratio", "square"),
        image_token_len=model_cfg.data.get("image_token_len", 256),
        num_frames=model_cfg.data.get("num_frames", -1),
        add_extra_token=model_cfg.data.get("add_extra_token", 1),
        ignore_index=model_cfg.data.get("ignore_index", -1),
        splice_single_frame=model_cfg.data.get("splice_single_frame", None),
        sep_token_between_frames=model_cfg.data.get("sep_token_between_frames", False),
        add_speakers=model_cfg.data.get("add_speakers", True),
        special_tokens=special_tokens,
    )
    return dataset