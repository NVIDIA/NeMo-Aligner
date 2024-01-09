# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Generic file to build datasets
modified from: https://github.com/NVIDIA/NeMo/blob/2baef811f21372c3340dd2d82635d2377e78a660/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
to allow us to build SFT, RewardModel and RLHF datasets
"""

import json
from functools import partial

import numpy as np
import torch
from megatron.core import parallel_state
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.utils import logging
from nemo_aligner.data.nlp.datasets import (
    DPOModelDataset,
    RegressionRewardModelDataset,
    RewardModelDataset,
    RLHFDataset,
)
from nemo_aligner.utils.utils import collate_with_batch_max_sequence_length


def build_dataset_generic(cls, cfg, data_prefix, data_impl, seq_length, seed, tokenizer, name):
    if isinstance(data_prefix, (ListConfig, list)):
        data_prefix = data_prefix[0]

    assert isinstance(data_prefix, str), "data.data_prefix values must be either a list or a string"

    if data_impl.startswith("json"):
        with open(data_prefix, "r", encoding="utf_8") as fr:
            data_payload = [json.loads(line.strip()) for line in fr]
    else:
        raise RuntimeError(f"data.data_impl must be either json or jsonl, but got {data_impl}")
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
        data_prefix=data_prefix,
        data=data_payload,
        seq_length=seq_length,
        seed=seed,
        drop_last=drop_last,
    )

    return dataset


def build_train_valid_test_datasets(cls, cfg, data_prefix, data_impl, seq_length, seed, tokenizer):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        train_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["train"],
            data_impl=data_impl,
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="train",
        )
        validation_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["validation"],
            data_impl=data_impl,
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="validation",
        )
        test_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["test"],
            data_impl=data_impl,
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="test",
        )
        return train_ds, validation_ds, test_ds

    else:
        raise TypeError("data.data_prefix needs to be a dict with keys equal to 'train', 'validation', and 'test'")


build_train_valid_test_rlhf_datasets = partial(build_train_valid_test_datasets, RLHFDataset)
build_train_valid_test_rm_datasets = partial(build_train_valid_test_datasets, RewardModelDataset)
build_train_valid_test_dpo_datasets = partial(build_train_valid_test_datasets, DPOModelDataset)
build_train_valid_test_regression_rm_datasets = partial(build_train_valid_test_datasets, RegressionRewardModelDataset)


def build_sft_dataset(data_cfg, tokenizer, num_samples, answer_only_loss=True, is_chat=True, special_tokens=None):
    dataset_cls = GPTSFTChatDataset if is_chat else GPTSFTDataset
    dataset = dataset_cls(
        file_path=data_cfg.file_path,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        add_bos=data_cfg.get("add_bos", False),
        add_eos=data_cfg.get("add_eos", True),
        add_sep=data_cfg.get("add_sep", False),
        sep_id=0,
        max_num_samples=num_samples,
        seed=data_cfg.get("seed", 1234),
        label_key=data_cfg.get("label_key", "answer"),
        answer_only_loss=answer_only_loss,
        truncation_field=data_cfg.get("truncation_field", "text"),
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        index_mapping_dir=data_cfg.get("index_mapping_dir", None),
        prompt_template=data_cfg.get("prompt_template", None),
        virtual_tokens=0,
        tokens_to_generate=data_cfg.get(
            "tokens_to_generate", 0
        ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
        memmap_workers=data_cfg.get(
            "memmap_workers", None
        ),  # used to set num. of workers to create the memmap index files
        hf_dataset=data_cfg.get(
            "hf_dataset", False
        ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        truncation_method=data_cfg.get(
            "truncation_method", "right"
        ),  # used to choose truncation method. Options: ['random', 'left', 'right']
        special_tokens=special_tokens,
    )
    return dataset


def collate_with_pad_to_max_batch(max_seqlen, tokenizer_eos_id, cfg):
    """collate function that pads each sequence to the max in the batch
    """
    return partial(
        collate_with_batch_max_sequence_length,
        response_token_length=max_seqlen,
        eos_id=tokenizer_eos_id,
        reset_position_ids=cfg.model.data.get("reset_position_ids", False),
        reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
        eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
    )


def build_dataloader(
    cfg,
    dataset,
    consumed_samples,
    mbs,
    gbs,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
    collate_fn=None,
    load_gbs=True,
):
    """Buld dataloader given an input dataset."""

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
    # Megatron sampler
    if hasattr(cfg.model.data, "dataloader_type") and cfg.model.data.dataloader_type is not None:
        if cfg.model.data.dataloader_type == "single":
            cls = MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
            batch_sampler = cls(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=mbs,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
                drop_last=drop_last,
                global_batch_size=gbs,
                pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
            )
        else:
            raise ValueError('cfg.data.dataloader_type must be "single"')
    else:
        raise ValueError('cfg.data.dataloader_type not found. Must be "single"')

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
