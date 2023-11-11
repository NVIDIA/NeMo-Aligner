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

import torch
from omegaconf import ListConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.utils import logging

from .datasets import OfflineDataset

try:
    from megatron.core import mpu

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def build_data_loader(dataset, data_cfg, consumed_samples=0, sample_split_size=None):
    """Build dataloader given an input dataset."""

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")

    sample_split_size = data_cfg.get("sample_split_size", None)
    sample_split_iter = data_cfg.get("sample_split_iter", 0)
    if sample_split_size is not None:
        # current samples offset
        sample_split_offset = sample_split_iter * sample_split_size
        total_samples = sample_split_offset + sample_split_size
        logging.info(f"DataLoader sample split: {sample_split_offset}:{total_samples}")
        consumed_samples += sample_split_offset
    else:
        total_samples = len(dataset)

    # corner case for empty dataset
    if consumed_samples >= total_samples:
        dataset = []
        batch_sampler = None
        collate_fn = None
    else:
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.micro_batch_size * mpu.get_data_parallel_world_size(),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            drop_last=True,
            pad_samples_to_global_batch_size=False,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )


def build_dataset(data_cfg, tokenizer, preprocess_callback=None):
    datasets = []
    is_list_config = isinstance(data_cfg.file_names, ListConfig)
    if not is_list_config:
        raise ValueError(f"train/validation datasets must be provided as a list of individual JSONL files.")
    # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
    # that is of the format [weight1,file_name1,weight2,file_name2,...]
    if data_cfg.concat_sampling_probabilities is None or not isinstance(
        data_cfg.concat_sampling_probabilities, ListConfig
    ):
        raise ValueError(
            (
                f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                f"Found: {data_cfg.concat_sampling_probabilities}"
            )
        )
    if len(data_cfg.get("concat_sampling_probabilities", None)) != len(data_cfg.file_names):
        raise ValueError(
            (
                f"concat_sampling_probabilities must be of the same size as file_names.",
                f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
            )
        )

    data_prefix = []
    for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
        data_prefix.append(weight)
        data_prefix.append(prefix)

    num_samples = data_cfg.num_samples
    if data_cfg.num_samples is not None:
        _, _, num_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        num_samples_after_blend = sum([x for x in num_samples_per_dataset])
    else:
        num_samples_per_dataset = [None] * len(data_cfg.file_names)
        num_samples_after_blend = 0

    for file_path, num_samples in zip(data_cfg.file_names, num_samples_per_dataset):
        dataset = OfflineDataset(
            data_cfg,
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=data_cfg.max_seq_length,
            min_seq_length=data_cfg.min_seq_length,
            preprocess_callback=preprocess_callback,
            tokens_to_generate=data_cfg.get("tokens_to_generate", 0),
            add_bos=data_cfg.get("add_bos", False),
            add_eos=data_cfg.get("add_eos", False),
            max_num_samples=num_samples,
            seed=data_cfg.get("seed", 1234),
            input_key=data_cfg.get("input_key", "input"),
            hf_dataset=data_cfg.get("hf_dataset", True),
            index_mapping_dir=data_cfg.get("index_mapping_dir", None),
        )
        datasets.append(dataset)

        if data_cfg.num_samples is None:
            num_samples_after_blend += len(dataset)

    dataset = BlendableDataset(
        datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_samples_after_blend
    )
    return dataset
