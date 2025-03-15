# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from functools import partial

import torch

from nemo.utils import logging
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo_aligner.data.nlp.builders import build_train_valid_test_datasets
from nemo_aligner.experimental.grpo.data.datasets import AllTaskDataset, environment_collate_with_batch_max_sequence_length
from nemo_aligner.experimental.grpo.utils import parallel_state

build_train_valid_test_task_datasets = partial(build_train_valid_test_datasets, AllTaskDataset)

def environment_collate_with_pad_to_max_batch(max_seqlen, tokenizer_eos_id, cfg, generate_masks_and_position_ids=True):
    """collate function that pads each sequence to the max in the batch"""
    return partial(
        environment_collate_with_batch_max_sequence_length,
        response_token_length=max_seqlen,
        eos_id=tokenizer_eos_id,
        reset_position_ids=cfg.model.data.get("reset_position_ids", False),
        reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
        eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        generate_masks_and_position_ids=generate_masks_and_position_ids,
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
    use_random_sampler=True,
):
    """Buld dataloader given an input dataset."""

    logging.info(
        f"Building dataloader with consumed samples: {consumed_samples}, "
        f"a DP rank of {parallel_state.get_data_parallel_rank()}, "
        f"and a DP world size of {parallel_state.get_data_parallel_world_size()}"
    )
 
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": gbs,
        "data_parallel_rank": 0,
        "data_parallel_size": 1,
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }
    
    if use_random_sampler:
        cls = MegatronPretrainingRandomBatchSampler if load_gbs else MegatronPretrainingRandomSampler
        common_params["seed"] = cfg.model.seed
    else:
        cls = MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
