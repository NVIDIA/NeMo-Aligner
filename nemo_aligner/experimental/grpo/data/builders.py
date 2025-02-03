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

from nemo_aligner.data.nlp.builders import build_train_valid_test_datasets
from nemo_aligner.experimental.grpo.data.datasets import AllTaskDataset, environment_collate_with_batch_max_sequence_length

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

