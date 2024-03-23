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

from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon


def build_train_valid_datasets(
    data_cfg, consumed_samples,
):
    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict["captions"] = input
            yield out_dict

    def transform_fn(sample):
        text = sample["txt"]
        return text

    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=transform_fn,
        compose_fn=tuple_to_dict,
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("dataset_path"):
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            is_train=False,
        )

    return train_data, val_data
