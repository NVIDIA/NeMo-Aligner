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

from nemo_aligner.data.mm.datasets import MultimodalChatDataset

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