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

"""Custom datasets for Multimodal training"""
import os
import re
import copy
import json
import gc
import copy
from dataclasses import dataclass
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from einops import rearrange
from typing import Any, Dict, List, Tuple, Sequence
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.multimodal.data.neva.neva_dataset import TarOrFolderImageLoader

MAX_NUM_IMAGES = 1
MIN_NUM_IMAGES = 1
IMAGE_PADDING_VAL = -100

def _preprocess_media_tokens(conversation: str, 
                            image_token: str = "[IMG]", 
                            image_patch_token: str = "[IMG]", 
                            image_break_token: str = "[IMG_BREAK]", 
                            image_end_token: str = "[IMG_END]", 
                            patch_size: int = 16, 
                            image_sizes: list = None,
):
    if image_sizes is None:
        return conversation
    
    # Calculate the number of pathes
    replace_strings = []
    for image_size in image_sizes:
        height, width = image_size
        num_height_tokens = height // patch_size
        num_width_tokens = width // patch_size
        replace_tokens = [
                    [image_patch_token] * num_width_tokens + [image_break_token]
                ] * num_height_tokens
        # Flatten list
        replace_tokens = [item for sublist in replace_tokens for item in sublist]
        replace_tokens[-1] = image_end_token
        replace_str = "".join(replace_tokens)
        replace_strings.append(replace_str)
        conversation = conversation.replace(image_token, "<placeholder>", 1)
    
    while "<placeholder>" in conversation:
        replace_str = replace_strings.pop(0)
        conversation = conversation.replace("<placeholder>", replace_str, 1)

    return conversation

def maybe_process_prompt_and_media(
        record,
        image_loader, 
        image_processor, 
        image_token, 
        image_patch_token, 
        image_break_token, 
        image_end_token, 
        image_patch_dim, 
        is_multimodal: bool = True,
    ):
    if "images" in record:
        if not isinstance(record['images'], list):
            record['images'] = [record['images']]
        images = []
        image_list = []
        for image_file in record["images"]:
            image = image_loader.open_image(image_file)
            if image is None:
                logging.warning(f"Image {image} could not be found!")
            images.append(image)

        if images:
            image_inputs = image_processor(images, return_tensors="pt")
            
            image_list, image_sizes = image_inputs["pixel_values"], image_inputs["image_sizes"]
            if isinstance(image_list[0], list): # List of List of Images            
                image_list = image_list[0]

            if isinstance(image_sizes[0], list):
                image_sizes = image_sizes[0]
                
            record['prompt'] = _preprocess_media_tokens(
                conversation=record['prompt'],
                image_token=image_token,
                image_patch_token=image_patch_token,
                image_break_token=image_break_token,
                image_end_token=image_end_token,
                patch_size=image_patch_dim,
                image_sizes=image_sizes
            )

        # image exist in the data
        if is_multimodal:
            # Image does not exist in the data, but the model is multimodal
            # TODO, if there are different videos on T dimensions.
            current_num_images = len(image_list)
            if current_num_images < MIN_NUM_IMAGES:
                image_list = []
                image_list.append(torch.zeros(3, image_patch_dim, image_patch_dim, dtype=torch.float))
        
        record["images"] = image_list
    return record

class MultimodalDPOModelDataset(Dataset):
    """This class works only with jsonl files. It assumes each line of the json file is a dictionary
       with the prompt, along with the chosen response (response only, no prompt), and the rejected response
       (response only, no prompt). This Dataset will combine the prompt with each corresponding chosen and 
       rejected response, and then tokenize it. It also returns the labels for each, which is the response tokens
       with -100 for the prompt part.
       
       WARNING: This class will tokenize the text, but it will raise an exception on model max seq len violations!
                Meaning it will not truncate tokens to fit to model max seq len, because of special prefix/suffix
                strings such as <extra_id_1>, it would not know where it is safe to truncate for each model. Therefore,
                the user must do all truncation logic in their preprocessing step when generating the jsonl
                used by this class. Put all special truncation logic there specific to your model.
    """

    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, image_processor, drop_last=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.data = data
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.eos_id = tokenizer.eos_id
        self.default_chosen_reward = cfg.data.get("default_chosen_reward", 1.0)
        self.default_rejected_reward = cfg.data.get("default_rejected_reward", 0.0)

        self.nograd_length = 32

        # Multimodal
        self.is_multimodal = cfg.data.get("is_multimodal", True)
        self.image_token = cfg.mm_cfg.get("image_token", "[IMG]")
        self.image_patch_token = cfg.mm_cfg.get("image_patch_token", "[IMG]")
        self.image_break_token = cfg.mm_cfg.get("image_break_token", "[IMG_BREAK]")
        self.image_end_token = cfg.mm_cfg.get("image_end_token", "[IMG_END]")
        self.image_folder = cfg.data.get("image_folder", None)        

        self.image_loader = TarOrFolderImageLoader(self.image_folder) if self.image_folder else None
        self.image_processor = image_processor
        self.image_patch_dim = cfg.mm_cfg.vision_encoder.patch_dim

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(self.data)

        # Replace image tags with
        img_pattern = r'<data-img\s+src=[\'"]([^\'"]+)[\'"](?:[^>]*)?/?>'
        self.data = []
        for record in data:
            # Search for <img src="/absolute/path/to/image" in the conversation
            #   add it as record['image'], remove src tag from the <img> tag
            record['images'] = []
            matches = re.finditer(img_pattern, record['prompt'])
            for match in matches:
                image_name = match.group(1).split("/")[-1]
                image_path = os.path.join(self.image_folder, image_name)
                if self.image_folder.endswith('.tar'):
                    if image_name not in self.image_loader.tar_index:
                        logging.warning(f"Image not found in tar: {image_name}")
                        continue
                else:
                    image_path = os.path.join(self.image_folder, image_name)
                    if not os.path.isfile(image_path):
                        logging.warning(f"Image not found: {image_path}")
                        continue
                record['images'].append(image_name)  # url
            record['prompt'] = re.sub(img_pattern, self.image_token, record['prompt'])
            self.data.append(record)

    def __len__(self):
        return len(self.data)

    def encode(self, text, append_eod=False):
        if self.cfg.data.get("apply_ftfy", False):
            import ftfy

            text = ftfy.fix_text(text)

        text_ids = self.tokenizer.text_to_ids(text)

        if len(text_ids) > 0 and append_eod:
            text_ids.append(self.tokenizer.eos_id)

        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """Returns a pair of chosen/rejected pairs, their respective lengths, and labels.
        """
        payload = copy.deepcopy(self.data[idx])
        payload = maybe_process_prompt_and_media(
                    payload,
                    self.image_loader,
                    self.image_processor,
                    self.image_token,
                    self.image_patch_token,
                    self.image_break_token,
                    self.image_end_token,
                    self.image_patch_dim,
                    self.is_multimodal,
                )
        
        prompt, prompt_len = self.encode(payload["prompt"], append_eod=False)
        chosen, chosen_len = self.encode(
            payload["prompt"] + payload["chosen_response"], append_eod=self.cfg.data.get("append_eod", False)
        )
        reject, reject_len = self.encode(
            payload["prompt"] + payload["rejected_response"], append_eod=self.cfg.data.get("append_eod", False)
        )
        # chosen_response_only, chosen_response_len = self.encode(payload['chosen_response'])
        # reject_response_only, reject_response_len = self.encode(payload['rejected_response'])
        chosen_labels = ([-100] * prompt_len) + chosen[prompt_len:]
        reject_labels = ([-100] * prompt_len) + reject[prompt_len:]

        assert chosen[0:prompt_len] == prompt, "the tokenizer for DPO has merged tokens between prompt and response"
        assert reject[0:prompt_len] == prompt, "the tokenizer for DPO has merged tokens between prompt and response"

        max_curr_seq_len = max(chosen_len, reject_len)
        if max_curr_seq_len > self.seq_length:
            logging.warning(
                f"WARNING: Tokenized text exceeds max seq length ({max_curr_seq_len} vs {self.seq_length})."
                + f"The example will be ignored."
            )

        chosen_tokens = torch.nn.functional.pad(
            torch.LongTensor(chosen), (0, max_curr_seq_len - chosen_len), mode="constant", value=self.eos_id
        )
        rejected_tokens = torch.nn.functional.pad(
            torch.LongTensor(reject), (0, max_curr_seq_len - reject_len), mode="constant", value=self.eos_id
        )
        labels_chosen_tokens = torch.nn.functional.pad(
            torch.LongTensor(chosen_labels), (0, max_curr_seq_len - len(chosen_labels)), mode="constant", value=-100
        )
        labels_reject_tokens = torch.nn.functional.pad(
            torch.LongTensor(reject_labels), (0, max_curr_seq_len - len(reject_labels)), mode="constant", value=-100
        )

        # ignore the example whose tokenized text exceeds max seq length.
        if max_curr_seq_len > self.seq_length:
            chosen_tokens = chosen_tokens[: self.nograd_length]
            rejected_tokens = rejected_tokens[: self.nograd_length]
            labels_chosen_tokens = torch.ones_like(chosen_tokens) * (-100)
            labels_reject_tokens = torch.ones_like(rejected_tokens) * (-100)
            chosen_len = self.nograd_length
            reject_len = self.nograd_length

        output = {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "chosen_length": chosen_len,
            "rejected_length": reject_len,
            "chosen_labels": labels_chosen_tokens,
            "rejected_labels": labels_reject_tokens,
            "chosen_reward": payload.get("chosen_reward", self.default_chosen_reward),
            "rejected_reward": payload.get("rejected_reward", self.default_rejected_reward),
            "media": payload.get("images", None),
        }
        return output
