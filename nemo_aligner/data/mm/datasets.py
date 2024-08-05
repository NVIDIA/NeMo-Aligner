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
import copy
import numpy as np
import torch
from typing import Any, Dict, List
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import _create_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.multimodal.data.neva.neva_dataset import NevaDataset, process_image
from PIL import Image
from transformers import CLIPImageProcessor, SiglipImageProcessor
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import TiledSiglipImageProcessor
from nemo.collections.multimodal.parts.utils import create_image_processor

MAX_NUM_IMAGES = 1

class MultimodalChatDataset(NevaDataset):
    def __init__(
            self, 
            data_cfg, 
            mm_cfg, 
            tokenizer, 
            media_type="image",
            image_folder=None,
            video_folder=None,
            image_aspect_ratio="square",
            image_token_len=256,
            num_frames=-1,
            add_extra_token=1,
            ignore_index=-1,
            splice_single_frame=None,
            sep_token_between_frames=False,
            special_tokens=None,
    ):
        
        super().__init__(
            tokenizer=tokenizer, 
            data_path=data_cfg.file_path, 
            multimodal_cfg=dict(
                is_multimodal=True,
                sep_image_conv_front=False,
                conv_template=None,
                crop_size=mm_cfg.vision_encoder.crop_size,
                image_token_len=image_token_len,
                image_folder=image_folder,
                video_folder=video_folder,
                image_aspect_ratio=image_aspect_ratio,
                use_im_start_end=mm_cfg.get("use_im_start_end", False),
                patch_dim=mm_cfg.vision_encoder.patch_dim,
                mm_mlp_adapter_type=mm_cfg.get("mm_mlp_adapter_type", "linear"),
                image_processor=create_image_processor(mm_cfg),
                add_extra_token=add_extra_token,
                context_length=data_cfg.max_seq_length,
                media_type=media_type,
                num_frames=num_frames,
            ),
            data_cfg=dict(
                splice_single_frame=splice_single_frame,
                num_frames=num_frames,
                sep_token_between_frames=sep_token_between_frames,
            )
        )
        
        self.data_cfg = data_cfg
        self.mm_cfg = mm_cfg
        self.tokenizer = tokenizer

        self.max_seq_length = data_cfg.max_seq_length
        self.add_extra_token = add_extra_token
        self.ignore_index = ignore_index

        self.image_token = mm_cfg.get("image_token", "<image>")        
        self.video_token = mm_cfg.get("video_token", "<video>")
        self.image_patch_token = mm_cfg.get("image_patch_token", "<extra_id_3>") 
        self.im_start_token = mm_cfg.get("im_start_token", "<extra_id_4>")
        self.im_end_token = mm_cfg.get("im_end_token", "<extra_id_5>")
        
        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens
        
    def get_prompt(self, system_token, system_message, messages) -> str:
        prompt_dict = []

        prompt = f"{self.special_tokens['system_turn_start']}{system_token}{self.special_tokens['end_of_name']}"        
        prompt += f"{system_message}"
        
        prompt_dict.append({"system": prompt})

        for turn in messages:
            speaker = f"{self.special_tokens['end_of_turn']}{self.special_tokens['turn_start']}{turn['from']}{self.special_tokens['end_of_name']}"
            message = f"{turn['value']}"
            prompt += speaker + message
            prompt_dict.append(
                {
                    "speaker": speaker,
                    "message": message,
                    "mask": turn['mask']
                }
            )
        return prompt, prompt_dict
    
    def _maybe_process_tokens(
        self,
        tokens_list: List[int],
        labels_list: List[int],
        context_length: int = None,
        add_extra_token: int = 1,
    ) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s). If the list of tokens exceeds the context
        length plus the number of extra tokens, it gets truncated. If it's smaller, it gets padded with zeros.

        Parameters
        ----------
        tokens : List[int]
            Conversation tokens for all turns and the system prompt.
        labels : List[int]
            Labels for all turns.        
        context_length : int
            The context length to be used for the output tensor.
        add_extra_token : int
            Number of extra tokens to add, should be either 0 or 1.
        Returns
        -------
        torch.LongTensor
            A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length + add_extra_token].
        """
        assert add_extra_token == 0 or add_extra_token == 1, "`add_extra_token` should be either 0 or 1."

        max_len = len(tokens_list)
        if context_length is not None:
            context_length = min(max_len - add_extra_token, context_length)
        else:
            context_length = max_len
        # truncate and padding
        tokens = torch.zeros(1, context_length + add_extra_token, dtype=torch.long)
        labels = torch.zeros(1, context_length + add_extra_token, dtype=torch.long)

        if max_len > context_length + add_extra_token:
            tokens_list = tokens_list[: context_length + add_extra_token]  # Truncate
            labels_list = labels_list[: context_length + add_extra_token]  # Truncate

        tokens[0, :len(tokens_list)] = torch.tensor(tokens_list)
        labels[0, :len(labels_list)] = torch.tensor(labels_list)
        return tokens, labels
        
    def preprocess_conversations(self, sources: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a given set of conversational sources using a flexible conversation template.

        Parameters:
        ----------
        sources: List[str]
            A list of dictionaries containing conversational data.

        Returns:
        ----------
        Dict: A dictionary containing 'tokens' and 'labels' tensors for model input.
        """
        add_extra_token = self.add_extra_token
        context_length = self.max_seq_length

        # Apply the prompt template
        for source in sources:
            messages = []
            system_message = source.get('system', "")
            system_token = source.get('system_token', "System")
            mask_role = source.get('mask', "User")
            for i, turn in enumerate(source['conversations']):
                if i % 2 == 1:
                    mask_turn = True if turn['from'] == mask_role else False
                    messages.append({"from": turn['from'], "value": turn['value'], "mask": mask_turn})
                else:
                    mask_turn = True if turn['from'] == mask_role else False
                    messages.append({"from": turn['from'], "value": turn['value'], "mask": mask_turn})
            _, context_dict = self.get_prompt(system_token, system_message, messages)

        # Mask targets
        tokens_list = []
        labels_list = []
        system_tokens = self.tokenizer.text_to_ids(context_dict[0]['system'])    
        tokens_list.extend(system_tokens)
        labels_list.extend([self.ignore_index for _ in range(len(system_tokens))])
        for turn in context_dict[1:]:
            speaker_tokens = self.tokenizer.text_to_ids(turn['speaker'])
            message_tokens = self.tokenizer.text_to_ids(turn['message'])        

            turn_tokens = speaker_tokens + message_tokens
            tokens_list.extend(turn_tokens)
            if turn['mask']: # mask everything
                labels_list.extend([self.ignore_index for _ in range(len(turn_tokens))])
            else:  # mask speaker tokens
                labels_list.extend([self.ignore_index for _ in range(len(speaker_tokens))] + message_tokens)

        tokens, labels = self._maybe_process_tokens(tokens_list, labels_list, context_length, add_extra_token)

        # Check if masking works correctly
        #print([x for x in zip(self.tokenizer.ids_to_tokens(tokens[0].numpy().tolist()), tokens[0].numpy().tolist(), labels[0].numpy().tolist())])

        if self.add_extra_token:
            tokens = tokens[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
        else:
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels[:, -1] = self.ignore_index
        return dict(
            tokens=tokens,
            labels=labels,
        )

    def preprocess_media_tokens(self, sources: dict, cur_token_len: int, use_plain: bool = False) -> Dict:
        """
        Preprocesses multimodal sources based on the provided configuration.

        This function modifies the sources for multimodal data processing. It checks if the data is multimodal and
        adjusts the token lengths accordingly. It also handles the start and end tokens for images and replaces
        image tokens in conversations.

        Parameters:
        - sources (dict): A dictionary containing the multimodal sources to be processed.
        - multimodal_cfg (dict): A configuration dictionary specifying various options for multimodal processing.
          It includes keys like 'is_multimodal', 'use_im_start_end', and 'sep_image_conv_front'.
        - cur_token_len (int): The current length of tokens to be considered for image processing.
        - use_plain (bool, optional): A boolean flag to use plain image token replacement without additional processing.
          Defaults to False.

        Returns:
        - dict: The processed sources dictionary after applying multimodal preprocessing steps.
        """
        multimodal_cfg = self.multimodal_cfg
        is_multimodal = multimodal_cfg['is_multimodal']
        media_type = multimodal_cfg['media_type']
        image_token_len = cur_token_len
        if media_type == 'image':
            default_token = self.image_token
        elif media_type == 'video':
            default_token = self.video_token
        else:
            return sources

        if not is_multimodal:
            return sources

        num_patches = image_token_len
        if media_type == 'video':
            num_patches *= multimodal_cfg['num_frames']

        if multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
            num_patches //= 4

        if multimodal_cfg['use_im_start_end']:
            replace_token = self.image_patch_token * num_patches
        else:
            replace_token = self.image_patch_token * (num_patches - 2)

        replace_token = self.im_start_token + replace_token + self.im_end_token

        for source in sources:
            conversation = source['conversations']
            if use_plain:
                assert default_token in conversation[0]['value']
                conversation[0]['value'] = default_token
            for turn in conversation:
                turn["value"] = turn["value"].replace(default_token, replace_token)

        return sources

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            if not isinstance(self.list_data_dict[i]['image'], list):
                self.list_data_dict[i]['image'] = [self.list_data_dict[i]['image']]

            images = []
            for image_file in self.list_data_dict[i]['image']:
                image = self.image_loader.open_image(image_file)
                if image is None:
                    logging.warning(f"Image {image_file} could not be found!")
                image = process_image(self.processor, image, self.multimodal_cfg['image_aspect_ratio'])
                images.append(image)
            media_tensors = torch.tensor([])
            if images:
                media_tensors = torch.stack(images)
                patch_dim = self.multimodal_cfg['patch_dim']

                height_num_patches = media_tensors[0].shape[1] // patch_dim
                width_num_patches = media_tensors[0].shape[2] // patch_dim

                if isinstance(self.processor, TiledSiglipImageProcessor):
                    height_num_patches = height_num_patches // self.processor.grid_height
                    width_num_patches = width_num_patches // self.processor.grid_width
                elif self.multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
                    if height_num_patches % 2 != 0:
                        height_num_patches += 1
                    if width_num_patches % 2 != 0:
                        width_num_patches += 1

                cur_token_len = height_num_patches * width_num_patches

                sources = self.preprocess_media_tokens(
                    copy.deepcopy(sources),
                    cur_token_len,
                    use_plain=(self.conv_template == "plain"),
                )
        elif 'video' in sources[0]:
            raise NotImplementedError("Only image modality is supported for now.")
        else:
            logging.warning("media not found in sources")
            media_tensors = torch.tensor([])
            sources = copy.deepcopy(sources)

        data_dict = self.preprocess_conversations(sources)

        if isinstance(i, int):
            data_dict = dict(tokens=data_dict["tokens"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if self.multimodal_cfg['is_multimodal']:
            if isinstance(self.processor, CLIPImageProcessor):
                crop_size = [self.processor.crop_size['height'], self.processor.crop_size['width']]
            else:
                crop_size = copy.deepcopy(self.multimodal_cfg['crop_size'])
                if isinstance(self.processor, TiledSiglipImageProcessor):
                    crop_size[0] *= self.processor.grid_height
                    crop_size[1] *= self.processor.grid_width
                    
            # Image does not exist in the data, but the model is multimodal
            # TODO, if there are different videos on T dimensions.
            if media_tensors.shape[0] < MAX_NUM_IMAGES:
                padding_size = MAX_NUM_IMAGES - media_tensors.shape[0]
                zero_padding = torch.zeros((padding_size, 3, crop_size[0], crop_size[1]), dtype=torch.float)
                media_tensors = torch.cat((media_tensors, zero_padding), dim=0)

            if self.multimodal_cfg['media_type'] == 'image':
                data_dict['image'] = media_tensors
            elif self.multimodal_cfg['media_type'] == 'video':
                data_dict['video'] = media_tensors

        return data_dict
    