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
import json
import torch
import torch.nn.functional as F
import re
from typing import Any, Dict, List, Tuple, Sequence
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import _create_ltor_masks_and_position_ids
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from torch.utils.data import Dataset, default_collate
from nemo.utils import logging
from nemo.collections.multimodal.data.neva.neva_dataset import TarOrFolderImageLoader
from PIL import Image
from transformers import CLIPImageProcessor, SiglipImageProcessor
from dataclasses import dataclass
from omegaconf import DictConfig
import transformers
from einops import rearrange
from nemo_aligner.utils.multimodal import TensorList

MIN_NUM_IMAGES = 1
IMAGE_PADDING_VAL = -100


    
def _preprocess_media_tokens(conversation: str, 
                            image_token: str = "[IMG]", 
                            image_patch_token: str = "[IMG]", 
                            im_start_token: str = "[IMG_BREAK]", 
                            im_end_token: str = "[IMG_END]", 
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
                    [image_patch_token] * num_width_tokens + [im_start_token]
                ] * num_height_tokens
        # Flatten list
        replace_tokens = [item for sublist in replace_tokens for item in sublist]
        replace_tokens[-1] = im_end_token
        replace_str = "".join(replace_tokens)
        replace_strings.append(replace_str)
        conversation = conversation.replace(image_token, "<placeholder>", 1)
    
    while "<placeholder>" in conversation:
        replace_str = replace_strings.pop(0)
        conversation = conversation.replace("<placeholder>", replace_str, 1)

    return conversation

class MultimodalChatDataset(Dataset):
    def __init__(
            self, 
            data_cfg, 
            mm_cfg, 
            tokenizer,
            image_processor,
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
            add_speakers=False,
            special_tokens=None,
    ):
        super().__init__()

        self.data_cfg = data_cfg
        self.mm_cfg = mm_cfg
        self.tokenizer = tokenizer

        self.max_seq_length = data_cfg.max_seq_length
        self.add_extra_token = add_extra_token
        self.ignore_index = ignore_index

        self.image_token = mm_cfg.get("image_token", "[IMG]")
        self.image_patch_token = mm_cfg.get("image_patch_token", "[IMG]")
        self.im_start_token = mm_cfg.get("im_start_token", "[IMG_BREAK]")
        self.im_end_token = mm_cfg.get("im_end_token", "[IMG_END]")
        self.image_folder = image_folder
        self.image_loader = TarOrFolderImageLoader(self.image_folder) if self.image_folder else None
        self.image_processor = image_processor
        
        self.patch_size = self.mm_cfg.vision_encoder.patch_dim
        
        logging.warning("Loading images from the dataset")
        self.list_data_dict = []
        for line in open(data_cfg.file_path, "r"):
            record = json.loads(line)
            # Search for <img src="/absolute/path/to/image" in the conversation
            #   add it as record['image'], remove src tag from the <img> tag
            record['images'] = []
            img_pattern = r'<img\s+src=[\'"]([^\'"]+)[\'"](?:[^>]*)?/?>'
            for turn in record['conversations']:
                matches = re.finditer(img_pattern, turn['value'])
                for match in matches:
                    image_name = match.group(1).split("/")[-1]
                    image_path = os.path.join(image_folder, image_name)

                    if not os.path.isfile(image_path):
                        logging.warning(f"Image not found: {image_path}")
                        continue
                    
                    record['images'].append(image_name)  # url

                turn['value'] = re.sub(img_pattern, self.image_token, turn['value'])

            self.list_data_dict.append(record)        
        
        self.add_speakers = add_speakers
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

    def __len__(self):
        return len(self.list_data_dict)
    
    def get_prompt(self, system_token, system_message, messages, add_speakers: bool = False) -> str:
        prompt_dict = []

        if system_message is not None:
            prompt = f"{self.special_tokens['system_turn_start']}{system_token}{self.special_tokens['end_of_name']}"        
            prompt += f"{system_message}"
        else:
            prompt = ""
        
        prompt_dict.append({"system": prompt})

        for turn in messages:
            if add_speakers:
                speaker = f"{self.special_tokens['turn_start']}{turn['from']}{self.special_tokens['end_of_name']}"
            else:
                speaker = f"{self.special_tokens['turn_start']}"
            message = f"{turn['value']}{self.special_tokens['end_of_turn']}"
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
            system_message = source.get('system', None)
            system_token = source.get('system_token', "System")
            mask_role = source.get('mask', "User")
            for i, turn in enumerate(source['conversations']):
                if i % 2 == 1:
                    mask_turn = True if turn['from'] == mask_role else False
                    messages.append({"from": turn['from'], "value": turn['value'], "mask": mask_turn})
                else:
                    mask_turn = True if turn['from'] == mask_role else False
                    messages.append({"from": turn['from'], "value": turn['value'], "mask": mask_turn})
            _, context_dict = self.get_prompt(system_token, system_message, messages, add_speakers=self.add_speakers)

        # Mask targets
        tokens_list = []
        labels_list = []
        if self.data_cfg.add_bos:
            tokens_list.extend([self.tokenizer.bos_id])
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

    def preprocess_media_tokens(self, sources: dict, image_sizes: List[Tuple[int, int]]) -> Dict:
        """
        Preprocesses multimodal sources based on the provided configuration.

        This function modifies the sources for multimodal data processing. It checks if the data is multimodal and
        adjusts the token lengths accordingly. It also handles the start and end tokens for images and replaces
        image tokens in conversations.

        Parameters:
        - sources (dict): A dictionary containing the multimodal sources to be processed.
        - image_sizes (List[Tuple]): A list denoting the size if each image in the sample
        Returns:
        - dict: The processed sources dictionary after applying multimodal preprocessing steps.
        """

        if image_sizes is None:
            return sources

        for source in sources:
            conversation = source['conversations']
            for turn in conversation:
                original_text = turn.get("value", "")
                turn["value"] = _preprocess_media_tokens(
                    conversation=original_text,
                    image_token=self.image_token,
                    image_patch_token=self.image_patch_token,
                    im_start_token=self.im_start_token,
                    im_end_token=self.im_end_token,
                    patch_size=self.patch_size,
                    image_sizes=image_sizes
                )
                
        return sources

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        crop_size = copy.deepcopy(self.mm_cfg.vision_encoder['crop_size'])
        if 'images' in sources[0]:
            if not isinstance(self.list_data_dict[i]['images'], list):
                self.list_data_dict[i]['images'] = [self.list_data_dict[i]['images']]

            images = []
            for image_file in self.list_data_dict[i]['images']:
                image = self.image_loader.open_image(image_file)
                if image is None:
                    logging.warning(f"Image {image_file} could not be found!")    
                    continue

                images.append(image)

            if images:
                image_inputs = self.image_processor(images, return_tensors="pt")
                image_list, image_sizes = image_inputs["pixel_values"], image_inputs["image_sizes"]

                if isinstance(image_list[0], list): # List of List of Images            
                    image_list = image_list[0]

                if isinstance(image_sizes[0], list):
                    image_sizes = image_sizes[0]

                sources = self.preprocess_media_tokens(sources=copy.deepcopy(sources), image_sizes=image_sizes)
                      
        else:
            logging.warning("media not found in sources")            
            sources = copy.deepcopy(sources)

        data_dict = self.preprocess_conversations(sources)

        if isinstance(i, int):
            data_dict = dict(tokens=data_dict["tokens"][0], labels=data_dict["labels"][0])

        current_num_images = len(image_list)
        if current_num_images < MIN_NUM_IMAGES:
            image_list.append = torch.zeros(3, crop_size[0], crop_size[1], dtype=torch.float)
        data_dict['image'] = image_list
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    model_cfg: DictConfig
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        max_len = (max_len - 1) // 64 * 64 + 64
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', -1)

        # Use default_collate for the tokens and labels
        batch = default_collate([{k: v for k, v in instance.items() if k != 'image'} for instance in instances])

        tokenizer = self.tokenizer
        model_cfg = self.model_cfg

        tokens = batch['tokens']
        labels = batch['labels']

        media = [torch.nested.nested_tensor(instance['image']) for instance in instances]
        media = TensorList(media)

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=tokenizer.eos_id,
            eod_mask_loss=model_cfg.data.get("eod_mask_loss", False),
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        batch = {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'media': media,
        }

        return batch
    

def extract_real_images(sample_images: torch.Tensor, padding_val: int = -100) -> List[torch.Tensor]:
    """
    Extracts real images from a padded image tensor.

    Args:
        sample_images (torch.Tensor): Tensor of shape [num_images, 3, H, W],
                                      where some images may be padded with -1.

    Returns:
        List[torch.Tensor]: List of real images, each of shape [3, H', W'].
    """
    real_images = []
    num_images = sample_images.shape[0]
    
    for idx in range(num_images):
        img = sample_images[idx]  # Shape: [3, H, W]
        
        # Check if the image is entirely padded
        if torch.all(img == padding_val):
            continue  # Skip padded image
        
        # Create a mask where pixels are not padding_val
        mask = img != padding_val  # Shape: [3, H, W]
        mask = mask.any(dim=0)  # Shape: [H, W]
        
        # Find the last row that has any non-padded pixel
        rows = torch.any(mask, dim=1)  # Shape: [H]
        if rows.any():
            last_row = torch.nonzero(rows, as_tuple=False).max().item() + 1  # +1 for exclusive slicing
        else:
            last_row = 0
        
        # Find the last column that has any non-padded pixel
        cols = torch.any(mask, dim=0)  # Shape: [W]
        if cols.any():
            last_col = torch.nonzero(cols, as_tuple=False).max().item() + 1  # +1 for exclusive slicing
        else:
            last_col = 0
        
        # Slice the image to its true dimensions
        if last_row > 0 and last_col > 0:
            sliced_img = img[:, :last_row, :last_col]  # Shape: [3, H', W']
            real_images.append(sliced_img)
        else:
            # If no non-padded pixels found, skip the image
            continue
    
    return real_images

@dataclass
class PaddedDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    model_cfg: DictConfig
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Determine the maximum token length in the batch
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        # Optional: Adjust max_len to be a multiple of 64 if needed
        # max_len = (max_len - 1) // 64 * 64 + 64

        # Pad tokens with 0 and labels with IMAGE_PADDING_VAL
        tokens_padded = torch.stack([
            F.pad(instance['tokens'], (0, max_len - instance['tokens'].shape[0]), 'constant', 0)
            for instance in instances
        ], dim=0)  # Shape: (batch_size, max_len)

        labels_padded = torch.stack([
            F.pad(instance['labels'], (0, max_len - instance['labels'].shape[0]), 'constant', IMAGE_PADDING_VAL)
            for instance in instances
        ], dim=0)  # Shape: (batch_size, max_len)

        # Handle images
        images = [instance['image'] for instance in instances]  # List[Tensor] each of shape (3 * MIN_NUM_IMAGES, H, W)

        # Determine the maximum number of images, height, and width in the batch
        max_num_images = max(img.shape[0] for img in images)
        max_H = max(img.shape[2] for img in images)
        max_W = max(img.shape[3] for img in images)

        # Pad each image tensor to have the same number of images, height, and width
        padded_images = []
        for img in images:
            num_images, C, H, W = img.shape
            pad_num_images = max_num_images - num_images
            pad_H = max_H - H
            pad_W = max_W - W

            # Initialize padding tensor if needed
            if pad_num_images > 0:
                # Create padding images filled with -1
                padding_images = torch.full(
                    (pad_num_images, C, H, W),
                    fill_value=IMAGE_PADDING_VAL,
                    dtype=img.dtype,
                    device=img.device
                )
                img = torch.cat([img, padding_images], dim=0)  # Shape: [max_num_images, 3, H, W]

            if pad_H > 0 or pad_W > 0:
                # Pad each image to match max_H and max_W
                # Pad format: (left, right, top, bottom)
                img = F.pad(img, (0, pad_W, 0, pad_H), mode='constant', value=IMAGE_PADDING_VAL)  # Shape: [max_num_images, 3, max_H, max_W]

            padded_images.append(img)

        # Stack images into a batch tensor
        images_batch = torch.stack(padded_images, dim=0)  # Shape: (batch_size, 3 * MIN_NUM_IMAGES, max_height, max_width)
        print(f"{images_batch.shape = }")
        for image_batch in images_batch:
            real_images = extract_real_images(image_batch, padding_val=IMAGE_PADDING_VAL)
            print(f"{len(real_images) = }")
            for real_image  in real_images:
                print(f"{real_image.shape = }")
        ashdkshankjdhas
        # Generate attention masks and position ids
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens_padded,
            eod_token=self.tokenizer.eos_id,
            eod_mask_loss=self.model_cfg.data.get("eod_mask_loss", False),
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        # Adjust masks and labels
        loss_mask[labels_padded == -1] = 0.0
        tokens_padded[tokens_padded == -1] = 0
        labels_padded[labels_padded == -1] = 0

        # Move tensors to the device
        device = tokens_padded.device
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        position_ids = position_ids.to(device)
        images_batch = images_batch.to(device)

        # Prepare the final batch dictionary
        batch = {
            'tokens': tokens_padded,          # Shape: (batch_size, max_len)
            'labels': labels_padded,          # Shape: (batch_size, max_len)
            'attention_mask': attention_mask,  # Shape: (batch_size, max_len)
            'loss_mask': loss_mask,            # Shape: (batch_size, max_len)
            'position_ids': position_ids,      # Shape: (batch_size, max_len)
            'media': images_batch,             # Shape: (batch_size, 3 * MIN_NUM_IMAGES, max_height, max_width)
        }

        return batch
    