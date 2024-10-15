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
from nemo.collections.multimodal.data.neva.neva_dataset import TarOrFolderImageLoader, process_image

import transformers
from nemo_aligner.utils.multimodal import TensorList

MAX_NUM_IMAGES = 1
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

def process_media_tokens(
        text: str, 
        media_token: str, 
        media_patch_token: str, 
        media_start_token: str, 
        media_end_token: str, 
        num_media_tokens: int, 
        use_im_start_end: bool
    ) -> Dict:
    if use_im_start_end:
        replace_token = media_patch_token * num_media_tokens
    else:
        replace_token = media_patch_token * (num_media_tokens - 2)
    replace_token = media_start_token + replace_token + media_end_token
    
    text = text.replace(media_token, replace_token)
    return text

def maybe_process_prompt_and_media(
        record,
        image_loader, 
        image_processor, 
        image_token, 
        image_patch_token, 
        image_start_token, 
        image_end_token, 
        image_aspect_ratio, 
        image_patch_dim, 
        use_im_start_end, 
        crop_size,
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
                im_start_token=image_start_token,
                im_end_token=image_end_token,
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
                image_list.append(torch.zeros(3, crop_size[0], crop_size[1], dtype=torch.float))
        
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
        self.media_type = cfg.data.get("media_type", "image") # Only image is supported for now
        self.image_token = cfg.mm_cfg.get("image_token", "[IMG]")
        self.image_patch_token = cfg.mm_cfg.get("image_patch_token", "[IMG]")
        self.image_folder = cfg.data.get("image_folder", None)        
        self.image_start_token = cfg.mm_cfg.get("im_start_token", "[IMG_BREAK]")
        self.image_end_token = cfg.mm_cfg.get("im_end_token", "[IMG_END]")
        self.add_extra_token = cfg.data.get("add_extra_token", 0) 
        self.image_loader = TarOrFolderImageLoader(self.image_folder) if self.image_folder else None
        self.image_processor = image_processor
        self.image_aspect_ratio = cfg.data.image_aspect_ratio
        self.image_patch_dim = cfg.mm_cfg.vision_encoder.patch_dim
        self.use_im_start_end = cfg.mm_cfg.use_im_start_end
        self.crop_size = cfg.mm_cfg.vision_encoder.crop_size

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
                    self.image_start_token,
                    self.image_end_token,
                    self.image_aspect_ratio,
                    self.image_patch_dim,
                    self.use_im_start_end,
                    self.crop_size,
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

def dpo_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    chosen_tokens = [item["chosen"] for item in batch]
    rejected_tokens = [item["rejected"] for item in batch]
    chosen_lengths = torch.LongTensor([item["chosen_length"] for item in batch])
    rejected_lengths = torch.LongTensor([item["rejected_length"] for item in batch])
    chosen_labels = [item["chosen_labels"] for item in batch]
    rejected_labels = [item["rejected_labels"] for item in batch]
    chosen_rewards = torch.FloatTensor([item["chosen_reward"] for item in batch])
    rejected_rewards = torch.FloatTensor([item["rejected_reward"] for item in batch])

    chosen_tokens = torch.nn.utils.rnn.pad_sequence(chosen_tokens, batch_first=True, padding_value=eos_id)
    rejected_tokens = torch.nn.utils.rnn.pad_sequence(rejected_tokens, batch_first=True, padding_value=eos_id)
    chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=-100)
    rejected_labels = torch.nn.utils.rnn.pad_sequence(rejected_labels, batch_first=True, padding_value=-100)

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        chosen_tokens, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss,
    )
    assert attention_mask.ndim == 4, "attention_mask is incorrect shape for dpo_custom_collate"
    if attention_mask.shape[0] == 1:
        # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
        # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
        attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))

    media = [torch.nested.nested_tensor(item['media']) for item in batch]
    media = TensorList(media)

    output = {
        "chosen": chosen_tokens,
        "rejected": rejected_tokens,
        "chosen_length": chosen_lengths,
        "rejected_length": rejected_lengths,
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "chosen_media": media,
        "rejected_media": media,
    }
        
    return output


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
            img_pattern = r'<data-img\s+src=[\'"]([^\'"]+)[\'"](?:[^>]*)?/?>'
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