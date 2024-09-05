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
import numpy as np
import torch
from einops import rearrange
from typing import Any, Dict, List
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.multimodal.data.neva.neva_dataset import NevaDataset, TarOrFolderImageLoader, process_image
from PIL import Image
from transformers import CLIPImageProcessor, SiglipImageProcessor

MAX_NUM_IMAGES = 1

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
        is_multimodal: bool = True
    ):
    if "image" in record:
        if not isinstance(record['image'], list):
            record['image'] = [record['image']]
        images = []
        for image_file in record["image"]:
            image = image_loader.open_image(image_file)
            if image is None:
                logging.warning(f"Image {image} could not be found!")
            image = process_image(image_processor, image, image_aspect_ratio)
            images.append(image)
        media_tensors = torch.tensor([])
        if images:
            media_tensors = torch.stack(images)
            height_num_patches = media_tensors[0].shape[1] // image_patch_dim
            width_num_patches  = media_tensors[0].shape[2] // image_patch_dim
            num_image_tokens = height_num_patches * width_num_patches
            record['prompt'] = process_media_tokens(
                record['prompt'],
                image_token,
                image_patch_token,
                image_start_token,
                image_end_token,
                num_image_tokens,
                use_im_start_end,
            )

        # image exist in the data
        if is_multimodal:
            # Image does not exist in the data, but the model is multimodal
            # TODO, if there are different videos on T dimensions.
            if media_tensors.shape[0] < MAX_NUM_IMAGES:
                padding_size = MAX_NUM_IMAGES - media_tensors.shape[0]
                zero_padding = torch.zeros((padding_size, 3, crop_size[0], crop_size[1]), dtype=torch.float)
                media_tensors = torch.cat((media_tensors, zero_padding), dim=0)
        
        record["image"] = media_tensors
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
        self.image_token = cfg.mm_cfg.get("image_token", "<image>")
        self.image_patch_token = cfg.mm_cfg.get("image_patch_token", "<extra_id_3>")
        self.image_folder = cfg.data.get("image_folder", None)        
        self.image_start_token = cfg.mm_cfg.get("im_start_token", "<extra_id_4>")
        self.image_end_token = cfg.mm_cfg.get("im_end_token", "<extra_id_5>")
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
        img_pattern = r'<img\s+src=[\'"]([^\'"]+)[\'"](?:[^>]*)?/?>'
        self.data = []
        for record in data:
            # This currently supports only a single image
            # search for <img src="/absolute/path/to/image" in the conversation
            #   add it as record['image'], remove src tag from the <img> tag
            record['image'] = []
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
                record['image'].append(image_name)  # url
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
        payload = self.data[idx]
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
            "media": payload.get("image", None),
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

    media = batch.get('image')
    media = rearrange(media, "b T c h w -> b T 1 c h w") # TODO (tugrul): support different images for chosen and rejected samples

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


class MultimodalChatDataset(NevaDataset):
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
                image_processor=image_processor,
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
            raise NotImplementedError("Video modality is not supported.")
        else:
            return sources

        if not is_multimodal:
            return sources

        num_patches = image_token_len
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
                width_num_patches  = media_tensors[0].shape[2] // patch_dim

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
    