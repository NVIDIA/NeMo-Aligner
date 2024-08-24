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
import gc
import os
import re
import copy
import numpy as np
import torch
from typing import Any, Dict, List
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import _create_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.multimodal.data.neva.neva_dataset import NevaDataset, TarOrFolderImageLoader, process_image
from PIL import Image
from einops import rearrange
from transformers import CLIPImageProcessor, SiglipImageProcessor

MAX_NUM_IMAGES = 1

def process_media_tokens(
        record: Dict, 
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
    
    record["text"] = record["text"].replace(media_token, replace_token)
    return record

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
        logging.warning(f"Processing image: {record['image']}")
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
            record = process_media_tokens(
                record,
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

class MultimodalRewardModelDataset(Dataset):
    """This class assumes that we only have 2 responses per prompt that is ranked. Chosen is the better
        one(even index) whereas Rejected is the worse response(odd index)
    """
    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed,  image_processor, drop_last=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.eos_id = tokenizer.eos_id

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
        assert np.max(documents) < len(data)

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", None)

        img_pattern = r'<img\s+src=[\'"]([^\'"]+)[\'"](?:[^>]*)?/?>'
        self.data = []
        for record in data:
            # This currently supports only a single image
            # search for <img src="/absolute/path/to/image" in the conversation
            #   add it as record['image'], remove src tag from the <img> tag
            record['image'] = []
            matches = re.finditer(img_pattern, record['text'])
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
            record['text'] = re.sub(img_pattern, self.image_token, record['text'])
            self.data.append(record)
        
        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

    def __len__(self):
        return len(self.data) // 2

    def encode(self, text):
        if self.cfg.data.get("apply_ftfy", False):
            import ftfy

            text = ftfy.fix_text(text)

        text_ids = self.tokenizer.text_to_ids(text)

        if len(text_ids) > 0 and self.cfg.data.get("append_eod", False):
            text_ids.append(self.tokenizer.eos_id)

        return text_ids, len(text_ids)

    def __getitem__(self, idx, multiple=2):
        """Returns a pair of chosen/rejected pairs, and their respective lengths.
        """
        found = False
        while not found:
            chosen = self.data[multiple * idx]
            rejected = self.data[multiple * idx + 1]
            items = []
            for i in range(multiple):
                item = self.data[multiple * idx + i]
                # Process media-related items
                item = maybe_process_prompt_and_media(
                    item,
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
                items.append(item)

            chosen, rejected = items
            chosen_media   = rearrange(chosen["image"]  , "T c h w -> T 1 c h w")
            rejected_media = rearrange(rejected["image"], "T c h w -> T 1 c h w")

            if self.cfg.data.data_impl.startswith("json"):
                chosen, _ = self.encode(chosen["text"])
                rejected, _ = self.encode(rejected["text"])

            if len(chosen) > self.seq_length or len(rejected) > self.seq_length:
                idx += multiple
                continue
            found = True
        
        # in the future, we should pad to the max seq len of the mini-batch instead of model.seq_length
        # max_curr_seq_len = max(len(chosen), len(rejected))

        chosen_np = np.array(chosen, dtype=np.int64)
        chosen_np_pad = np.pad(
            chosen_np, (0, max(0, self.seq_length - chosen_np.shape[0])), mode="constant", constant_values=self.eos_id
        )
        rejected_np = np.array(rejected, dtype=np.int64)
        rejected_np_pad = np.pad(
            rejected_np,
            (0, max(0, self.seq_length - rejected_np.shape[0])),
            mode="constant",
            constant_values=self.eos_id,
        )

        chosen_tokens = torch.tensor(chosen_np_pad)
        rejected_tokens = torch.tensor(rejected_np_pad)

        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            chosen_tokens, self.eos_id, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss,
        )

        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        # We make the loss_mask zero to mask out loss from these samples
        if idx == -1:
            logging.info("WARNING: Got -1 as item index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        output = {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "chosen_media": chosen_media,
            "rejected_media": rejected_media,
            "chosen_length": chosen_np.shape[0],
            "rejected_length": rejected_np.shape[0],
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        return output

class MultimodalRegressionRewardModelDataset(MultimodalRewardModelDataset):
    """This class assumes each line of the dataset file is a dictionary with "text" and "label" field, 
        where "text" is a string representing the input prompt, and "label" is a list of float or int values. 
        Note that when training the model with multiple datasets which contain different attributes,
        we should set missing attributes to model.regression.loss_mask_val(according to training_rm.yaml)
        in the dataset files so that their losses are masked. At least one attribute should be present for each sample.

        WARNING: It's recommended to preprocess your data in advance to ensure all samples are within self.seq_length.
                 Otherwise if all samples in a batch are longer than self.seq_length, you may get NaN loss.
    """

    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, image_processor, drop_last=True,
    ):

        assert cfg.data.data_impl.startswith(
            "json"
        ), f"data.data_impl must be either json or jsonl, but got {cfg.data.data_impl}"

        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            name=name,
            data_prefix=data_prefix,
            documents=documents,
            data=data,
            seq_length=seq_length,
            seed=seed,
            image_processor=image_processor,
            drop_last=drop_last,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one training sample, its label, and its respective length.
        """

        orig_idx = idx = idx % len(self)
        while True:
            sample = self.data[idx]
            # Process media-related items
            sample = maybe_process_prompt_and_media(
                sample,
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
            sample_text, sample_length = self.encode(sample["text"])
            sample_label = sample["label"]
            if idx == orig_idx:
                orig_length = sample_length
            if sample_length <= self.seq_length:
                break

            idx = (idx + 1) % len(self)
            if idx == orig_idx:
                raise RuntimeError(f"All samples have length > {self.seq_length}")

        assert isinstance(sample_label, list) and all(
            isinstance(value, (float, int)) for value in sample_label
        ), "label should be a list of float or int values"

        sample_label = [float(value) for value in sample_label]

        label_tensor = torch.tensor(sample_label, dtype=torch.float)

        text_np = np.array(sample_text, dtype=np.int64)
        text_np_pad = np.pad(
            text_np, (0, max(0, self.seq_length - text_np.shape[0])), mode="constant", constant_values=self.eos_id
        )

        text_tensor = torch.tensor(text_np_pad)
        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            text_tensor, self.eos_id, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss,
        )

        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        # We make the loss_mask zero to mask out loss from these samples
        if idx == -1:
            logging.waring("WARNING: Got -1 as item index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        # Replace current sample (when it exceeds max length) with another sample but mask its loss
        if idx != orig_idx:
            logging.warning(
                f"Sample {orig_idx} in dataset '{self.name}' has length "
                f"{orig_length} > {self.seq_length} "
                f"=> replacing it with sample {idx} and masking its loss"
            )
            loss_mask = torch.zeros_like(loss_mask)

        # Rearrange the media tensor to accommodate video for future releases
        media_tensor = rearrange(sample["image"], "T c h w -> T 1 c h w")

        output = {
            "inputs": text_tensor,
            "media": media_tensor,
            "lengths": text_np.shape[0],
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": label_tensor,
        }

        gc.collect()
        return output
