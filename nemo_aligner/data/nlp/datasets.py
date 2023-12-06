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

"""Custom datasets for RLHF training"""

import os

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import _create_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging


class RLHFDataset(Dataset):
    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, drop_last=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.data = data
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        if "length_params" in cfg:
            max_sample_length = seq_length - cfg.length_params.max_length
        else:
            max_sample_length = seq_length // 2

        assert max_sample_length > 0, f"max sample length must be greater than 0, but got {max_sample_length}"

        self.max_sample_length = max_sample_length

        self.use_json = self.cfg.data.data_impl.startswith("json")

        self.shuffled_indices = list(range(len(self)))

        np_rng = np.random.default_rng(seed=seed)
        np_rng.shuffle(self.shuffled_indices)

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(self.data)

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", None)

        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        if self.cfg.data.get("apply_ftfy", False):
            import ftfy

            text = ftfy.fix_text(text)

        text_ids = self.tokenizer.text_to_ids(text)

        if len(text_ids) > 0 and self.cfg.data.get("append_eod", False):
            text_ids.append(self.tokenizer.eos_id)

        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """
        mask_sample = False
        if idx == -1:
            # This may happen on the last batch due to the padding that occurs in
            #   https://github.com/NVIDIA/NeMo/blob/643d814fc2d885b7348ac676333ebd76cd79b663/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py#L168
            # in which case we may want to mask the loss associated to these padded samples.
            # However, this class is not currently used, so for now we raise an exception: this may be revisited
            # at a later time if this situation actually occurs in practice.
            # logging.warning("Got -1 as item index in RLHFDataset => masking loss from this sample")
            raise NotImplementedError("Obtained unexpected `idx == -1`, see comments in code for details")

        orig_idx = idx = idx % len(self)
        while True:
            shuffled_idx = self.shuffled_indices[idx]
            sample = self.data[shuffled_idx]
            if self.use_json:
                sample, _ = self.encode(sample["text"])
            if len(sample) <= self.max_sample_length:
                break
            idx = (idx + 1) % len(self)
            if idx == orig_idx:
                raise RuntimeError(f"All samples have length > {self.max_sample_length}")
            continue

        if idx != orig_idx:
            logging.warning(
                f"Sample {orig_idx} in dataset '{self.name}' has length "
                f"{len(self.data[self.shuffled_indices[orig_idx]])} > {self.max_sample_length} "
                f"=> replacing it with sample {idx} and masking its loss"
            )
            mask_sample = True

        if self.use_json:
            # `sample` is a regular Python list.
            sample_tensor = torch.tensor(sample, dtype=torch.int64)
        else:
            # `sample` is a NumPy array.
            sample_tensor = torch.from_numpy(sample.astype(np.int64))

        # if we want to mask the sample we should
        # set the loss multiplier to 0
        loss_multiplier = not mask_sample

        output = {
            "text": sample_tensor,
            "length": sample_tensor.shape[0],
            "loss_multiplier": loss_multiplier,
        }
        return output


class RewardModelDataset(Dataset):
    """This class assumes that we only have 2 responses per prompt that is ranked. Chosen is the better
        one(even index) whereas Rejected is the worse response(odd index)
    """

    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, drop_last=True,
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

        self.shuffled_indices = list(range(len(self)))

        np_rng = np.random.default_rng(seed=seed)
        np_rng.shuffle(self.shuffled_indices)

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(self.data)

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", None)

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
            shuffled_idx = self.shuffled_indices[idx]
            chosen = self.data[multiple * shuffled_idx]
            rejected = self.data[multiple * shuffled_idx + 1]
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
            "chosen_length": chosen_np.shape[0],
            "rejected_length": rejected_np.shape[0],
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        return output


class DPOModelDataset(Dataset):
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
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, drop_last=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.data = data.copy()
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.eos_id = tokenizer.eos_id

        np_rng = np.random.default_rng(seed=seed)
        np_rng.shuffle(self.data)

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(self.data)

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
        assert (
            max_curr_seq_len <= self.seq_length
        ), "tokenized text exceeds max seq len! truncate your data in preprocessing prior to DPO training"

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

        output = {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "chosen_length": chosen_len,
            "rejected_length": reject_len,
            "chosen_labels": labels_chosen_tokens,
            "rejected_labels": labels_reject_tokens,
        }
        return output


class RegressionRewardModelDataset(RewardModelDataset):
    """This class assumes each line of the dataset file is a dictionary with "text" and "label" field, 
        where "text" is a string representing the input prompt, and "label" is a list of float or int values. 
        Note that when training the model with multiple datasets which contain different attributes,
        we should set missing attributes to model.regression.loss_mask_val(according to training_rm.yaml)
        in the dataset files so that their losses are masked. At least one attribute should be present for each sample.

        WARNING: It's recommended to preprocess your data in advance to ensure all samples are within self.seq_length.
                 Otherwise if all samples in a batch are longer than self.seq_length, you may get NaN loss.
    """

    def __init__(
        self, cfg, tokenizer, name, data_prefix, documents, data, seq_length, seed, drop_last=True,
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
            shuffled_idx = self.shuffled_indices[idx]
            sample = self.data[shuffled_idx]
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
                f"Sample {self.shuffled_indices[orig_idx]} in dataset '{self.name}' has length "
                f"{orig_length} > {self.seq_length} "
                f"=> replacing it with sample {self.shuffled_indices[idx]} and masking its loss"
            )
            loss_mask = torch.zeros_like(loss_mask)

        output = {
            "inputs": text_tensor,
            "lengths": text_np.shape[0],
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": label_tensor,
        }
        return output
