import os
import jsonlines

import torch

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo_aligner.utils.utils import batch_pad_to_fixed_len

# TODO @sahilj handle too-long prompts and masking them out throughout the whole process and renormalizing on loss
class AllTaskDataset:
    def __init__(
        self,
        data_path,
        tokenizer,
        apply_chat_template: bool = True,
        system_prompt_file: str = None,
        prompt_file: str = None,
        seq_length=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.system_prompt = None
        self.prompt = "{}"

        assert os.path.exists(self.data_path), f"{self.data_path} must exist"

        with jsonlines.open(self.data_path) as reader:
            self.data = [obj for obj in reader]

        if self.apply_chat_template:
            if system_prompt_file is not None:
                assert os.path.exists(
                    system_prompt_file
                ), f"Sys prompt file {system_prompt_file} was specified but does not exist"
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    self.system_prompt = f.read()
        elif system_prompt_file is not None:
            print(
                f"WARNING: system_prompt_file specified for {type(self)} but apply_chat_template is false. Ignoring this prompt."
            )

        if prompt_file is not None:
            assert os.path.exists(prompt_file), f"prompt file {prompt_file} was specified but does not exist"
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.prompt = f.read()

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        text_ids = self.tokenizer.text_to_ids(text)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """
        task_name = self.data[idx]["task_name"]
        extra_verifier_info = None
        if task_name == "math":
            text_str = self.data[idx]["problem"]
            extra_verifier_info = {"ground_truth": self.data[idx]["expected_answer"]}
        else:
            raise NotImplementedError(f"task name {task_name} in your dataset doesn't have a handler yet!")

        if self.apply_chat_template:
            chat = []
            if self.system_prompt:
                chat.append({"role": "system", "content": self.system_prompt})
            chat.append({"role": "user", "content": self.prompt.format(text_str)})
            text = self.tokenizer.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            text = self.prompt.format(text_str)

        sample, _ = self.encode(text)
        sample_tensor = torch.as_tensor(sample, dtype=torch.int64)

        output = {
            "text": sample_tensor,
            "length": sample_tensor.shape[0],
            "extra_verifier_info": extra_verifier_info,
            "loss_multiplier": True,
            "idx": idx,
            "task_name": task_name,
        }
        return output


def environment_collate_with_batch_max_sequence_length(
    data_batch,
    response_token_length,
    eos_id,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    generate_masks_and_position_ids,
):
    """collate function that batches by max sequence length
    """
    texts = [item["text"] for item in data_batch]
    loss_multipliers = torch.as_tensor([item["loss_multiplier"] for item in data_batch]).view(len(data_batch), 1)
    lengths = torch.as_tensor([item["length"] for item in data_batch])
    extra_verifier_info = [item["extra_verifier_info"] for item in data_batch]
    task_name = [item["task_name"] for item in data_batch]
    idx = [item["idx"] for item in data_batch]

    batch_max_length = lengths.max()

    texts = batch_pad_to_fixed_len(texts, batch_max_length + response_token_length, eos_id)

    output = {
        "text": texts,
        "length": lengths,
        "extra_verifier_info": extra_verifier_info,
        "idx": idx,
        "task_name": task_name,
    }

    other = {}
    if generate_masks_and_position_ids:
        # NOTE: the attention mask is 1x1xSxS, which will broadcast on the batch dimension
        attention_masks, loss_masks, position_ids = get_ltor_masks_and_position_ids(
            texts, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss
        )
        other = {
            "attention_mask": attention_masks,
            # to preserve the loss mask from the dataset
            "loss_mask": loss_masks * loss_multipliers,
            "position_ids": position_ids,
        }

    return output | other
