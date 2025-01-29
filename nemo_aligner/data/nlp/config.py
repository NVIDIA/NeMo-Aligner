from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import torch
from torch.utils.data import DataLoader

from nemo_aligner.algorithms.dpo import DPOTrainer, dpo_custom_collate
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_dpo_datasets,
    build_train_valid_test_dpo_packed_datasets,
    build_train_valid_test_rlhf_datasets,
    identity_collate,
)


class GlobalCollateProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        ...


@dataclass
class DataConfig(ABC):
    micro_batch_size: int
    global_batch_size: int
    seq_length: int

    data_impl: str = "jsonl"
    splits_string: Optional[str] = None
    # seq_length: Optional[str] = None ## TODO: infer this from model!
    skip_warmup: bool = True
    num_workers: int = 0
    reset_position_ids: bool = False  # Reset position ids after end-of-document token
    reset_attention_mask: bool = False  # Reset attention mask after end-of-document token
    eod_mask_loss: bool = False  # Mask loss for the end of document tokens
    add_bos: bool = False
    add_eos: bool = False
    add_sep: bool = False
    append_eod: bool = False
    index_mapping_dir: Optional[
        int
    ] = None  # path to save index mapping .npy files, by default will save in the same location as data_prefix
    data_prefix: Optional[dict[str, list[str]]] = None  ## should be specified in experiment
    validation_drop_last: bool = True
    seed: int = 1234

    # special tokens for the chat prompts, a dictionary of {token_type: token}. note that some tokenizer may combine the characters at the junction between {end_of_turn}{turn_start}. e.g. '<im end><im start>', the '><' sometimes is merged to be a single token. This is not supported, try to avoid
    ## TODO: figure this out
    """chat_prompt_tokens: dict = field(
        default_factory=({
            "system_turn_start": "\x00",
            "turn_start": "\x11",
            "label_start": "\x12",
            "end_of_turn": "\x0A",  # \0x0A is '\n'
            "end_of_name": "\x0A",  # \0x0A is '\n'
        })
    )"""

    label_key: str = "answer"
    truncation_field: str = "text"
    pad_to_max_length: bool = False
    prompt_template: Optional[str] = None  ## TODO: needs to be specified by downstream task
    memmap_workers: Optional[int] = None
    hf_dataset: bool = False
    truncation_method: str = "right"
    output_original_text: bool = False

    @abstractmethod
    def build_dataloaders(self) -> tuple[DataLoader, DataLoader, GlobalCollateProtocol]:
        """subclass should implement this"""


@dataclass
class DPODataConfig(DataConfig):
    packed_sequence: bool = False
    packed_sequence_return_cu_seqlen: bool = True
    pad_length_to_multiple_of: Optional[
        int
    ] = None  # If using sequence_parallel, ensure divisible by tensor_model_parallel_size
    default_chosen_reward: float = 1.0  # the default reward for the chosen response in RPO
    default_rejected_reward: float = 0.0  # the default reward for the rejected response in RPO
    apply_ftfy: bool = False

    def build_dataloaders(self, tokenizer) -> tuple[DataLoader, DataLoader, GlobalCollateProtocol]:
        # use the entire dataset
        train_valid_test_num_samples = [-1 * self.global_batch_size] * 3
        ## build the dataset (should be mostly unchanged from before, except the config)
        if self.data_impl == "packed_jsonl":
            build_fn = build_train_valid_test_dpo_packed_datasets
        else:
            build_fn = build_train_valid_test_dpo_datasets
        train_ds, validation_ds, _ = build_fn(
            cfg=self,
            data_prefix=self.data_prefix,
            data_impl=self.data_impl,
            splits_string=self.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.seq_length,  ## TODO: check
            seed=self.seed,  ## TODO: check
            tokenizer=tokenizer,  ## TODO: tokenizer
        )

        global_collate_fn = train_ds.global_collate_fn if self.data_impl == "packed_jsonl" else dpo_custom_collate
        train_dataloader = build_dataloader(
            cfg=self,
            dataset=train_ds,
            consumed_samples=0,  # consumed_samples, ## TODO: UPDATE
            mbs=self.micro_batch_size,
            gbs=self.global_batch_size,
            load_gbs=True,
            pad_samples_to_global_batch_size=False,
            collate_fn=identity_collate,
        )

        val_dataloader = build_dataloader(
            cfg=self,
            dataset=validation_ds,
            consumed_samples=0,
            mbs=self.micro_batch_size,
            gbs=self.global_batch_size,
            load_gbs=True,
            pad_samples_to_global_batch_size=False,
            collate_fn=identity_collate,
            use_random_sampler=False,
        )
        return train_dataloader, val_dataloader, global_collate_fn


@dataclass
class RLHFDataConfig(DataConfig):
    pass
