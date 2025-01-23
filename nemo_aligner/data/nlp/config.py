from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    data_impl: str = 'jsonl'
    splits_string: Optional[str] = None
    #seq_length: Optional[str] = None ## TODO: infer this from model!
    skip_warmup: bool = True
    num_workers: int = 0
    reset_position_ids: bool = False # Reset position ids after end-of-document token
    reset_attention_mask: bool = False # Reset attention mask after end-of-document token
    eod_mask_loss: bool = False # Mask loss for the end of document tokens
    add_bos: bool = False
    add_eos: bool = False
    add_sep: bool = False
    append_eod: bool = False
    index_mapping_dir: Optional[int] = None # path to save index mapping .npy files, by default will save in the same location as data_prefix
    data_prefix: Optional[int] = None ## should be specified in experiment
    validation_drop_last: bool = True
    #seed: int = 1234
    chat_prompt_tokens:  # special tokens for the chat prompts, a dictionary of {token_type: token}. note that some tokenizer may combine the characters at the junction between {end_of_turn}{turn_start}. e.g. '<im end><im start>', the '><' sometimes is merged to be a single token. This is not supported, try to avoid
        {
            "system_turn_start": "\x00",
            "turn_start": "\x11",
            "label_start": "\x12",
            "end_of_turn": "\x0A",  # \0x0A is '\n'
            "end_of_name": "\x0A",  # \0x0A is '\n'
        }
    label_key: "answer"
    truncation_field: "text"
    pad_to_max_length: bool = False
    prompt_template: None ## TODO: needs to be specified by downstream task
    memmap_workers: Optional[int] = None
    hf_dataset: bool = False
    truncation_method: str = "right"
    output_original_text: bool = False

@dataclass
class DPODataConfig(DataConfig):
    packed_sequence: bool = False
    packed_sequence_return_cu_seqlen: True
    pad_length_to_multiple_of: Optional[int] = None  # If using sequence_parallel, ensure divisible by tensor_model_parallel_size
    default_chosen_reward: float = 1. # the default reward for the chosen response in RPO
    default_rejected_reward: float = 0. # the default reward for the rejected response in RPO
    apply_ftfy: bool = False
