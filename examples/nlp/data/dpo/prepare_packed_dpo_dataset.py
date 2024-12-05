# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.sequence_packing_utils import create_hist, create_packing_strategy
from nemo_aligner.data.nlp.builders import build_train_valid_test_dpo_datasets
from nemo_aligner.data.nlp.datasets import DPOModelDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

""" 
Script to prepare packed dataset from a DPO dataset in the jsonl format.
Three  main steps are run in this script:
1. The online processing code in DPOModelDataset is run (including prompt template manipulation, 
sequence length truncation, tokenization, etc) and the result is an array of tokenized sequences, 
represented by indices). 
2. chosen and rejected sequences are concatenated for each example
3. The sequences are grouped by length, and a packing algorithm is run. (https://en.wikipedia.org/wiki/Bin_packing_problem#Offline_algorithms)
Currently, two variants of "first fit" are supported.
"first_fit_decreasing" sorts the sequences in decreasing order before applying first-fit. 
It generates a more optimal packing, but it tends to keep all short sequences together, which may affect convergence.
"first_fit_shuffle" runs first-fit in a random order. Packing is less optimal but it keeps the dataset order random.
The recommendation is to run "first_fit_shuffle" and check the packed sequence lengths in the printout. 
If they are similar to the target length (i.e. packing is efficient), then use shuffle. Otherwise try first_fit_decreasing.

Example usage:

python scripts/nlp_language_modeling/prepare_packed_dpo_dataset.py \
   model.data.train_ds.file_names=[/path/to/training.jsonl] \
   model.encoder_seq_length=1024 \
   +tokenizer_path=<see note 1 below> \
   +tokenizer_type=sentencepiece \
   +output_dir=/path/to/output_folder \
   +pack_sizes=[2048,4096,8192]
   
Note: 
  - Tokenizer path supports SentencePiece tokenizer and HF tokenizer. 
    For SentencePiece tokenizer, specify the file /path/to/tokenizer.model 
    For HF tokenizer, specify a folder /path/to/hf_folder which contains tokenizer.json, tokenizer_config.json
    and special_tokens_map.json or the HF name of the tokenizer to use (e.g. "meta-llama/Meta-Llama-3-8B")

  - If your model or dataset requires non-default configs for DPO training in NeMo, you will
    need to pass in the same configs to ``model.data.train_ds`` as you would for training with unpacked dataset.

  - ``model.encoder_seq_length`` is the length to truncate each sequence before packing multiple sequences
    to the size of packed sequence (``pack_size``). 

  - ``pack_sizes`` is a list of packed sequence lengths. In this example, there will be three output files, one for
    each pack size. The output files are named ``<output_folder>/packed_{pack_size}_seed{seed}.npy``.
    This argument is a list because you will likely want to experiment with a few ``pack_sizes`` to find out which length
    can fill the GPU memory without exceeding it. Adjusting ``pack_size`` is analogous to adjusting the micro batch size in
    the unpacked case.
      - **important**: ``pack_sizes`` should be at least double the value of model.encoder_seq_length in order to guarantee
        that chosen and rejected sequences for a given example can be packed together.
"""


def tokenize_dataset(cfg: "DictConfig", tokenizer_type):
    """
    Tokenizes a dataset using the same configuration file as DPOModelDataset.

    This function reads a dataset and tokenizes based on the provided configuration.

    Args:
      cfg: A Hydra configuration object containing parameters for tokenization.

    Returns:
      A NumPy array containing the tokenized sequences from the dataset.
    """

    logging.info("Tokenizing dataset...")

    if tokenizer_type == "huggingface":
        # pass in either a local Hugging Face folder which contains tokenizer.json or a path to the tokenizer on huggingface
        tokenizer = get_nmt_tokenizer(library="huggingface", model_name=cfg.tokenizer_path, use_fast=True)
    elif tokenizer_type == "sentencepiece":
        tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=cfg.tokenizer_path)
    else:
        raise ValueError(f"unsupported tokenizer type {tokenizer_type}")

    with open(cfg.model.data.data_prefix, "r", encoding="utf_8") as fr:
        data_payload = [json.loads(line.strip()) for line in fr]
    documents = np.arange(len(data_payload), step=1, dtype=np.int32)
    dataset = DPOModelDataset(
        cfg=cfg.model,
        name="packing_dataset",
        tokenizer=tokenizer,
        data_prefix=cfg.model.data.data_prefix,
        documents=documents,
        data=data_payload,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.get("seed", 1234),
        drop_last=True,  ## False not currently supported
        pad_chosen_rejected_to_max=False,
    )

    combined_dataset = []
    for item in dataset:
        if item["ignore_example"]:
            continue
        input_ids = torch.cat((item["chosen"], item["rejected"])).numpy()
        labels = torch.cat((item["chosen_labels"], item["rejected_labels"])).numpy()
        reward = torch.tensor([item["chosen_reward"], item["rejected_reward"]]).numpy()
        boundary = len(item["chosen"])
        lengths = np.array([item["chosen_length"], item["rejected_length"]])
        new_item = {
            "input_ids": input_ids,
            "labels": labels,
            "reward": reward,
            "lengths": lengths,
            "boundary": boundary,
        }
        combined_dataset.append(new_item)

    return np.array(combined_dataset)


## modified version of https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/sequence_packing_utils.py#L178 for DPO
## pack size should be at least 2*encoder_seq_length since the packed sequences include both the chosen and rejected sequences
## for a given example
def fill_packing_strategy(
    assignments: List[List[int]], sequences: Dict[int, List[Dict]], pack_size: int
) -> List[Dict]:
    """
    Fills the packing strategy with actual sequence data based on assignments and sequence information.

    This function takes the assignments generated by the packing algorithm (containing sequence length indices),
    the original sequences data, and the pack size. It iterates through the assignments, retrieves the corresponding
    sequences from the sequences dictionary, and constructs the final output data structure with input IDs, loss masks
    (if available), and starting indices for each sequence in a packed sequence.

    Args:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin (output of 'create_packing_strategy').
          sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences
                      from the dataset (output of 'create_hist').
          pack_size: The maximum capacity of each bin.

    Returns:
          output_data: A list of dictionaries, where each dictionary represents a packed sequence with its input IDs,
                        loss mask (if available), and starting indices.
    """
    ifile_handles = dict()
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            perm = np.random.permutation(len(per_seq_data))

            perm = np.random.permutation(len(per_seq_data))
            input_ids = np.array([x["input_ids"] for x in per_seq_data])[perm].tolist()
            labels = np.array([x["labels"] for x in per_seq_data])[perm].tolist()
            reward = np.array([x["reward"] for x in per_seq_data])[perm].tolist()
            lengths = np.array([x["lengths"] for x in per_seq_data])[perm].tolist()
            boundary = np.array([x["boundary"] for x in per_seq_data])[perm].tolist()

            ifile_handles[seq_len] = (input_ids, labels, reward, lengths, boundary)

    input_ids, labels, reward, lengths, seq_boundaries = {}, {}, {}, {}, {}

    for oindex, assignment in tqdm(enumerate(assignments), total=len(assignments)):
        _input_ids, _labels, _reward, _lengths, _seq_boundaries = [], [], [], [], [0]

        for seq_length in assignment:

            previous_seq_len = len(_input_ids)

            _input_ids.extend(ifile_handles[seq_length][0].pop())
            _labels.extend(ifile_handles[seq_length][1].pop())
            _reward.extend(ifile_handles[seq_length][2].pop())
            _lengths.extend(ifile_handles[seq_length][3].pop())

            ## store the boundaries for the chosen, rejected sequences
            _seq_boundaries.append(previous_seq_len + ifile_handles[seq_length][4].pop())
            _seq_boundaries.append(len(_input_ids))

        input_ids[oindex] = _input_ids
        labels[oindex] = _labels
        reward[oindex] = _reward
        lengths[oindex] = _lengths
        seq_boundaries[oindex] = _seq_boundaries

    output_data = []
    for i in range(len(input_ids)):
        item_dict = {
            "input_ids": input_ids[i],
            "labels": labels[i],
            "reward": reward[i],
            "lengths": lengths[i],
            "seq_boundaries": seq_boundaries[i],
        }
        output_data.append(item_dict)

    # (input_ids, labels, reward, lengths, boundary) = length 5
    for i in range(5):
        assert all(
            not seq[i] for seq in ifile_handles.values()
        ), "Error: There are items left over from the assignment"
    return output_data


@dataclass
class PackingArgs:
    output_dir: str = "output"
    pack_sizes: Tuple[int] = (2048,)
    packing_algorithm: str = "first_fit_shuffle"
    tokenizer_type: str = "sentencepiece"  ## one of "huggingface" or "sentencepiece"

    def from_config(self, cfg: "DictConfig"):
        for required_arg in ("output_dir", "pack_sizes"):
            assert cfg.get(required_arg, None), f"Please specify +{required_arg}=..."
        self.output_dir = cfg.output_dir
        self.pack_sizes = cfg.pack_sizes
        self.packing_algorithm = cfg.get("packing_algorithm", "first_fit_shuffle")
        self.tokenizer_type = cfg.tokenizer_type
        return self


@hydra_runner(config_path="../../gpt/conf", config_name="gpt_dpo")
def main(cfg: "DictConfig") -> None:
    args = PackingArgs().from_config(cfg)
    dataset = tokenize_dataset(cfg, args.tokenizer_type)
    sequences, histogram = create_hist(
        dataset, 2 * cfg.model.data.seq_length
    )  ## multiply by 2 because packed sequences include chosen and rejected
    for pack_size in args.pack_sizes:
        assignments = create_packing_strategy(histogram, pack_size, args.packing_algorithm)
        output_data = fill_packing_strategy(assignments, sequences, pack_size)

        # save output data
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"packed_{pack_size}_seed{cfg.model.get('seed', 1234)}.npy")
        np.save(output_path, output_data)
        logging.info(f"Done, output written to {output_path}")

    logging.info(
        f"""
✅ Packed datasets with pack sizes {args.pack_sizes} are prepared successfully. 
To train with packed sequences, you need to make changes to the DPO config file.
See the NeMo-Aligner sequence packing documentation for more details:
https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/user-guide/dpo.rst#sequence-packing-with-dpo 
"""
    )


if __name__ == "__main__":
    main()
