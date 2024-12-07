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

"""Generic file to build datasets
modified from: https://github.com/NVIDIA/NeMo/blob/2baef811f21372c3340dd2d82635d2377e78a660/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
to allow us to build SFT, RewardModel and RLHF datasets
"""
import json
import os
from functools import partial

import numpy as np
import torch
import torch.utils.data
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset, GPTSFTPackedDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.utils import logging
from nemo_aligner.data.nlp.datasets import (
    DPOModelDataset,
    DPOPackedDataset,
    KnowledgeDistillationDataset,
    KTOModelDataset,
    RegressionRewardModelDataset,
    RewardModelDataset,
    RLHFDataset,
)
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import collate_with_batch_max_sequence_length


class ChunkedJsonl:
    CHUNK_ID_STRING = "CHUNK_ID"

    def __init__(self, path_placeholder, n_chunks, n_examples_per_chunk):
        assert (
            self.CHUNK_ID_STRING in path_placeholder
        ), f"{path_placeholder=} does not contain {repr(self.CHUNK_ID_STRING)}"
        self.path_placeholder = path_placeholder

        # get the maximum number of chunks
        max_n_chunks = 0
        for i in range(n_chunks):
            max_n_chunks = i + 1  ## fix zero-indexing
            if not os.path.exists(self.path_placeholder.replace(self.CHUNK_ID_STRING, str(i))):
                break
        assert max_n_chunks > 0, f"no files match the required path {path_placeholder}"
        self.n_chunks = min(n_chunks, max_n_chunks)

        print("Initializing chunked jsonl...")
        lengths = [n_examples_per_chunk for _ in range(self.n_chunks)]
        print(f"Number of Chunks = {self.n_chunks} | Number of Examples = {n_examples_per_chunk * self.n_chunks}")
        self._lengths = np.asarray(lengths)
        self._length_accumulated = np.cumsum(lengths)

    def __len__(self):
        return self._lengths.sum()

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f"The item idx {i} is greater than the length of the dataset ({len(self)})")
        chunk_id = np.searchsorted(self._length_accumulated, i, side="right")
        idx_in_chunk = i if chunk_id == 0 else i - self._length_accumulated[chunk_id - 1]
        with open(self.path_placeholder.replace(self.CHUNK_ID_STRING, str(chunk_id))) as f:
            for line_idx, l in enumerate(f):
                if line_idx == idx_in_chunk:
                    return json.loads(l)
        raise ValueError(f"Reading the item {i} failed. Computed chunk_id={chunk_id}, idx_in_chunk={idx_in_chunk}.")


def build_dataset_generic(
    cls,
    cfg,
    data_prefix,
    data_impl,
    num_samples,
    seq_length,
    seed,
    tokenizer,
    name,
    n_chunks=None,
    n_examples_per_chunk=None,
):
    def _build_dataset(current_data_prefix, current_num_samples):
        if data_impl == "mmap":
            data_payload = get_indexed_dataset_(current_data_prefix, data_impl, cfg.data.get("skip_warmup", True))
        elif data_impl.startswith("json"):
            with open(current_data_prefix, "r", encoding="utf_8") as fr:
                data_payload = [json.loads(line.strip()) for line in fr]
        elif data_impl == "packed_jsonl":
            data_payload = np.load(current_data_prefix, allow_pickle=True)
        elif data_impl == "chunked_jsonl":
            assert isinstance(n_chunks, int) and n_chunks >= 1, f"Not valid n_chunks {n_chunks}"
            data_payload = ChunkedJsonl(current_data_prefix, n_chunks, n_examples_per_chunk)
        else:
            raise RuntimeError(
                f"data.data_impl must be one of mmap, json, jsonl, packed_jsonl, or chunked_jsonl, but got {data_impl}"
            )
        total_num_of_documents = len(data_payload)

        # Print stats about the splits.
        logging.info(" > dataset split:")
        logging.info("     Total {} documents is : {} ".format(name, total_num_of_documents))

        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)

        dataset = cls(
            cfg=cfg,
            tokenizer=tokenizer,
            name=name,
            data_prefix=current_data_prefix,
            documents=np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            data=data_payload,
            seq_length=seq_length,
            seed=seed,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)
    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        data_prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(data_prefixes)):
            dataset = _build_dataset(data_prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples)


def build_train_valid_test_datasets(
    cls,
    cfg,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    tokenizer,
    n_chunks=None,
    n_examples_per_chunk=None,
):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data path is of type dictionary.")

        if isinstance(n_examples_per_chunk, DictConfig):
            train_examples_per_chunk = n_examples_per_chunk["train"]
            validation_examples_per_chunk = n_examples_per_chunk["validation"]
            test_examples_per_chunk = n_examples_per_chunk["test"]
        else:
            train_examples_per_chunk = n_examples_per_chunk
            validation_examples_per_chunk = n_examples_per_chunk
            test_examples_per_chunk = n_examples_per_chunk

        if isinstance(n_chunks, DictConfig):
            train_n_chunks = n_chunks["train"]
            validation_n_chunks = n_chunks["validation"]
            test_n_chunks = n_chunks["test"]
        else:
            train_n_chunks = n_chunks
            validation_n_chunks = n_chunks
            test_n_chunks = n_chunks

        train_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["train"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[0]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="train",
            n_chunks=train_n_chunks,
            n_examples_per_chunk=train_examples_per_chunk,
        )
        validation_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["validation"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[1]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="validation",
            n_chunks=validation_n_chunks,
            n_examples_per_chunk=validation_examples_per_chunk,
        )
        test_ds = build_dataset_generic(
            cls=cls,
            cfg=cfg,
            data_prefix=data_prefix["test"],
            data_impl=data_impl,
            num_samples=int(train_valid_test_num_samples[2]),
            seq_length=seq_length,
            seed=seed,
            tokenizer=tokenizer,
            name="test",
            n_chunks=test_n_chunks,
            n_examples_per_chunk=test_examples_per_chunk,
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cls=cls,
                cfg=cfg,
                data_prefix=data_prefix[0],
                data_impl=data_impl,
                splits_string=splits_string,
                train_valid_test_num_samples=train_valid_test_num_samples,
                seq_length=seq_length,
                seed=seed,
                tokenizer=tokenizer,
                n_chunks=n_chunks,
                n_examples_per_chunk=n_examples_per_chunk,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        data_prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(data_prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cls=cls,
                cfg=cfg,
                data_prefix=data_prefixes[i],
                data_impl=data_impl,
                splits_string=splits_string,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seq_length=seq_length,
                seed=seed,
                tokenizer=tokenizer,
                n_chunks=n_chunks,
                n_examples_per_chunk=n_examples_per_chunk,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cls,
    cfg,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    tokenizer,
    n_chunks=None,
    n_examples_per_chunk=None,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset or jsonl
    if data_impl == "mmap":
        data_payload = get_indexed_dataset_(data_prefix, data_impl, cfg.data.get("skip_warmup", True))
    elif data_impl.startswith("json"):
        with open(data_prefix, "r", encoding="utf_8") as fr:
            data_payload = [json.loads(line.strip()) for line in fr]
    elif data_impl == "packed_jsonl":
        data_payload = np.load(data_prefix, allow_pickle=True)
    elif data_impl == "chunked_jsonl":
        assert isinstance(n_chunks, int) and n_chunks >= 1, f"Not valid n_chunks {n_chunks}"
        data_payload = ChunkedJsonl(data_prefix, n_chunks, n_examples_per_chunk)
    else:
        raise RuntimeError(
            f"data.data_impl must be one of mmap, json, jsonl, packed_jsonl, or chunked_jsonl, but got {data_impl}"
        )
    total_num_of_documents = len(data_payload)
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(" > dataset split:")

    def print_split_stats(name, index):
        logging.info("    {}:".format(name))
        logging.info(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "validation":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = cls(
                cfg=cfg,
                tokenizer=tokenizer,
                name=name,
                data_prefix=data_prefix,
                documents=documents,
                data=data_payload,
                num_samples=train_valid_test_num_samples[index],
                seq_length=seq_length,
                seed=seed,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "validation")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


build_train_valid_test_rlhf_datasets = partial(build_train_valid_test_datasets, RLHFDataset)
build_train_valid_test_rm_datasets = partial(build_train_valid_test_datasets, RewardModelDataset)
build_train_valid_test_dpo_datasets = partial(build_train_valid_test_datasets, DPOModelDataset)
build_train_valid_test_dpo_packed_datasets = partial(build_train_valid_test_datasets, DPOPackedDataset)
build_train_valid_test_kto_datasets = partial(build_train_valid_test_datasets, KTOModelDataset)
build_train_valid_test_regression_rm_datasets = partial(build_train_valid_test_datasets, RegressionRewardModelDataset)
build_train_valid_test_knowledge_distillation_datasets = partial(
    build_train_valid_test_datasets, KnowledgeDistillationDataset
)


def build_sft_dataset(
    data_cfg, tokenizer, num_samples, answer_only_loss=True, is_chat=True, special_tokens=None, model_cfg=None
):
    packed_sequence = data_cfg.get("packed_sequence", False)
    dataset_kwargs = {}

    # TE requires that the first input dim is divisible by 8 and the second by 16 for fp8
    # When using sequence parallel, sequence will further be split by TP size
    # When using context parallel, sequence is split by CP size as well
    pad_seq_length_to_mult = 16
    if model_cfg is not None:
        pad_seq_length_to_mult = (
            8 * model_cfg.get("tensor_model_parallel_size", 1) if model_cfg.get("sequence_parallel", False) else 16
        )
        pad_seq_length_to_mult *= model_cfg.get("context_parallel_size", 1)

    if is_chat:
        assert not packed_sequence, "Sequence packing is currently not supported with chat datasets."
        dataset_cls = GPTSFTChatDataset
    elif packed_sequence:
        dataset_cls = GPTSFTPackedDataset
        # Whether to return `cu_seqlen` to pass to the model. Having `cu_seqlen` in the model input
        # enables THD attention kernel, which is the correct format for training with packed sequence to prevent
        # cross-sequence attention. This flag should be True unless you have a specific use case.
        dataset_kwargs = {"return_cu_seqlen": data_cfg.get("packed_sequence_return_cu_seqlen", True)}
        assert data_cfg.micro_batch_size == 1, "Micro batch size must be 1 if using packed sequence"
    else:
        dataset_cls = GPTSFTDataset

    dataset = dataset_cls(
        file_path=data_cfg.file_path,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        pad_seq_length_to_mult=pad_seq_length_to_mult,
        add_bos=data_cfg.get("add_bos", False),
        add_eos=data_cfg.get("add_eos", True),
        add_sep=data_cfg.get("add_sep", False),
        sep_id=0,
        max_num_samples=num_samples,
        seed=data_cfg.get("seed", 1234),
        label_key=data_cfg.get("label_key", "answer"),
        answer_only_loss=answer_only_loss,
        truncation_field=data_cfg.get("truncation_field", "text"),
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        index_mapping_dir=data_cfg.get("index_mapping_dir", None),
        prompt_template=data_cfg.get("prompt_template", None),
        virtual_tokens=0,
        memmap_workers=data_cfg.get(
            "memmap_workers", None
        ),  # used to set num. of workers to create the memmap index files
        hf_dataset=data_cfg.get(
            "hf_dataset", False
        ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        truncation_method=data_cfg.get(
            "truncation_method", "right"
        ),  # used to choose truncation method. Options: ['random', 'left', 'right']
        special_tokens=special_tokens,
        output_original_text=data_cfg.get("output_original_text", False),
        **dataset_kwargs,
    )
    return dataset


def collate_with_pad_to_max_batch(max_seqlen, tokenizer_eos_id, cfg, generate_masks_and_position_ids=True):
    """collate function that pads each sequence to the max in the batch"""
    return partial(
        collate_with_batch_max_sequence_length,
        response_token_length=max_seqlen,
        eos_id=tokenizer_eos_id,
        reset_position_ids=cfg.model.data.get("reset_position_ids", False),
        reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
        eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        generate_masks_and_position_ids=generate_masks_and_position_ids,
    )


def identity_collate(batch):
    """
    Useful since torch's data loader's default collate will crash with ragged sequences.
    Also, this function is needed b/c lambda functions aren't pickle-able.
    """
    return batch


def build_dataloader(
    cfg,
    dataset,
    consumed_samples,
    mbs,
    gbs,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
    collate_fn=None,
    load_gbs=True,
    use_random_sampler=True,
):
    """Buld dataloader given an input dataset."""

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = MegatronPretrainingRandomBatchSampler if load_gbs else MegatronPretrainingRandomSampler
        common_params["seed"] = cfg.model.seed
    else:
        cls = MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
