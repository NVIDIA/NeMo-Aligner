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
from functools import partial
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch.distributed
from omegaconf import OmegaConf

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo_aligner.algorithms.dpo import dpo_custom_collate
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_dpo_datasets,
    build_train_valid_test_dpo_packed_datasets,
)
from nemo_aligner.utils import parallel_state


@pytest.fixture
def llama3_tokenizer():
    return AutoTokenizer("meta-llama/Meta-Llama-3-8b")


@pytest.fixture
def str_to_list_tokenizer():
    class StringToListTokenizer:
        eos_id: int = -1

        def text_to_ids(self, text: str) -> list[int]:
            return [int(x) for x in text.split()]

    return StringToListTokenizer()


@pytest.fixture
def make_tmp_jsonl():
    with TemporaryDirectory() as tmp_dir:

        def write_jsonl(jsonl: list[dict], prefix="tmp"):
            jsonl_path = f"{tmp_dir}/{prefix}.jsonl"
            with open(jsonl_path, "w") as f:
                for obj in jsonl:
                    f.write(json.dumps(obj) + "\n")
            return jsonl_path

        yield write_jsonl


@pytest.mark.run_only_on("GPU")
def test_dpo_loader(init_model_parallel, make_tmp_jsonl, llama3_tokenizer):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    tmp_jsonl = make_tmp_jsonl(
        [
            {
                "prompt": f"<extra_id_0>System\n\n<extra_id_1>User\n{'+'.join('1'*i)}={i}?\n<extra_id_1>Assistant\n",
                "chosen_response": f"yes\n<extra_id_1>",
                "rejected_response": f"no\n<extra_id_1>",
            }
            for i in range(1, 100, 10)
        ]
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "data": {
                    "data_prefix": {"train": [tmp_jsonl], "validation": [tmp_jsonl], "test": [tmp_jsonl]},
                    "splits_string": None,
                    "num_workers": 2,
                },
                "seed": 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()

    train_ds, _, _ = build_train_valid_test_dpo_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl="jsonl",
        splits_string=None,
        train_valid_test_num_samples=[-1 * gbs] * 3,
        seq_length=1024,
        seed=cfg.model.seed,
        tokenizer=llama3_tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=mbs,
        gbs=gbs,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=lambda x: x,
    )

    distributed_collate_fn = partial(
        dpo_custom_collate,
        eos_id=llama3_tokenizer.eos_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        mbatch = distributed_collate_fn(mbatch)
        padded_seq_len = mbatch["chosen"].shape[1]
        for in_name, in_tensor in mbatch.items():
            assert in_tensor.shape[0] == minibs, f"Expected {in_name}.shape={in_tensor.shape} first dim to be {minibs}"

        assert mbatch["chosen"].shape == (minibs, padded_seq_len)
        assert mbatch["rejected"].shape == (minibs, padded_seq_len)
        assert mbatch["chosen_length"].shape == (minibs,)
        assert mbatch["rejected_length"].shape == (minibs,)
        assert mbatch["chosen_labels"].shape == (minibs, padded_seq_len)
        assert mbatch["rejected_labels"].shape == (minibs, padded_seq_len)
        assert mbatch["attention_mask"].shape == (minibs, 1, padded_seq_len, padded_seq_len)
        assert mbatch["position_ids"].shape == (minibs, padded_seq_len)
        assert mbatch["chosen_rewards"].shape == (minibs,)
        assert mbatch["rejected_rewards"].shape == (minibs,)
        num_mini_batches += 1
    assert num_mini_batches == 2


@pytest.mark.run_only_on("GPU")
def test_dpo_loader_original(init_model_parallel, make_tmp_jsonl, llama3_tokenizer):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    tmp_jsonl = make_tmp_jsonl(
        [
            {
                "prompt": f"<extra_id_0>System\n\n<extra_id_1>User\n{'+'.join('1'*i)}={i}?\n<extra_id_1>Assistant\n",
                "chosen_response": f"yes\n<extra_id_1>",
                "rejected_response": f"no\n<extra_id_1>",
            }
            for i in range(1, 100, 10)
        ]
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "data": {
                    "data_prefix": {"train": [tmp_jsonl], "validation": [tmp_jsonl], "test": [tmp_jsonl]},
                    "splits_string": None,
                    "num_workers": 2,
                },
                "seed": 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()

    train_ds, _, _ = build_train_valid_test_dpo_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl="jsonl",
        splits_string=None,
        train_valid_test_num_samples=[-1 * gbs] * 3,
        seq_length=1024,
        seed=cfg.model.seed,
        tokenizer=llama3_tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=mbs,
        gbs=gbs,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=lambda x: x,
    )

    distributed_collate_fn = partial(
        dpo_custom_collate,
        eos_id=llama3_tokenizer.eos_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        mbatch = distributed_collate_fn(mbatch)
        padded_seq_len = mbatch["chosen"].shape[1]
        for in_name, in_tensor in mbatch.items():
            assert in_tensor.shape[0] == minibs, f"Expected {in_name}.shape={in_tensor.shape} first dim to be {minibs}"

        assert mbatch["chosen"].shape == (minibs, padded_seq_len)
        assert mbatch["rejected"].shape == (minibs, padded_seq_len)
        assert mbatch["chosen_length"].shape == (minibs,)
        assert mbatch["rejected_length"].shape == (minibs,)
        assert mbatch["chosen_labels"].shape == (minibs, padded_seq_len)
        assert mbatch["rejected_labels"].shape == (minibs, padded_seq_len)
        assert mbatch["attention_mask"].shape == (minibs, 1, padded_seq_len, padded_seq_len)
        assert mbatch["position_ids"].shape == (minibs, padded_seq_len)
        assert mbatch["chosen_rewards"].shape == (minibs,)
        assert mbatch["rejected_rewards"].shape == (minibs,)
        num_mini_batches += 1
    assert num_mini_batches == 2


@pytest.mark.run_only_on("GPU")
def test_dpo_loader_pad_to_multiple(init_model_parallel, make_tmp_jsonl, str_to_list_tokenizer):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    tmp_jsonl = make_tmp_jsonl(
        [
            {
                "prompt": f"{' '.join(str(x) for x in range(i))} ",
                "chosen_response": f"{i * 10}",
                "rejected_response": f"{i * 100}",
            }
            for i in range(1, 100, 10)
        ]
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "data": {
                    "data_prefix": {"train": [tmp_jsonl], "validation": [tmp_jsonl], "test": [tmp_jsonl]},
                    "splits_string": None,
                    "num_workers": 2,
                },
                "seed": 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()
    expected_seq_len_multiple = 29  # pick a prime to make sure

    train_ds, _, _ = build_train_valid_test_dpo_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl="jsonl",
        splits_string=None,
        train_valid_test_num_samples=[-1 * gbs] * 3,
        seq_length=1024,
        seed=cfg.model.seed,
        tokenizer=str_to_list_tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=mbs,
        gbs=gbs,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=lambda x: x,
    )

    distributed_collate_fn = partial(
        dpo_custom_collate,
        eos_id=str_to_list_tokenizer.eos_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        pad_length_to_multiple_of=expected_seq_len_multiple,
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        chosen_lengths = [len(x["chosen"]) for x in mbatch]
        rejected_lengths = [len(x["rejected"]) for x in mbatch]
        assert chosen_lengths == rejected_lengths

        assert len(set(chosen_lengths)) == len(
            chosen_lengths
        ), f"Lengths should be unique in this test: {chosen_lengths=}"

        mbatch = distributed_collate_fn(mbatch)
        assert mbatch["chosen"].shape[1] % expected_seq_len_multiple == 0
        assert mbatch["rejected"].shape[1] % expected_seq_len_multiple == 0
        assert mbatch["chosen_labels"].shape[1] % expected_seq_len_multiple == 0
        assert mbatch["rejected_labels"].shape[1] % expected_seq_len_multiple == 0
        assert mbatch["attention_mask"].shape[2] % expected_seq_len_multiple == 0
        assert mbatch["attention_mask"].shape[3] % expected_seq_len_multiple == 0
        assert mbatch["position_ids"].shape[1] % expected_seq_len_multiple == 0

        # Check that all ranks have the same length
        max_chosen_seq_length = torch.tensor(mbatch["chosen"].shape[1], device="cuda")
        torch.distributed.all_reduce(
            max_chosen_seq_length, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group()
        )
        assert mbatch["chosen"].shape[1] == max_chosen_seq_length.item()

        num_mini_batches += 1

    assert num_mini_batches == 2





@pytest.mark.run_only_on("GPU")
def test_packed_dpo_loader(init_model_parallel, make_tmp_jsonl, llama3_tokenizer):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    np_data = np.array([
        {
            "input_ids": np.ones(15),
            "labels": np.concatenate((-100 * np.ones(7), np.ones(8))),
            "reward": np.ones(4),
            "lengths": [5, 3, 4, 3],
            "seq_boundaries": [0, 5, 8, 12, 15],
        },
    ] * 8)

    tmp_dir = TemporaryDirectory()
    data_path = f"{tmp_dir.name}/data.npy"
    np.save(data_path, np_data)

    cfg = OmegaConf.create(
        {
            "model": {
                "data": {
                    "data_prefix": {"train": [data_path], "validation": [data_path], "test": [data_path]},
                    "splits_string": None,
                    "num_workers": 2,
                },
                "seed": 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()

    train_ds, _, _ = build_train_valid_test_dpo_packed_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl="packed_jsonl",
        splits_string=None,
        train_valid_test_num_samples=[-1 * gbs] * 3,
        seq_length=1024,
        seed=cfg.model.seed,
        tokenizer=llama3_tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=mbs,
        gbs=gbs,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=lambda x: x,
    )

    distributed_collate_fn = partial(
        train_ds.global_collate_fn,
        eos_id=llama3_tokenizer.eos_id,
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        mbatch = distributed_collate_fn(mbatch)
        padded_seq_len = mbatch["input_ids"].shape[1]
        for in_name, in_tensor in mbatch.items():
            assert in_tensor.shape[0] == minibs, f"Expected {in_name}.shape={in_tensor.shape} first dim to be {minibs}"

        assert mbatch["input_ids"].shape == (minibs, padded_seq_len)
        assert mbatch["labels"].shape == (minibs, padded_seq_len)
        assert mbatch["lengths"].shape == (minibs,len(np_data[0]["lengths"]))
        assert mbatch["rewards"].shape == (minibs,len(np_data[0]["lengths"]))
        ### last cu_seqlen set to max_length, the we add one padding element which gets removed during training
        assert torch.equal(mbatch["cu_seqlens"][0], torch.tensor([0, 4, 6, 9, 16, -1]))
        assert mbatch["cu_seqlens_argmin"][0] == torch.tensor([5])
        ### this will end up being the final example because it's padded
        ### should be fine because final padding tokens are not included in the loss
        assert mbatch["max_seqlen"][0] == torch.tensor([7])

        num_mini_batches +=1

    assert num_mini_batches == 2

    tmp_dir.cleanup()


@pytest.mark.run_only_on("GPU")
def test_packed_dpo_loader_pad_to_multiple(init_model_parallel, make_tmp_jsonl, str_to_list_tokenizer):
    init_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    np_data = np.array([
        {
            "input_ids": np.ones(15),
            "labels": np.concatenate((-100 * np.ones(7), np.ones(8))),
            "reward": np.ones(8),
            "lengths": [5, 3, 4, 3],
            "seq_boundaries": [0, 5, 8, 12, 15],
        },
    ] * 8)

    tmp_dir = TemporaryDirectory()
    data_path = f"{tmp_dir.name}/data.npy"
    np.save(data_path, np_data)

    cfg = OmegaConf.create(
        {
            "model": {
                "data": {
                    "data_prefix": {"train": [data_path], "validation": [data_path], "test": [data_path]},
                    "splits_string": None,
                    "num_workers": 2,
                },
                "seed": 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()
    expected_seq_len_multiple = 29  # pick a prime to make sure

    train_ds, _, _ = train_ds, _, _ = build_train_valid_test_dpo_packed_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl="packed_jsonl",
        splits_string=None,
        train_valid_test_num_samples=[-1 * gbs] * 3,
        seq_length=1024,
        seed=cfg.model.seed,
        tokenizer=str_to_list_tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=0,
        mbs=mbs,
        gbs=gbs,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=lambda x: x,
    )

    distributed_collate_fn = partial(
        train_ds.global_collate_fn,
        eos_id=str_to_list_tokenizer.eos_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        pad_length_to_multiple_of=expected_seq_len_multiple,
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:

        mbatch = distributed_collate_fn(mbatch)
        for k in ["input_ids", "labels", "position_ids"]:
            assert mbatch[k].shape[1] % expected_seq_len_multiple == 0

        # Check that all ranks have the same length
        max_chosen_seq_length = torch.tensor(mbatch["input_ids"].shape[1], device="cuda")
        torch.distributed.all_reduce(
            max_chosen_seq_length, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group()
        )
        assert mbatch["input_ids"].shape[1] == max_chosen_seq_length.item()

        num_mini_batches += 1

    assert num_mini_batches == 2

    tmp_dir.cleanup()
