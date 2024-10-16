import pytest
import torch.distributed
from tempfile import TemporaryDirectory
from omegaconf import OmegaConf
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
import json
from functools import partial
from nemo_aligner.algorithms.dpo import dpo_custom_collate

from nemo_aligner.data.nlp.builders import build_dataloader, build_train_valid_test_dpo_datasets


@pytest.fixture
def llama3_tokenizer():
    return AutoTokenizer("meta-llama/Meta-Llama-3-8b")


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
                'seed': 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()

    train_ds, validation_ds, test_ds = build_train_valid_test_dpo_datasets(
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
        collate_fn=partial(
            dpo_custom_collate,
            eos_id=llama3_tokenizer.eos_id,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        padded_seq_len = mbatch['chosen'].shape[1]
        for in_name, in_tensor in mbatch.items():
            assert in_tensor.shape[0] == minibs, f"Expected {in_name}.shape={in_tensor.shape} first dim to be {minibs}"

        assert mbatch['chosen'].shape == (minibs, padded_seq_len)
        assert mbatch['rejected'].shape == (minibs, padded_seq_len)
        assert mbatch['chosen_length'].shape == (minibs, )
        assert mbatch['rejected_length'].shape == (minibs, )
        assert mbatch['chosen_labels'].shape == (minibs, padded_seq_len)
        assert mbatch['rejected_labels'].shape == (minibs, padded_seq_len)
        assert mbatch['attention_mask'].shape == (minibs, 1, padded_seq_len, padded_seq_len)
        assert mbatch['position_ids'].shape == (minibs, padded_seq_len)
        assert mbatch['chosen_rewards'].shape == (minibs, )
        assert mbatch['rejected_rewards'].shape == (minibs, )
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
                'seed': 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()

    train_ds, _, _= build_train_valid_test_dpo_datasets(
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
        collate_fn=partial(
            dpo_custom_collate,
            eos_id=llama3_tokenizer.eos_id,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        padded_seq_len = mbatch['chosen'].shape[1]
        for in_name, in_tensor in mbatch.items():
            assert in_tensor.shape[0] == minibs, f"Expected {in_name}.shape={in_tensor.shape} first dim to be {minibs}"

        assert mbatch['chosen'].shape == (minibs, padded_seq_len)
        assert mbatch['rejected'].shape == (minibs, padded_seq_len)
        assert mbatch['chosen_length'].shape == (minibs, )
        assert mbatch['rejected_length'].shape == (minibs, )
        assert mbatch['chosen_labels'].shape == (minibs, padded_seq_len)
        assert mbatch['rejected_labels'].shape == (minibs, padded_seq_len)
        assert mbatch['attention_mask'].shape == (minibs, 1, padded_seq_len, padded_seq_len)
        assert mbatch['position_ids'].shape == (minibs, padded_seq_len)
        assert mbatch['chosen_rewards'].shape == (minibs, )
        assert mbatch['rejected_rewards'].shape == (minibs, )
        num_mini_batches += 1
    assert num_mini_batches == 2

@pytest.mark.run_only_on("GPU")
def test_dpo_loader_pad_to_multiple(init_model_parallel, make_tmp_jsonl, llama3_tokenizer):
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
                'seed': 42,
            }
        }
    )
    mbs = 1
    minibs = 2
    gbs = minibs * torch.distributed.get_world_size()
    expected_seq_len_multiple = 29  # pick a prime to make sure 

    train_ds, validation_ds, test_ds = build_train_valid_test_dpo_datasets(
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
        collate_fn=partial(
            dpo_custom_collate,
            eos_id=llama3_tokenizer.eos_id,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            pad_length_to_multiple_of=expected_seq_len_multiple,
        ),
    )

    num_mini_batches = 0
    for mbatch in train_dataloader:
        assert mbatch['chosen'].shape[1] % expected_seq_len_multiple == 0
        assert mbatch['rejected'].shape[1] % expected_seq_len_multiple == 0
        assert mbatch['chosen_labels'].shape[1] % expected_seq_len_multiple == 0
        assert mbatch['rejected_labels'].shape[1] % expected_seq_len_multiple == 0
        assert mbatch['attention_mask'].shape[2] % expected_seq_len_multiple == 0
        assert mbatch['attention_mask'].shape[3] % expected_seq_len_multiple == 0
        assert mbatch['position_ids'].shape[1] % expected_seq_len_multiple == 0
        num_mini_batches += 1
    assert num_mini_batches == 2