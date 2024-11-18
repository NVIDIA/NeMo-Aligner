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

import os

import pytest
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo_aligner.models.nlp.gpt.megatron_gpt_ppo_actor import MegatronGPTActorModel
from nemo_aligner.testing.utils import Utils
from nemo_aligner.utils.train_script_utils import init_distributed, resolve_and_create_trainer

dir_path = os.path.dirname(os.path.abspath(__file__))
# TODO: This file exists because in cases where TRTLLM MPI communicators are involved,
#  the cleanup of the communicators can trigger a segfault at the end of a pytest
#  run that gives a false negative when all tests passes. Instead we will use a file
#  as a marker that the test has succeeded.
SUCCESS_FILE = os.path.join(dir_path, "PYTEST_SUCCESS")


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    """
    parser.addoption(
        "--cpu", action="store_true", help="pass that argument to use CPU during testing (DEFAULT: False = GPU)"
    )
    parser.addoption("--mpi", action="store_true", default=False, help="Run only MPI tests")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )


@pytest.fixture
def device(request):
    """ Simple fixture returning string denoting the device [CPU | GPU] """
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    if request.node.get_closest_marker("run_only_on"):
        if request.node.get_closest_marker("run_only_on").args[0] != device:
            pytest.skip("skipped on this device: {}".format(device))


@pytest.fixture
def init_model_parallel():
    from nemo_aligner.testing.utils import Utils

    def initialize(*args, **kwargs):
        Utils.initialize_model_parallel(*args, **kwargs)

    # Yield the initialized function, which is available to the test
    yield initialize

    # Teardown: Called when the test ends
    Utils.destroy_model_parallel()


@pytest.fixture
def llama3_tokenizer():
    return AutoTokenizer("meta-llama/Meta-Llama-3-8b")


@pytest.fixture
def dummy_gpt_model(init_model_parallel):
    init_model_parallel(1, 1)

    model_cfg = {
        "precision": 32,
        "micro_batch_size": 4,
        "global_batch_size": 8,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "resume_from_checkpoint": None,
        "encoder_seq_length": 512,
        "max_position_embeddings": 512,
        "num_layers": 1,
        "hidden_size": 128,
        "ffn_hidden_size": 512,
        "num_attention_heads": 2,
        "init_method_std": 0.02,
        "hidden_dropout": 0.1,
        "kv_channels": None,
        "apply_query_key_layer_scaling": True,
        "layernorm_epsilon": 1e-5,
        "make_vocab_size_divisible_by": 128,
        "pre_process": True,
        "post_process": True,
        "persist_layer_norm": True,
        "gradient_as_bucket_view": True,
        "tokenizer": {"library": "huggingface", "type": "meta-llama/Meta-Llama-3-8B", "use_fast": True,},
        "native_amp_init_scale": 4294967296,
        "native_amp_growth_interval": 1000,
        "hysteresis": 2,
        "fp32_residual_connection": False,
        "fp16_lm_cross_entropy": False,
        "megatron_amp_O2": False,
        "seed": 1234,
        "use_cpu_initialization": False,
        "onnx_safe": False,
        "apex_transformer_log_level": 30,
        "activations_checkpoint_method": None,
        "activations_checkpoint_num_layers": 1,
        "data": {
            "data_prefix": "???",
            "index_mapping_dir": None,
            "data_impl": "mmap",
            "splits_string": "900,50,50",
            "seq_length": 512,
            "skip_warmup": True,
            "num_workers": 2,
            "dataloader_type": "single",
            "reset_position_ids": False,
            "reset_attention_mask": False,
            "eod_mask_loss": False,
        },
        "optim": {
            "name": "fused_adam",
            "lr": 2e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "sched": {"name": "CosineAnnealing", "warmup_steps": 500, "constant_steps": 50000, "min_lr": "2e-5"},
        },
    }

    trainer_cfg = {
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "gpu",
        "precision": 32,
        "logger": False,
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "max_epochs": 1000,
        "max_steps": 100000,
        "log_every_n_steps": 10,
        "val_check_interval": 100,
        "limit_val_batches": 50,
        "limit_test_batches": 500,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
    }

    strategy = NLPDDPStrategy()
    trainer = Trainer(strategy=strategy, **trainer_cfg)
    cfg = DictConfig(model_cfg)
    model = MegatronGPTModel(cfg=cfg, trainer=trainer)
    yield model


@pytest.fixture
def dummy_actor_gpt_model_with_pp():

    model_cfg = {
        "precision": "bf16-mixed",
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "resume_from_checkpoint": None,
        "encoder_seq_length": 8192,
        "max_position_embeddings": 8192,
        "num_layers": 2,
        "hidden_size": 128,
        "ffn_hidden_size": 448,
        "num_attention_heads": 4,
        "init_method_std": 0.01,
        "hidden_dropout": 0.0,
        "kv_channels": None,
        "apply_query_key_layer_scaling": True,
        "layernorm_epsilon": 1e-5,
        "make_vocab_size_divisible_by": 128,
        "pre_process": True,
        "post_process": True,
        "persist_layer_norm": True,
        "gradient_as_bucket_view": True,
        "tokenizer": {"library": "huggingface", "type": "meta-llama/Meta-Llama-3-8B", "use_fast": True,},
        "native_amp_init_scale": 4294967296,
        "native_amp_growth_interval": 1000,
        "hysteresis": 2,
        "fp32_residual_connection": False,
        "fp16_lm_cross_entropy": False,
        "megatron_amp_O2": True,
        "seed": 1234,
        "use_cpu_initialization": False,
        "onnx_safe": False,
        "apex_transformer_log_level": 30,
        "activations_checkpoint_method": None,
        "activations_checkpoint_num_layers": None,
        "data": {
            "data_impl": "mmap",
            "splits_string": "99990,8,2",
            "seq_length": 8192,
            "skip_warmup": True,
            "num_workers": 2,
            "dataloader_type": "single",
            "reset_position_ids": True,
            "reset_attention_mask": True,
            "eod_mask_loss": False,
            "index_mapping_dir": None,
            "data_prefix": [0.99, "/train-data",],
        },
        "optim": {
            "name": "distributed_fused_adam",
            "lr": 0.0001,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "bucket_cap_mb": 125,
            "overlap_grad_sync": True,
            "overlap_param_sync": True,
            "contiguous_grad_buffer": True,
            "contiguous_param_buffer": True,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 500,
                "constant_steps": 0,
                "min_lr": 1e-5,
                "max_steps": 2,
            },
            "grad_sync_dtype": "bf16",
        },
        "mcore_gpt": True,
        "rampup_batch_size": None,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "num_query_groups": 2,
        "use_scaled_init_method": True,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "normalization": "rmsnorm",
        "do_layer_norm_weight_decay": False,
        "bias": False,
        "activation": "fast-swiglu",
        "headscale": False,
        "transformer_block_type": "pre_ln",
        "openai_gelu": False,
        "normalize_attention_scores": True,
        "position_embedding_type": "rope",
        "rotary_percentage": 1.0,
        "apply_rope_fusion": True,
        "cross_entropy_loss_fusion": True,
        "attention_type": "multihead",
        "share_embeddings_and_output_weights": False,
        "grad_allreduce_chunk_size_mb": 125,
        "grad_div_ar_fusion": True,
        "gradient_accumulation_fusion": True,
        "bias_activation_fusion": True,
        "bias_dropout_add_fusion": True,
        "masked_softmax_fusion": True,
        "sync_batch_comm": False,
        "num_micro_batches_with_partial_activation_checkpoints": None,
        "activations_checkpoint_layers_per_pipeline": None,
        "sequence_parallel": False,
        "deterministic_mode": False,
        "transformer_engine": True,
        "fp8": False,
        "fp8_e4m3": False,
        "fp8_hybrid": False,
        "fp8_margin": 0,
        "fp8_interval": 1,
        "fp8_amax_history_len": 1024,
        "fp8_amax_compute_algo": "max",
        "ub_tp_comm_overlap": False,
        "use_flash_attention": True,
        "gc_interval": 2,
        "nsys_profile": {
            "enabled": False,
            "trace": ["nvtx", "cuda"],
            "start_step": 10,
            "end_step": 10,
            "ranks": [0],
            "gen_shape": False,
        },
        "dist_ckpt_format": "zarr",
        "dist_ckpt_load_on_device": True,
        "dist_ckpt_parallel_save": False,
        "target": "nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel",
        "nemo_version": "2.0.0rc1",
        "ppo": {
            "rollout_micro_batch_size": 8,
            "num_rollout_samples": 512,
            "forward_micro_batch_size": 2,
            "val_rollout_micro_batch_size": 2,
            "num_val_samples": 1,
            "offload_adam_states": True,
            "entropy_bonus": 0.0,
            "ratio_eps": 0.2,
            "sampling_params": {
                "use_greedy": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "add_BOS": False,
                "all_probs": False,
                "compute_logprob": False,
                "end_strings": ["<|endoftext|>", "<extra_id_1>"],
            },
            "length_params": {"max_length": 512, "min_length": 1},
        },
    }

    trainer_cfg = {
        "devices": 2,
        "num_nodes": 1,
        "accelerator": "gpu",
        "precision": 32,
        "logger": False,
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "max_epochs": 1000,
        "max_steps": 100000,
        "log_every_n_steps": 10,
        "val_check_interval": 100,
        "limit_val_batches": 50,
        "limit_test_batches": 500,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
        "ppo": {
            "critic_warmup_steps": 0,
            "max_epochs": 1,
            "max_steps": -1,
            "val_check_interval": 10,
            "save_interval": 10,
            "gradient_clip_val": 1.0,
            "initial_policy_kl_penalty": 0.01,
            "use_absolute_kl": True,
            "discount_factor": 1.0,
            "gae_lambda": 0.95,
            "normalize_advantages": True,
            "rollout_batch_seq_length": None,
            "trt_llm": {
                "enable": True,
                "reshard": False,
                "max_input_len": 256,
                "seed": 42,
                "model_type": "llama",
                "unload_engine_train": True,
            },
        },
    }
    cfg = {
        "trainer": trainer_cfg,
        "model": model_cfg,
    }
    cfg = DictConfig(cfg)

    trainer = resolve_and_create_trainer(cfg, "ppo")
    model = MegatronGPTActorModel(cfg=cfg.model, trainer=trainer)
    init_distributed(trainer, model, cfg.model.get("transformer_engine", False))
    yield model
    Utils.destroy_model_parallel()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line("markers", "mpi(reason=None): marks tests as requiring MPI")


def pytest_collection_modifyitems(config, items):
    run_mpi_tests = config.getoption("--mpi")

    # Skip all mpi tests if --mpi is not provided
    if not run_mpi_tests:
        skip_mpi = pytest.mark.skip(reason="Skipping MPI test: --mpi option not provided")
        for item in items:
            if "mpi" in item.keywords:
                item.add_marker(skip_mpi)
    else:
        # If --mpi is provided, only run mpi tests, skip all others
        skip_non_mpi = pytest.mark.skip(reason="Skipping non-MPI test: --mpi option provided")
        for item in items:
            if "mpi" not in item.keywords:
                item.add_marker(skip_non_mpi)


def pytest_sessionstart(session):
    # Remove the file at the start of the session, if it exists
    if os.path.exists(SUCCESS_FILE) and (
        os.environ.get("LOCAL_RANK", None) == "0" or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", None) == "0"
    ):
        os.remove(SUCCESS_FILE)


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    import torch

    # After the test session completes, destroy the NCCL process group. This suppresses a NCCL warning from pytorch>=2.4
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if exitstatus == 0:
        with open(SUCCESS_FILE, "w") as f:
            ...
