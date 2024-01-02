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
import asyncio
from contextlib import contextmanager
import os
import threading
from functools import partial

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo_aligner.utils.deep_search.search_callables import SearchCallable
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.deep_search.text_generation_strategy import GPTModelTextGenerationStrategy

from nemo.core.config import hydra_runner
from nemo.utils import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.utils import logging
import datetime
import time
from nemo_aligner.utils.train_script_utils import (
    init_distributed,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    megatron_gpt_generate,
)


try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

"""Inference server for NeMo-RLHF Initial Policy model.
"""

if not torch.cuda.is_available():
    raise OSError("GPU is needed for the inference")

ENDPOINT_BIND_ADDRESS = "0.0.0.0"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)8s - %(process)8d - %(threadName)s - %(name)s: %(message)s"

def remove_padded_prompts(response, nb_paddings):
    result = {}
    for k, v in response.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result

@contextmanager
def timer(name):
    beg = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - beg} seconds")


def print_gpu_memory_usage():
    # print the GPU memory ussage by all devices, print out in GB
#     for device in range(torch.cuda.device_count()):
#         print(f"GPU {device} memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3}")
#         print(f"GPU {device} memory cached: {torch.cuda.memory_cached(device) / 1024 ** 3}")
#         print(f"GPU {device} memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3}")
#         print(f"GPU {device} memory allocated max: {torch.cuda.max_memory_allocated(device) / 1024 ** 3}")
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            # Set the device to the GPU
            torch.cuda.device(i)
            # Get the memory allocated for the current GPU
            allocated_memory = torch.cuda.memory_allocated(i)
            print(f"GPU {i}: Allocated memory: {allocated_memory / 1024**3} GB")
    else:
        print("CUDA is not available. You might be running on a CPU.")

def init_distributed_parameters(trainer, cfg):
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        if trainer.num_devices and trainer.num_nodes:
            app_state.world_size = trainer.num_devices * trainer.num_nodes
 


@hydra_runner(config_path="conf", config_name="gpt_inference")
def main(cfg) -> None:
    logging.info("\n\n************** Inference configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"
    print_gpu_memory_usage()

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if pretrained_cfg.get('mcore_gpt', False):
                # with dist checkpointing we can use the model parallel config specified by the user
                pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ['bf16', 'bf16-mixed'] and cfg.get('megatron_amp_O2', False):
                pretrained_cfg.megatron_amp_O2 = True
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        # checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        #trainer._checkpoint_connector.restore(checkpoint_path)
        # trainer._checkpoint_connector._restore_modules_and_callbacks(cfg.checkpoint_dir)
        init_distributed_parameters(trainer, cfg)
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        # overwrite parameters
        overwrite_cfg = {
            "sequence_parallel": False,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_method": None,
            "precision": trainer.precision,
            "tensor_model_parallel_size": cfg.tensor_model_parallel_size,
            "pipeline_model_parallel_size": cfg.pipeline_model_parallel_size,
            "megatron_amp_O2": cfg.get('megatron_amp_O2', False),
            "tokenizer": cfg.tokenizer,
        }
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=None, trainer=trainer, **overwrite_cfg)
        # model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
        # # the world_size is set to 1, because it read from trainer.world_size when loading the checkponts        
        # init_distributed_parameters(trainer, cfg)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    print_gpu_memory_usage()

    init_distributed(trainer, model, False)
    print_gpu_memory_usage()

    dp_size = parallel_state.get_data_parallel_world_size()
    max_batch_size = cfg.inference.micro_batch_size * dp_size


    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    # strategy_args = {"strategy": strategy}
    strategy = GPTModelTextGenerationStrategy(model)
    strategy_args = {"strategy": strategy}

    prompts = [
        "What is the meaning of life?",
        "How many stars are in the sky?",
    ]


    response = megatron_gpt_generate(
        model, prompts, model.tokenizer, length_params, sampling_params, **strategy_args
    )

    # if fp8_enabled:
    #     response = remove_padded_prompts(response, nb_paddings)
    print("***************************")
    print(response)
    print("***************************")


    def get_infer_fn(model, length_params, sampling_params, **strategy_args):
        def infer_fn(inputs):
            return megatron_gpt_generate(model, inputs, model.tokenizer, length_params, sampling_params, **strategy_args) 
        return infer_fn
    
    infer_fn = get_infer_fn(model, length_params, sampling_params, **strategy_args)

    if torch.distributed.get_rank() == 0:
        infer_callable = SearchCallable(model_name="reward_model", infer_fn=infer_fn, lock=threading.Lock())
        triton_config = TritonConfig(
            allow_http=True,
            allow_grpc=False,
            allow_metrics=False,
            http_address=ENDPOINT_BIND_ADDRESS,
            http_port=cfg.inference.port,
        )
        dynamic_batcher = DynamicBatcher(max_queue_delay_microseconds=2000)
        model_config = ModelConfig(batching=True, max_batch_size=max_batch_size, batcher=dynamic_batcher)

        with Triton(config=triton_config) as triton:
            triton.bind(
                model_name=infer_callable.model_name,
                infer_func=infer_callable.infer,
                inputs=infer_callable.inputs,
                outputs=infer_callable.outputs,
                config=model_config,
            )
            triton.serve()
    else:
        while True:
            choice = ServerSignal.INVALID.cuda()
            torch.distributed.broadcast(choice, 0)
            if choice.item() == ServerSignal.FORWARD:
                infer_fn(['empty'])
            else:
                raise RuntimeError(f"Invalid operation: {choice.item()}")



if __name__ == "__main__":
    with torch.no_grad():
        main()  # noqa pylint: disable=no-value-for-parameter
