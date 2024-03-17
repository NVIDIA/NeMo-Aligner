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

import datetime
import threading

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.deep_search.search_callables import SearchCallable
from nemo_aligner.utils.deep_search.text_gen_utils import search
from nemo_aligner.utils.deep_search.text_generation_strategy import HybridGPTSearchTextGenerationStrategy
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="gpt_hybrid_infer")
def main(cfg) -> None:
    """
    Binary ranking reward models use comparison based objective similar to the one found in the
    InstructGPT paper: https://arxiv.org/pdf/2203.02155.pdf and have no explicit labels.
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    hybrid_model_cls = MegatronGPTHybridModel

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    # logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        hybrid_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=not cfg.pretrained_checkpoint.from_mcts_trained,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    # # pull values from checkpoint
    # trainer_restore_path = trainer.ckpt_path
    # if trainer_restore_path is not None:
    #     custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
    #     consumed_samples = custom_trainer_state_dict["consumed_samples"]
    # else:
    #     custom_trainer_state_dict = None
    #     consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    # ptl_model.save_to("hybrid_model.nemo")

    dp_size = parallel_state.get_data_parallel_world_size()
    max_batch_size = cfg.inference.micro_batch_size * dp_size

    # strategy_args = {"strategy": strategy}
    strategy = HybridGPTSearchTextGenerationStrategy(ptl_model)
    strategy_args = {"strategy": strategy}

    def get_infer_fn(model, top_k, max_depth, add_bos_token, **strategy_args):
        # one token at a time
        def infer_fn(inputs=None, action=None, context_ids=None, session_info=None):
            return search(
                model,
                inputs,
                action,
                context_ids,
                session_info,
                tokens_to_generate=max_depth,  # max search depth
                top_k=top_k,
                add_bos_token=add_bos_token,
                **strategy_args,
            )

        return infer_fn

    infer_fn = get_infer_fn(
        ptl_model, cfg.inference.top_k, cfg.inference.tokens_to_generate, cfg.inference.add_bos_token, **strategy_args
    )
    # infer_fn = lambda : None
    # r = infer_fn(["hello", "ok"], context_ids=['context1', 'context2'], session_info='session')
    # print(r)

    # import sys

    # sys.exit(0)

    if torch.distributed.get_rank() == 0:
        infer_callable = SearchCallable(model_name="search", infer_fn=infer_fn, lock=threading.Lock())
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
                infer_fn()
            else:
                raise RuntimeError(f"Invalid operation: {choice.item()}")


if __name__ == "__main__":
    main()
