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

import threading

import torch
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.models.mm.mgpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.servers.server_callables import RewardModelCallable
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype
from nemo_aligner.utils.text_generation_utils import MGPTModelTextGenerationStrategy


"""PyTriton Based Inference Server for the Reward Model"""

ENDPOINT_BIND_ADDRESS = "0.0.0.0"

# Not sure if this is needed
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

#mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="inference_rm")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.rm_model_file, cfg.model)
    
    # We need this to looking for the base LLM path, which is not needed
    with open_dict(cfg):
        cfg.model.mm_cfg.llm.from_pretrained = None
        cfg.model.mm_cfg.llm.freeze = False   # Note, setting it to True at inference time causes wrong regression model outputs

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    # needed for autocasting BF16
    set_autocast_gpu_dtype(cfg.trainer.precision)
    if trainer.precision == "16":
        cfg.model.megatron_amp_O2 = False
    elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
        cfg.model.megatron_amp_O2 = True

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]
    print(f"Loading reward model of type {reward_model_type}, class {reward_model_cls}")

    print(f"Loading reward model from {cfg.rm_model_file}, class {reward_model_cls}")
    ptl_model = load_from_nemo(reward_model_cls, cfg.model, trainer, 
                               #strict=True, 
                                strict=False,
                                load_base_model_only=True,

                               restore_path=cfg.rm_model_file,)

    prompt = """<extra_id_0>System

<extra_id_1>User
Explain, how did the invention of the train change society?
<extra_id_1>Assistant
The invention of the train had a profound impact on society, revolutionizing transportation and opening up new opportunities for economic growth and social change. The train allowed people to travel long distances in a relatively short amount of time, making it easier to commute to work, travel for leisure, and move goods and resources across the country. This increased mobility led to the growth of cities and the development of new industries, as people and goods could now be transported more efficiently. The train also played a significant role in the expansion of the United States, as it allowed people to move west and settle new territories. The train also had a profound impact on social change, as it allowed people to travel more easily and meet new people from different regions and backgrounds. This increased interaction led to the growth of new ideas and the spread of new cultural practices, leading to a more diverse and interconnected society. Overall, the invention of the train had a significant impact on society, leading to greater mobility, economic growth, and social change.
<extra_id_2>"""
    # Text generation strategy
    # TODO: figure out why it's not working, strategy is specified in the config
    strategy = MGPTModelTextGenerationStrategy(ptl_model)
    ptl_model.strategy = strategy

    print("calling inference after load_from_nemo:", ptl_model.infer([prompt]))


    ptl_model.freeze()



    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()


    dp_size = parallel_state.get_data_parallel_world_size()
    max_batch_size = cfg.inference.micro_batch_size * dp_size

    infer_fn = ptl_model.infer
    ptl_model.prepare_for_inference()

    print("calling inference after prepare_for_inference:", ptl_model.infer([prompt]))





    if torch.distributed.get_rank() == 0:
        infer_callable = RewardModelCallable(model_name="reward_model", infer_fn=infer_fn, lock=threading.Lock())
        triton_config = TritonConfig(
            allow_http=True,
            allow_grpc=False,
            allow_metrics=False,
            http_address=ENDPOINT_BIND_ADDRESS,
            http_port=cfg.inference.port,
            exit_timeout_secs=1.0
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
    with torch.no_grad():
        main()
