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
from pytorch_lightning.trainer.trainer import Trainer
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.servers.server_callables import RewardModelCallable
from nemo_aligner.utils.distributed import broadcast_2d_tensor, rebalance_nd_tensor
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype

"""PyTriton Based Inference Server for the Reward Model"""

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


def run_inference(infer_fn, inputs=None, extra=None):
    """only rank 0 has valid data
    """
    tokens, lengths = None, None
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        tokens = torch.as_tensor(inputs["inputs"], dtype=torch.long, device=torch.cuda.current_device())
        lengths = torch.as_tensor(inputs["sequence_length"], dtype=torch.long, device=torch.cuda.current_device())

    tokens = broadcast_2d_tensor(tokens, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank]
    lengths = broadcast_2d_tensor(lengths, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank].squeeze(-1)

    outputs = infer_fn(inputs=(tokens, lengths))

    rewards = outputs
    rewards = rebalance_nd_tensor(rewards, group=parallel_state.get_data_parallel_group()).squeeze().cpu().numpy()

    return rewards


@hydra_runner(config_path="conf", config_name="inference_rm")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.rm_model_file, cfg.model)

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    # needed for autocasting BF16
    set_autocast_gpu_dtype(cfg.trainer.precision)
    if trainer.precision == "16":
        cfg.model.megatron_amp_O2 = False
    elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
        cfg.model.megatron_amp_O2 = True

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    ptl_model = load_from_nemo(reward_model_cls, cfg.model, trainer, strict=True, restore_path=cfg.rm_model_file,)

    ptl_model.freeze()

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()

    dp_size = parallel_state.get_data_parallel_world_size()
    forward_micro_batch_size = cfg.model.forward_micro_batch_size

    infer_fn = ptl_model.infer
    ptl_model.prepare_for_inference()

    if torch.distributed.get_rank() == 0:
        infer_callable = RewardModelCallable(
            model_name="reward_model",
            infer_fn=infer_fn,
            tokenizer=ptl_model.tokenizer,
            lock=threading.Lock(),
            forward_micro_batch_size=forward_micro_batch_size,
        )
        triton_config = TritonConfig(
            allow_http=True,
            allow_grpc=False,
            allow_metrics=False,
            http_address=ENDPOINT_BIND_ADDRESS,
            http_port=cfg.inference.port,
        )
        dp_size = parallel_state.get_data_parallel_world_size()
        preferred_batch_size = [dp_size * forward_micro_batch_size * (i + 1) for i in range(1000)]

        dynamic_batcher = DynamicBatcher(max_queue_delay_microseconds=2000, preferred_batch_size=preferred_batch_size,)
        model_config = ModelConfig(batching=True, max_batch_size=9999999, batcher=dynamic_batcher)

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
                run_inference(infer_fn)
            else:
                raise RuntimeError(f"Invalid operation: {choice.item()}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
