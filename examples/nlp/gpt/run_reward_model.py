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
from functools import partial

import jsonlines
import torch
from megatron.core import parallel_state
from pytorch_lightning.trainer.trainer import Trainer
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.servers.server_callables import RewardModelCallable
from nemo_aligner.utils.distributed import SyncTimer, broadcast_2d_tensor, rebalance_nd_tensor
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype

"""PyTriton Based Inference Server for the Reward Model"""

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


@hydra_runner(config_path="conf", config_name="inference_rm")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.rm_model_file, cfg.model)
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert os.path.exists(cfg.dataset_path), "{} does not exist".format(cfg.dataset_path)

    ds = []

    with jsonlines.open(cfg.dataset_path) as reader:
        for obj in reader:
            ds.append(obj["text"])

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
    dp_size = parallel_state.get_data_parallel_world_size()

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()

    infer_fn = ptl_model.infer
    ptl_model.prepare_for_inference()
    tokenizer = ptl_model.tokenizer

    def collate_fn(batch):
        tokens = [tokenizer.text_to_ids(s) for s in batch]
        max_len = max(len(x) for x in tokens)
        tokens, sequence_lengths = pad_batch(tokens, tokenizer.eos_id, max_len)

        return tokens, sequence_lengths

    dataloader = build_dataloader(
        cfg=cfg,
        dataset=ds,
        mbs=cfg.model.forward_mbs,
        gbs=cfg.model.forward_mbs * dp_size,
        collate_fn=collate_fn,
        load_gbs=True,
        use_random_sampler=False,
        consumed_samples=0,
        drop_last=False,
    )

    rewards_all = []
    for batch in tqdm(dataloader):
        batch = tuple(map(torch.as_tensor, batch))
        rewards = infer_fn(batch).flatten().tolist()
        rewards_all.extend(rewards)

    outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
    torch.distributed.all_gather_object(outputs, rewards_all, parallel_state.get_data_parallel_group())
    output_tensor = torch.as_tensor(outputs).flatten()
    print("### OUTPUT TENSOR SHAPE", output_tensor.shape)
    print("### OUTPUT TENSOR MEAN", output_tensor.mean())
    print("### OUTPUT TENSOR STD", output_tensor.std())


if __name__ == "__main__":
    with torch.no_grad():
        main()
