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
from datetime import datetime, timedelta

import jsonlines
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import mpu
from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm
from utils import get_max_time_per_run, load_nemo_or_checkpoint, set_seed

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.algorithms.offline.random_sampler import RandomSampler
from nemo_aligner.data.nlp.offline.builders import build_data_loader, build_dataset
from nemo_aligner.utils.utils import set_autocast_gpu_dtype

if not torch.cuda.is_available():
    raise OSError("GPU is needed for the inference")


@hydra_runner(config_path="conf", config_name="megatron_random_sampler")
def main(cfg) -> None:
    set_seed(cfg.seed)

    # init start time
    max_time_per_run = cfg.get("max_time_per_run", None)
    if max_time_per_run:
        start_time = datetime.now()
        max_time_per_run = get_max_time_per_run(max_time_per_run)

    # needed for autocasting BF16
    set_autocast_gpu_dtype(cfg.trainer.precision)

    # init trainer, model, sampler
    trainer = Trainer(strategy=NLPDDPStrategy(timeout=timedelta(seconds=99999999)), **cfg.trainer)
    model = load_nemo_or_checkpoint(MegatronGPTModel, trainer, cfg)
    sampler = RandomSampler(trainer, model, cfg.inference)

    # load consumed_samples from checkpoint
    consumed_samples = 0
    ckpt_path = f"{cfg.output_file}_temp"
    os.makedirs(ckpt_path, exist_ok=True)

    dp_rank = mpu.get_data_parallel_rank()
    status_ckpt_path = f"{ckpt_path}/{dp_rank}.ckpt"
    if os.path.exists(status_ckpt_path):
        consumed_samples = torch.load(status_ckpt_path)["consumed_samples"]

    # init dataloader
    dataset = build_dataset(cfg.data, model.tokenizer)
    dataloader = build_data_loader(dataset, cfg.data, consumed_samples)

    global_rank = dist.get_rank()
    # is the first node in TP and PP group ?
    is_the_first_tp_pp_node = global_rank == mpu.get_tensor_model_parallel_src_rank() and mpu.is_pipeline_last_stage()
    # is the first node in TP, PP and DP group ?
    is_the_first_tp_pp_dp_node = is_the_first_tp_pp_node and global_rank == mpu.get_data_parallel_src_rank()

    # ckpt func
    def save_checkpoint(global_step, objs):
        if not len(objs):
            return
        if is_the_first_tp_pp_node:
            filename = f"{ckpt_path}/rank_{dp_rank}.json"
            with jsonlines.open(filename, mode="a") as writer:
                writer.write_all(objs)
                objs.clear()

            save_dict = {
                "consumed_samples": global_step * cfg.data.micro_batch_size * mpu.get_data_parallel_world_size()
            }
            print(f"Rank {dp_rank} - Save ckpt: {save_dict}")
            torch.save(save_dict, status_ckpt_path)

    # inference loop
    global_step = consumed_samples // mpu.get_data_parallel_world_size() // cfg.data.micro_batch_size
    if is_the_first_tp_pp_node:
        print(f"Rank {dp_rank} - Init global steps: {global_step}")
    objs = []
    checkpoint_interval = cfg.get("checkpoint_interval", None)

    for sample in tqdm(dataloader, desc="Generating responses...", disable=not is_the_first_tp_pp_dp_node):
        current_device = torch.cuda.current_device()
        input = (sample["input_ids"].to(current_device), sample["length"].to(current_device))

        # generate n samples per prompt for rejection sampling
        N = int(cfg.data.get("best_of_n", 1))
        for _ in range(N):
            response_list = sampler.generate(input)["sentences"]
            # ignore other ranks in TP and PP group
            if is_the_first_tp_pp_node:
                for i, data in enumerate(sample["data"]):
                    raw_input = data[cfg.data.input_key]  # raw_input may not contain "User:" or "Assistant:"
                    prompt = data["<prompt>"]  # prompt is "User: xxx Assistant: xxx"
                    response = response_list[i][len(prompt) :]
                    obj = {cfg.data.input_key: raw_input, cfg.data.output_key: response}
                    objs.append(obj)

        global_step += 1
        if checkpoint_interval:
            dist.barrier()

        # check max_time_per_run
        if max_time_per_run is not None and datetime.now() >= start_time + max_time_per_run:
            if is_the_first_tp_pp_dp_node:
                print(f"max_time_per_run {max_time_per_run} reached.")
            exit(0)

        # save checkpoint interval
        if checkpoint_interval:
            if global_step % checkpoint_interval == 0:
                save_checkpoint(global_step, objs)

    save_checkpoint(global_step, objs)
    dist.barrier()
    if is_the_first_tp_pp_dp_node:
        # merge the output json files in the first node
        objs = []
        for dp_rank in tqdm(range(mpu.get_data_parallel_world_size()), desc="Merging output files..."):
            filename = f"{ckpt_path}/rank_{dp_rank}.json"
            with jsonlines.open(filename, mode="r") as reader:
                for obj in reader:
                    objs.append(obj)

        # write objs
        with jsonlines.open(cfg.output_file, mode="w") as writer:
            writer.write_all(objs)


if __name__ == "__main__":
    with torch.no_grad():
        main()  # noqa pylint: disable=no-value-for-parameter
