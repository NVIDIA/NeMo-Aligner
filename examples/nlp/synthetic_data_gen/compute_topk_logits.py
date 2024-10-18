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

import json
import os
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo_aligner.data.nlp.builders import build_sft_dataset
from nemo_aligner.utils.distributed import compute_topk_logits_in_batched_sequence
from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def write_generations(output_path, indices, batch, topk_logits, topk_token_ids, log_sum_exp_logits, num_padding):
    # save the results to the file
    topk_logits = topk_logits.tolist()
    topk_token_ids = topk_token_ids.tolist()
    log_sum_exp_logits = log_sum_exp_logits.tolist()
    batch = {k: v if isinstance(v, (list, dict, str, int, float)) else v.tolist() for k, v in batch.items()}
    with open(output_path, "a", encoding="utf-8") as write_file:
        for i in range(len(batch["tokens"]) - num_padding): ## do not write the dummy padding examples to disc
            obj = {k: v[i] for k, v in batch.items()}
            obj["topk_logits"] = topk_logits[i]
            obj["topk_token_ids"] = topk_token_ids[i]
            obj["log_sum_exp_logits"] = log_sum_exp_logits[i]
            obj["index"] = indices[i]
            write_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        write_file.flush()


@hydra_runner(config_path="conf", config_name="compute_topk_logits")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    logger = CustomLoggerWrapper(trainer.loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    # load the model
    ptl_model = load_from_nemo(
        MegatronGPTModel,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=False,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )
    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # load the dataset
    dataset = build_sft_dataset(
        cfg.data.data,
        ptl_model.tokenizer,
        num_samples=None,
        answer_only_loss=True,
        is_chat=cfg.data.chat,
        special_tokens=cfg.data.chat_prompt_tokens,
    )

    # get the processed indices which avoids repeated computations.
    if not os.path.exists(os.path.dirname(os.path.abspath(cfg.output_path))):
        os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)))
    processed_indices = set()
    if os.path.exists(cfg.output_path):
        processed_indices = set([json.loads(l)["index"] for l in open(cfg.output_path)])

    start_from_idx = cfg.get("start_from_idx", 0)
    end_at_idx = cfg.get("end_at_idx", len(dataset) - 1)

    for i in range(start_from_idx, end_at_idx + 1, cfg.batch_size):
        end_i = min(end_at_idx, i + cfg.batch_size - 1)
        num_padding = cfg.batch_size - (end_i - i + 1)

        logging.info(f"Processing {i}:{end_i} items / till {end_at_idx}.")
        indices = [j for j in range(i, end_i + 1) if j not in processed_indices]

        ## pad to batch size using the last example
        if num_padding:
            indices = indices + [indices[-1]] * num_padding

        if len(indices) == 0:
            continue

        # prepare the batch
        batch = [dataset[j] for j in indices]
        batch = dataset.collate_fn(batch)

        # compute the topk logits
        with torch.no_grad():
            topk_logits, topk_token_ids, log_sum_exp_logits = compute_topk_logits_in_batched_sequence(
                ptl_model.model,
                batch["tokens"].to(torch.cuda.current_device()),
                batch["position_ids"].to(torch.cuda.current_device()),
                batch["attention_mask"].to(torch.cuda.current_device()),
                cfg.top_k,
                cfg.trainer.precision,
                forward_micro_batch_size=cfg.forward_micro_batch_size,
            )

        # write the results
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            batch.pop("position_ids")
            batch.pop("attention_mask")
            write_generations(
                cfg.output_path, indices, batch, topk_logits.cpu(), topk_token_ids.cpu(), log_sum_exp_logits.cpu(), num_padding
            )

    logging.info("Finish generations.")


if __name__ == "__main__":
    main()
