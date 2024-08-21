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
from dataclasses import dataclass

import jsonlines
import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.nlp.gpt.megatron_gpt_critic import MegatronGPTCriticModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

"""Script to start SFT training"""


@dataclass
class ValueDataset:
    path_to_jsonl: str

    def __post_init__(self):
        assert os.path.exists(self.path_to_jsonl), f"{self.path_to_jsonl=} needs to exist"
        with jsonlines.open(self.path_to_jsonl) as reader:
            self.data = list(iter(reader))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="gpt_value")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # load the pretrained RM
    ptl_model = load_from_nemo(
        MegatronGPTCriticModel,
        cfg.model,
        trainer,
        strict=False,  # TODO: change back to True
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
        load_base_model_only=True,
    )

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path

    # TODO: log this restore path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_ds = ValueDataset(cfg.train_path)
    validation_ds = ValueDataset(cfg.validation_path)

    eos_id = ptl_model.tokenizer.eos_id

    tokenizer = ptl_model.tokenizer

    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    SYSTEM_PROMPT_TEMPLATE = f"<extra_id_0>System\n{SYSTEM_PROMPT}\n"

    USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

    # had to delete the value here because we concat the prompts
    ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n"

    # TODO: what to do with this?
    LABEL_PREFIX = "<extra_id_2>"

    def collate_fn(batch, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):

        tokens = []
        masks = []
        values = []

        values_padding = [-100] * 9

        for b in batch:
            assert len(b["masks"]) == len(b["values"]) == len(b["token_ids"])

            prompt = SYSTEM_PROMPT_TEMPLATE + USER_TURN_TEMPLATE.format(value=b["prompt"]) + ASSISTANT_TURN_TEMPLATE
            prompt = tokenizer.text_to_ids(b["prompt"])
            response = prompt + b["token_ids"]
            mask = len(prompt) * [0] + b["masks"]

            value = [values_padding] * len(prompt) + [
                [-100, -100, -100, -100, score["helpfulness"], score["correctness"], score["coherence"], -100, -100,]
                for score in b["values"]
            ]

            tokens.append(torch.as_tensor(response, dtype=torch.long))
            masks.append(torch.as_tensor(mask, dtype=torch.float32))
            values.append(torch.as_tensor(value, dtype=torch.float32))

        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=eos_id)
        scores = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=-100)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens, eos_id, False, True, False,)

        if attention_mask.shape[0] == 1:
            # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
            # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
            attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))

        output = {
            "tokens": tokens,
            "scores": scores,
            "mask": masks,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return output

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        pad_samples_to_global_batch_size=False,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        pad_samples_to_global_batch_size=False,
        load_gbs=True,
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    sft_trainer = SupervisedTrainer(
        cfg=cfg.trainer.sft,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        sft_trainer.load_state_dict(custom_trainer_state_dict)

    sft_trainer.fit()


if __name__ == "__main__":
    main()
