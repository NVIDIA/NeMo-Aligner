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
from functools import partial

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, build_train_valid_test_knowledge_distillation_datasets
from nemo_aligner.models.nlp.gpt.megatron_gpt_knowledge_distillation import GPTKnowledgeDistillationModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

"""Script to start knowledge_distillation training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def _collate_fn(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, use_k_add_1_logits=False):
        tokens = [item['tokens'] for item in batch]
        labels = [item['labels'] for item in batch]
        loss_mask = [item['loss_mask'] for item in batch]
        
        topk_logits = [item['topk_logits'] for item in batch]
        topk_token_ids = [item['topk_token_ids'] for item in batch]

        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        loss_mask = torch.nn.utils.rnn.pad_sequence(loss_mask, batch_first=True, padding_value=-100)
        assert len(tokens.shape) == 2, "tokens size should be [B, seq_length], got {tokens.shape}"
        assert len(labels.shape) == 2, "labels size should be [B, seq_length], got {labels.shape}"
        assert len(loss_mask.shape) == 2, "loss_mask size should be [B, seq_length], got {loss_mask.shape}"
        
        topk_logits = torch.nn.utils.rnn.pad_sequence(topk_logits, batch_first=True, padding_value=0)
        topk_token_ids = torch.nn.utils.rnn.pad_sequence(topk_token_ids, batch_first=True, padding_value=0)
        assert len(topk_logits.shape) == 3, "topk_logits size should be [B, seq_length], got {topk_logits.shape}"
        assert len(topk_token_ids.shape) == 3, "topk_token_ids size should be [B, seq_length], got {topk_token_ids.shape}"
        
        if use_k_add_1_logits:
            log_sum_exp_logits = [item['log_sum_exp_logits'] for item in batch]
            log_sum_exp_logits = torch.nn.utils.rnn.pad_sequence(log_sum_exp_logits, batch_first=True, padding_value=0)
            assert len(log_sum_exp_logits.shape) == 2, "log_sum_exp_logits size should be [B, seq_length], got {log_sum_exp_logits.shape}"
        else:
            log_sum_exp_logits = None
    
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss,
        )
        assert attention_mask.ndim == 4, "attention_mask is incorrect shape for dpo_custom_collate"
        if attention_mask.shape[0] == 1:
            # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
            # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
            attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))
            
        output = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "topk_logits": topk_logits,
            "topk_token_ids": topk_token_ids,
            "log_sum_exp_logits": log_sum_exp_logits,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return output


@hydra_runner(config_path="conf", config_name="gpt_knowledge_distillation")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "knowledge_distillation")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        GPTKnowledgeDistillationModel,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=False,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    init_peft(ptl_model, cfg.model)
    
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

     # use the entire dataset
    train_valid_test_num_samples = [-1 * cfg.model.global_batch_size] * 3

    train_ds, validation_ds, _ = build_train_valid_test_knowledge_distillation_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=ptl_model.tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=partial(
            _collate_fn,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
            use_k_add_1_logits=cfg.model.knowledge_distillation.get("use_k_add_1_logits", False),
        ),
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        pad_samples_to_global_batch_size=False,
        collate_fn=partial(
            _collate_fn,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
            use_k_add_1_logits=cfg.model.knowledge_distillation.get("use_k_add_1_logits", False),
        ),
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))
    kd_trainer = SupervisedTrainer(
        cfg=cfg.trainer.knowledge_distillation,
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
        kd_trainer.load_state_dict(custom_trainer_state_dict)

    kd_trainer.fit()


if __name__ == "__main__":
    main()
