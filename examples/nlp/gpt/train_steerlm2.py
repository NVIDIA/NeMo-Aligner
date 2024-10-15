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


import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import get_prompt_template_example
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.data.nlp.datasets import SteerLM2Dataset
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.models.nlp.gpt.gpt_steerlm_model import GPTSteerLMModel
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
from nemo_aligner.utils.utils import load_from_nemo

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def build_sft_dataset(data_cfg, tokenizer, num_samples, answer_only_loss=True, is_chat=True, special_tokens=None):
    dataset_cls = SteerLM2Dataset
    dataset = dataset_cls(
        file_path=data_cfg.file_path,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        add_bos=data_cfg.get("add_bos", False),
        add_eos=data_cfg.get("add_eos", True),
        add_sep=data_cfg.get("add_sep", False),
        sep_id=0,
        max_num_samples=num_samples,
        seed=data_cfg.get("seed", 1234),
        label_key=data_cfg.get("label_key", "answer"),
        answer_only_loss=answer_only_loss,
        truncation_field=data_cfg.get("truncation_field", "text"),
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        index_mapping_dir=data_cfg.get("index_mapping_dir", None),
        prompt_template=data_cfg.get("prompt_template", None),
        virtual_tokens=0,
        memmap_workers=data_cfg.get(
            "memmap_workers", None
        ),  # used to set num. of workers to create the memmap index files
        hf_dataset=data_cfg.get(
            "hf_dataset", False
        ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        truncation_method=data_cfg.get(
            "truncation_method", "right"
        ),  # used to choose truncation method. Options: ['random', 'left', 'right']
        special_tokens=special_tokens,
        output_original_text=data_cfg.get("output_original_text", False),
    )
    return dataset


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.peft = cfg.model.peft
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get("hidden_dropout", 0.0)
        gpt_cfg.attention_dropout = cfg.model.get("attention_dropout", 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        gpt_cfg.use_flash_attention = cfg.model.get("use_flash_attention", False)
        # if TP/PP size is -1, use default TP/PP size as original model
        if cfg.model.get("tensor_model_parallel_size", 1) > 0:
            gpt_cfg.tensor_model_parallel_size = cfg.model.get("tensor_model_parallel_size", 1)
        if cfg.model.get("pipeline_model_parallel_size", 1) > 0:
            gpt_cfg.pipeline_model_parallel_size = cfg.model.get("pipeline_model_parallel_size", 1)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get("pipeline_model_parallel_split_rank", 0)
        gpt_cfg.steerlm2 = cfg.model.steerlm2

        if cfg.model.data.get("chat", False):
            # chat model, overwrite the prompt template
            prompt_template = get_prompt_template_example(cfg.model.data.chat_prompt_tokens)
            gpt_cfg.data.train_ds.prompt_template = prompt_template
            gpt_cfg.data.validation_ds.prompt_template = prompt_template

        sft_cls = GPTSFTModel
        gpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        if cfg.model.get("use_flash_attention", None) is not None:
            gpt_cfg.use_flash_attention = cfg.model.use_flash_attention

        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            gpt_cfg.seq_len_interpolation_factor = cfg.model.seq_len_interpolation_factor

        gpt_cfg.inference = cfg.model.get("inference", {})

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


@hydra_runner(config_path="conf", config_name="gpt_sft")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    ptl_model, updated_cfg = load_from_nemo(
        GPTSteerLMModel,
        cfg,
        trainer,
        strict=True,
        modify_config_fn=_modify_config,
        restore_path=cfg.model.restore_from_path,
        return_updated_cfg=True,
    )

    init_peft(ptl_model, updated_cfg)

    with open_dict(cfg):
        # overwrite the model config with the config from the checkpoint
        cfg.model.encoder_seq_length = ptl_model.cfg.encoder_seq_length

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

    train_data_cfg = cfg.model.data.train_ds
    val_data_cfg = cfg.model.data.validation_ds

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None
    train_ds = build_sft_dataset(
        train_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )
    if cfg.model.data.get("sample", False):
        num_samples = cfg.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
    else:
        num_samples = None
    validation_ds = build_sft_dataset(
        val_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=train_data_cfg.micro_batch_size,
        gbs=train_data_cfg.global_batch_size,
        collate_fn=train_ds.collate_fn,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not train_data_cfg.drop_last,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=val_data_cfg.micro_batch_size,
        gbs=val_data_cfg.global_batch_size,
        collate_fn=validation_ds.collate_fn,
        drop_last=val_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not val_data_cfg.drop_last,
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
