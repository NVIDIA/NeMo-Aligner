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
from nemo_aligner.data.mm.builders import build_mm_sft_dataset
from nemo_aligner.models.mm.mgpt.mgpt_sft_model import MultimodalGPTSFTModel
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
from nemo.collections.multimodal.data.neva.neva_dataset import DataCollatorForSupervisedDataset

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def _modify_config(mgpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(mgpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(mgpt_cfg):
        mgpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        mgpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        mgpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        mgpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        mgpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        mgpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        mgpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        mgpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        mgpt_cfg.peft = cfg.model.peft
        mgpt_cfg.data = cfg.model.data
        mgpt_cfg.optim = cfg.model.optim
        mgpt_cfg.precision = cfg.trainer.precision
        mgpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        mgpt_cfg.restore_from_path = cfg.model.restore_from_path
        mgpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        mgpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        mgpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        mgpt_cfg.hidden_dropout = cfg.model.get("hidden_dropout", 0.0)
        mgpt_cfg.attention_dropout = cfg.model.get("attention_dropout", 0.0)
        mgpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        mgpt_cfg.use_flash_attention = cfg.model.get("use_flash_attention", False)
        # if TP/PP size is -1, use default TP/PP size as original model
        if cfg.model.get("tensor_model_parallel_size", 1) > 0:
            mgpt_cfg.tensor_model_parallel_size = cfg.model.get("tensor_model_parallel_size", 1)
        if cfg.model.get("pipeline_model_parallel_size", 1) > 0:
            mgpt_cfg.pipeline_model_parallel_size = cfg.model.get("pipeline_model_parallel_size", 1)
        mgpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get("pipeline_model_parallel_split_rank", 0)

        if cfg.model.data.get("chat", False):
            # chat model, overwrite the prompt template
            prompt_template = get_prompt_template_example(cfg.model.data.chat_prompt_tokens)
            mgpt_cfg.data.train_ds.prompt_template = prompt_template
            mgpt_cfg.data.validation_ds.prompt_template = prompt_template

        sft_cls = MultimodalGPTSFTModel
        mgpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        if cfg.model.get("use_flash_attention", None) is not None:
            mgpt_cfg.use_flash_attention = cfg.model.use_flash_attention

        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            mgpt_cfg.seq_len_interpolation_factor = cfg.model.seq_len_interpolation_factor

        mgpt_cfg.inference = cfg.model.get("inference", {})

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(mgpt_cfg)
            mgpt_cfg.cfg = mgpt_cfg

        # Multimodal utility
        if 'mm_cfg' not in mgpt_cfg:
            mgpt_cfg.mm_cfg = OmegaConf.create()
            mgpt_cfg.mm_cfg.llm = OmegaConf.create()
            mgpt_cfg.mm_cfg.vision_encoder = OmegaConf.create()        
        # data
        mgpt_cfg.data.is_multimodal = cfg.model.data.get("is_multimodal", True)
        mgpt_cfg.data.media_type = cfg.model.data.get("media_type", "image")
        mgpt_cfg.data.image_folder = cfg.model.data.get("image_folder", None)
        mgpt_cfg.data.image_aspect_ratio = cfg.model.data.get("image_aspect_ratio", "square")
        mgpt_cfg.data.image_token_len = cfg.model.data.get("image_token_len", 256)
        # model
        mgpt_cfg.mm_cfg.llm.from_pretrained = cfg.model.mm_cfg.llm.get("from_pretrained", None)
        mgpt_cfg.mm_cfg.llm.freeze = cfg.model.mm_cfg.llm.get("freeze", True)
        mgpt_cfg.mm_cfg.vision_encoder.from_pretrained = cfg.model.mm_cfg.vision_encoder.get("from_pretrained", None)
        mgpt_cfg.mm_cfg.vision_encoder.from_hf = cfg.model.mm_cfg.vision_encoder.get("from_hf", True)
        mgpt_cfg.mm_cfg.vision_encoder.patch_dim = cfg.model.mm_cfg.vision_encoder.get("patch_dim", 14)
        mgpt_cfg.mm_cfg.vision_encoder.crop_size = cfg.model.mm_cfg.vision_encoder.get("crop_size", [224, 224])
        mgpt_cfg.mm_cfg.vision_encoder.hidden_size = cfg.model.mm_cfg.vision_encoder.get("hidden_size", 1024)
        mgpt_cfg.mm_cfg.vision_encoder.vision_select_layer = cfg.model.mm_cfg.vision_encoder.get("vision_select_layer", -2)
        mgpt_cfg.mm_cfg.vision_encoder.class_token_length = cfg.model.mm_cfg.vision_encoder.get("class_token_length", 1)
        mgpt_cfg.mm_cfg.vision_encoder.freeze = cfg.model.mm_cfg.vision_encoder.get("freeze", True)
        mgpt_cfg.mm_cfg.pretrain_mm_mlp_adapter = cfg.model.mm_cfg.get("pretrain_mm_mlp_adapter", None)
        mgpt_cfg.mm_cfg.mm_mlp_adapter_type = cfg.model.mm_cfg.get("mm_mlp_adapter_type", "linear")
        mgpt_cfg.mm_cfg.use_im_start_end = cfg.model.mm_cfg.get("use_im_start_end", False)
        mgpt_cfg.mm_cfg.im_start_token = cfg.model.mm_cfg.get("im_start_token", "<extra_id_4>")
        mgpt_cfg.mm_cfg.im_end_token = cfg.model.mm_cfg.get("im_end_token", "<extra_id_5>")
        mgpt_cfg.mm_cfg.image_patch_token = cfg.model.mm_cfg.get("image_patch_token", "<extra_id_3>")
        
        # check if we are pretraining the adapter or finetuning the LLM
        if mgpt_cfg.mm_cfg.llm.freeze and mgpt_cfg.restore_from_path is None: # if pretraining
            mgpt_cfg.restore_from_path = mgpt_cfg.mm_cfg.llm.from_pretrained

        # Check if an external tokenizer is given
        artifacts = ["model", "vocab_file", "merge_file", "additional_special_tokens"]
        for artifact in artifacts:
            if mgpt_cfg.tokenizer.get(artifact, None):
                mgpt_cfg.tokenizer[artifact] = cfg.model.tokenizer[artifact]

    return mgpt_cfg


@hydra_runner(config_path="conf", config_name="mgpt_sft")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    restore_mm_adapter = False if (cfg.model.mm_cfg.llm.freeze and cfg.model.restore_from_path is None) else True
    ptl_model, updated_cfg = load_from_nemo(
        MultimodalGPTSFTModel,
        cfg,
        trainer,
        strict=True,
        modify_config_fn=_modify_config,
        restore_path=cfg.model.restore_from_path if restore_mm_adapter else cfg.model.mm_cfg.llm.from_pretrained,
        return_updated_cfg=True,
        restore_mm_adapter=restore_mm_adapter,
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
    mm_cfg = cfg.model.mm_cfg

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None

    collate_fn = DataCollatorForSupervisedDataset(cfg.model, ptl_model.tokenizer)
    
    train_ds = build_mm_sft_dataset(
        cfg.model,
        train_data_cfg,
        mm_cfg,
        ptl_model.tokenizer,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )
    if cfg.model.data.get("sample", False):
        num_samples = cfg.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
    else:
        num_samples = None
    validation_ds = build_mm_sft_dataset(
        cfg.model,
        val_data_cfg,
        mm_cfg,
        ptl_model.tokenizer,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=train_data_cfg.micro_batch_size,
        gbs=train_data_cfg.global_batch_size,
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
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
