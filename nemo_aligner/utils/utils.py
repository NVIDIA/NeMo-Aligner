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

"""Misc helper functions"""

import gc
import os
import re
import tempfile
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from omegaconf import OmegaConf

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import NeMoModelCheckpoint
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel


class CustomSaveRestoreConnector(NLPSaveRestoreConnector):
    """A save connector that will ask the Reward model to not try to load
        the rm head if load_base_model_only is True
    """

    def __init__(self, *args, load_base_model_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.__load_base_model_only = load_base_model_only

    def restore_from(self, *args, **kwargs):
        if not self.__load_base_model_only:
            return super().restore_from(*args, **kwargs)

        with patch.object(GPTRewardModel, "return_rm_head_in_state_dict", False):
            output = super().restore_from(*args, **kwargs)

        return output


def custom_save_ckpt_func(self, trainer, pl_module, monitor_candidates, is_train_end=False, save_top_only=False):
    """work around used so we can save models manually"""
    super(NeMoModelCheckpoint, self)._save_topk_checkpoint(trainer, monitor_candidates)
    if save_top_only:
        return

    super(NeMoModelCheckpoint, self)._save_last_checkpoint(trainer, monitor_candidates)

    if is_train_end:
        # stop the checkpoint logic from saving another last checkpoint
        with patch.object(trainer, "val_check_interval", 0):
            self.on_train_end(trainer, pl_module)


def load_from_nemo(
    cls, model_cfg, trainer, strict=True, modify_config_fn=None, restore_path=None, load_base_model_only=False
):
    """load a model using nemo checkpoint
    """
    connector = CustomSaveRestoreConnector(load_base_model_only=load_base_model_only)
    assert os.path.exists(restore_path), f"tried to load from {restore_path=} but it does not exist"

    # if we gave it a directory, then load as if it was extracted already
    if os.path.isdir(restore_path):
        connector.model_extracted_dir = restore_path

    if modify_config_fn is not None:
        origin_cfg = cls.restore_from(
            restore_path=restore_path, trainer=trainer, return_config=True, save_restore_connector=connector,
        )
        model_cfg = modify_config_fn(origin_cfg, model_cfg, add_cfg_to_tree=False)

    model = cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=connector,
        strict=strict,
    )
    return model


def load_checkpoint_model_config(restore_path):
    """load only the model config from a checkpoint
    """
    config_name_in_ckpt = NLPSaveRestoreConnector()._model_config_yaml
    assert os.path.exists(restore_path), f"tried to load from {restore_path=} but it does not exist"

    # if we gave it a directory, then load the cfg directly
    if os.path.isdir(restore_path):
        cfg_path = os.path.join(restore_path, config_name_in_ckpt)
        assert os.path.exists(cfg_path), f"tried to load cfg at {cfg_path=} but it does not exist"
        return OmegaConf.load(cfg_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(restore_path, tmpdir, extract_config_only=True)
        cfg = OmegaConf.load(os.path.join(tmpdir, config_name_in_ckpt))

    return cfg


def load_and_override_model_config(restore_path, model_cfg_to_overwrite, remove_meta_info=True):
    """load the config in the model checkpoint and then overwrite it
        with whatever is provided 
    """
    checkpoint_cfg = load_checkpoint_model_config(restore_path)

    if remove_meta_info:
        checkpoint_cfg.pop("target", None)
        checkpoint_cfg.pop("nemo_version", None)

    return OmegaConf.merge(checkpoint_cfg, model_cfg_to_overwrite)


def masked_mean(values, mask, dim=None):
    """
    Masks values with mask, and computes the mean of the values using the masked values.
    """
    return values[mask.bool()].mean(dim=dim)


def masked_std(values, mask, dim=None):
    """
    Masks values with mask, and computes the std of the values using the masked values.
    """
    return values[mask.bool()].std(dim=dim)


def extract_value_from_ckpt(key, ckpt_path):
    try:
        val = int(float(re.findall(rf"{key}\=([0-9]+\.*[0-9]*)", ckpt_path)[0]))
    except (ValueError, TypeError, IndexError):
        logging.warning(f"Cannot parse the checkpoint file to get {key}, we assume it is zero.")
        val = 0
    return val


def set_autocast_gpu_dtype(precision):
    if precision == 16:
        torch.set_autocast_gpu_dtype(torch.float16)
    elif precision == "bf16":
        torch.set_autocast_gpu_dtype(torch.bfloat16)


def calculate_response_lengths(tokens, eos_id):
    """calculates the response length of the tokens after padding"""
    return (tokens != eos_id).sum(-1)


def calculate_dialogue_response_lengths(
    tokens, prompt_lengths, tokenizer, end_strings, max_generation_length, max_sequence_length
):
    # for EOS
    eos_length = calculate_response_lengths(tokens, tokenizer.eos_id)

    if "<extra_id_1>" in end_strings:
        # for the extra_id_1
        extra_id_1_idx = tokenizer.text_to_ids("<extra_id_1>")[-1]
        mask = tokens == extra_id_1_idx

        # take the last extra id token index(assumes we are not padding with extra_id_1)
        length_with_extra_id_1 = torch.argmax(
            mask * torch.arange(tokens.size(-1), device=torch.cuda.current_device()), dim=-1
        )

        # if it terminated on the extra token id, then it must have been generated by the model, otherwise it couldn't have
        length_with_extra_id_1 = torch.where(
            length_with_extra_id_1 >= prompt_lengths, length_with_extra_id_1, torch.iinfo(torch.int32).max
        )

        # either terminated using eos id or extra id 1
        lengths = torch.minimum(eos_length, length_with_extra_id_1)
    else:
        lengths = eos_length

    # we also want the model to learn EOS or extra id 1
    lengths = lengths + 1
    # Ensure we never go over `length_params.max_length`. Note that this means the response may not necessarily
    # end with EOS / extra_id_1 (we should not enforce it as PPO training requires the real generated token).
    max_lengths = prompt_lengths + max_generation_length
    lengths = torch.minimum(lengths, max_lengths)

    # Prompts' max size and `max_length` should be such that we never exceed the encoder input size.
    assert (lengths <= max_sequence_length).all()
    return lengths


def configure_batch_sizes(mbs, gbs, dp=1):
    app_state = AppState()
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        data_parallel_size=dp,
    )


def select_log_probs(full_log_probs, indices):
    """This function selects out the log probs using the indices tensor to get the correct logprobs per token.

    Args:
        full_log_probs (torch.Tensor): log probs with shape of B x S x V
        indices (torch.Tensor): B x S

    Returns:
        log_probs (torch.Tensor): B x (S-1)
        full_log_probs (torch.Tensor): B x (S-1) x V

    NOTE:
        - because the transformer predicts the next word, the last word it predicts in the
        full log probs does not have a label, so will not be included
    """
    full_log_probs = full_log_probs[:, :-1, :].contiguous()
    indices = indices[:, 1:].unsqueeze(-1)
    log_probs = torch.gather(input=full_log_probs, dim=2, index=indices).squeeze(dim=-1).contiguous()

    return log_probs, full_log_probs


def dist_adam_load_state_bucket_into_device(state_bucket, device):
    """put the state bucket onto a device
    """
    attrs_to_offload = ["params_shard", "param_remainders_shard", "exp_avg_shard", "exp_avg_sq_shard"]

    for attr in attrs_to_offload:
        tensor = getattr(state_bucket, attr)
        if tensor is not None:
            setattr(state_bucket, attr, tensor.to(device=device, non_blocking=True))


@contextmanager
def offload_distributed_adam(state_dict):
    """context manager to offload distributed adam states
    """
    # off load onto cpu
    for state_bucket in state_dict["state"]["buckets"]:
        dist_adam_load_state_bucket_into_device(state_bucket, device="cpu")

    # make sure the offloading is finished before returning
    torch.cuda.synchronize()

    try:
        yield

    finally:
        # onload back onto gpu
        for state_bucket in state_dict["state"]["buckets"]:
            dist_adam_load_state_bucket_into_device(state_bucket, device=torch.cuda.current_device())

        # make sure the onloading is finished before returning
        torch.cuda.synchronize()


def collate_with_batch_max_sequence_length(
    data_batch, response_token_length, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss
):
    """collate function that batches by max sequence length
    """
    texts = [item["text"] for item in data_batch]
    loss_multipliers = torch.as_tensor([item["loss_multiplier"] for item in data_batch]).view(len(data_batch), 1)
    lengths = torch.as_tensor([item["length"] for item in data_batch])
    batch_max_length = lengths.max()

    # pad each sequence to len(prompt) + response token length
    texts = [
        torch.cat([seq, torch.full((batch_max_length + response_token_length - len(seq),), eos_id, dtype=seq.dtype)])
        for seq in texts
    ]

    texts = torch.stack(texts)

    # NOTE: the attention mask is 1x1xSxS, which will broadcast on the batch dimension
    attention_masks, loss_masks, position_ids = get_ltor_masks_and_position_ids(
        texts, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss
    )

    return {
        "text": texts,
        "length": lengths,
        "attention_mask": attention_masks,
        # to preserve the loss mask from the dataset
        "loss_mask": loss_masks * loss_multipliers,
        "position_ids": position_ids,
    }


def apply_func_to_dict(func, dictionary):
    return {k: func(v) for k, v in dictionary.items()}


def move_to_device_if_tensor(device, item):
    if torch.is_tensor(item):
        item = item.to(device)
    return item


cuda_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cuda"))
cpu_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cpu"))


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def retrieve_model_state_dict_in_cpu(model, megatron_amp_O2=True):
    """get a copy of the model states in CPU
    """
    cpu_dict = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            item = item.detach().to(device="cpu", non_blocking=True, copy=True)

        cpu_dict[name] = item

    if megatron_amp_O2:
        cpu_dict = convert_to_amp_o2_format(cpu_dict)

    torch.cuda.synchronize()
    return cpu_dict


@torch.no_grad()
def swap_dict(resident_model, cpu_weights, offload_onto_cpu=True, megatron_amp_O2=True):
    """swap the state dict with a specified state dict, and offload the current state dict onto CPU
        if needed
    """
    offloaded_weights = {}

    if offload_onto_cpu:
        offloaded_weights = retrieve_model_state_dict_in_cpu(resident_model, megatron_amp_O2=megatron_amp_O2)

    resident_model.load_state_dict(cpu_weights)
    return offloaded_weights


@contextmanager
def cpu_weight_swap(resident_model, cpu_weights, megatron_amp_O2=True):
    """swap the weights into GPU, and then swap it out once return
    """
    cpu_dict = swap_dict(resident_model, cpu_weights, megatron_amp_O2=megatron_amp_O2)
    try:
        yield

    finally:
        swap_dict(resident_model, cpu_dict, offload_onto_cpu=False, megatron_amp_O2=megatron_amp_O2)


def convert_to_amp_o2_format(state_dict):
    """when amp_o2 is enabled, the model gets wrapped in a Float16Module which changes
        the keys and how it loads need to add module onto it
    """
    new_state_dict = {}

    for key in state_dict.keys():
        new_key = key.replace("model.", "model.module.", 1)
        new_state_dict[new_key] = state_dict[key]

    return new_state_dict
