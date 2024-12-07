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
import functools
import gc
import itertools
import os
import re
import tempfile
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial, wraps
from typing import Any, Iterator, List, Optional
from unittest.mock import patch

import torch
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensorFactory
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
from omegaconf import DictConfig, OmegaConf
from torch.masked import as_masked_tensor

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import NeMoModelCheckpoint
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel


class CustomSaveRestoreConnector(NLPSaveRestoreConnector):
    """A save connector that will ask the Reward model to not try to load
        the rm head if load_base_model_only is True
    """

    def __init__(self, *args, load_base_model_only=False, replace_sharded_tensor_key: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__load_base_model_only = load_base_model_only
        self.__replace_sharded_tensor_key = replace_sharded_tensor_key

    def restore_from(self, *args, **kwargs):
        if not self.__load_base_model_only:
            return super().restore_from(*args, replace_sharded_tensor_key=self.__replace_sharded_tensor_key, **kwargs)

        with patch.object(GPTRewardModel, "return_rm_head_in_state_dict", False):
            output = super().restore_from(
                *args, replace_sharded_tensor_key=self.__replace_sharded_tensor_key, **kwargs
            )

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
    cls,
    model_cfg,
    trainer,
    strict=True,
    modify_config_fn=None,
    restore_path=None,
    load_base_model_only=False,
    return_updated_cfg=False,
):
    """load a model using nemo checkpoint
    """
    assert os.path.exists(restore_path), f"tried to load from {restore_path=} but it does not exist"

    is_2_0_ckpt = load_2_0_checkpoint_model_config(restore_path) is not None
    if is_2_0_ckpt:
        replace_sharded_tensor_key = "module"
    else:
        replace_sharded_tensor_key = None

    connector = CustomSaveRestoreConnector(
        load_base_model_only=load_base_model_only, replace_sharded_tensor_key=replace_sharded_tensor_key
    )

    if is_2_0_ckpt:
        connector.model_weights_ckpt = "weights"

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

    if is_2_0_ckpt:
        connector.model_weights_ckpt = "model_weights.ckpt"

    return (model, model_cfg) if return_updated_cfg else model


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
        # Extracts only model config
        members = NLPSaveRestoreConnector._filtered_tar_info(restore_path, filter_fn=lambda name: ".yaml" in name)
        NLPSaveRestoreConnector._unpack_nemo_file(restore_path, tmpdir, members=members)
        cfg = OmegaConf.load(os.path.join(tmpdir, config_name_in_ckpt))

    return cfg


def load_2_0_checkpoint_model_config(restore_path: str):
    try:
        from nemo.lightning import io
        from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
        from nemo.lightning.io.pl import ckpt_to_weights_subdir

        if (
            os.path.isdir(ckpt_to_context_subdir(restore_path))
            and os.path.isdir(ckpt_to_weights_subdir(restore_path, is_saving=False))
            and os.path.isfile(os.path.join(ckpt_to_context_subdir(restore_path), "io.json"))
        ):
            config = io.load_context(restore_path, subpath="model.config")
            tokenizer_cfg = OmegaConf.load(os.path.join(ckpt_to_context_subdir(restore_path), "model.yaml")).tokenizer

            def get_tokenizer_args(tokenizer_cfg):
                if "AutoTokenizer" in tokenizer_cfg._target_:
                    tokenizer_type = "huggingface"
                    tokenizer_name = (
                        tokenizer_cfg.pretrained_model_name
                        if isinstance(tokenizer_cfg.pretrained_model_name, str)
                        else tokenizer_cfg.pretrained_model_name.attr
                    )
                    if os.path.isfile(
                        os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)
                    ) or os.path.isdir(os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)):
                        tokenizer_name = os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)

                    args = {
                        "library": tokenizer_type,
                        "type": tokenizer_name,
                        "use_fast": True,
                    }
                    if tokenizer_cfg.get("vocab_file", None):
                        args["vocab_file"] = os.path.join(
                            ckpt_to_context_subdir(restore_path), tokenizer_cfg.vocab_file
                        )
                    if tokenizer_cfg.get("merges_file", None):
                        args["merges_file"] = os.path.join(
                            ckpt_to_context_subdir(restore_path), tokenizer_cfg.merges_file
                        )

                    return args
                elif "SentencePieceTokenizer" in tokenizer_cfg._target_:
                    tokenizer_type = "sentencepiece"
                    tokenizer_name = tokenizer_cfg.model_path
                    if os.path.isfile(
                        os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)
                    ) or os.path.isdir(os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)):
                        tokenizer_name = os.path.join(ckpt_to_context_subdir(restore_path), tokenizer_name)
                    elif not os.path.isfile(tokenizer_name):
                        raise FileNotFoundError(f"Tokenizer file {tokenizer_name} not found")

                    return {"library": tokenizer_type, "type": None, "model": tokenizer_name}
                else:
                    raise ValueError(f"Unknown tokenizer type: {tokenizer_cfg}")

            tokenizer_args = get_tokenizer_args(tokenizer_cfg)

            config_dict = {}
            for k, v in config.__dict__.items():
                if isinstance(v, (float, int, str, bool)):
                    config_dict[k] = v
                elif k == "activation_func":
                    config_dict["activation"] = v.__name__

            if config_dict["activation"] == "silu":
                config_dict["activation"] = "fast-swiglu"

            config_dict["encoder_seq_length"] = config_dict["seq_length"]

            config_dict["mcore_gpt"] = True
            config_dict["max_position_embeddings"] = config_dict.get("seq_length")
            config_dict["tokenizer"] = tokenizer_args
            config_dict["bias"] = config_dict.get("add_bias_linear", True)
            config_dict["qkv_bias"] = config_dict.get("add_qkv_bias", False)

            try:
                strategy: dict[str, Any] = io.load_context(restore_path, subpath="trainer.strategy").__dict__
                config_dict["gradient_as_bucket_view"] = strategy.get("gradient_as_bucket_view", True)
                # TODO: Add any other parameters required from strategy here
            except Exception:
                # Default to True based on default values in https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm/recipes
                config_dict["gradient_as_bucket_view"] = True

            try:
                precision_plugin: dict[str, Any] = io.load_context(restore_path, subpath="trainer.plugins").__dict__
                config_dict["fp16"] = precision_plugin.get("fp16", False)
                config_dict["bf16"] = precision_plugin.get("bf16", True)
                # TODO: Add any other parameters required from precision plugin here
            except Exception:
                # Default to True based on default values in https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm/recipes
                config_dict["fp16"] = False
                config_dict["bf16"] = True

            if not os.path.isfile(os.path.join(restore_path, "model_config.yaml")):
                OmegaConf.save(config=OmegaConf.create(config_dict), f=os.path.join(restore_path, "model_config.yaml"))

            return config_dict
    except Exception:
        # If there's a failure loading the path as a NeMo 2.0 checkpoint,
        # return None and continue loading NeMo 1.0 checkpoint.
        return None

    return None


def load_and_override_model_config(restore_path, model_cfg_to_overwrite, remove_meta_info=True):
    """load the config in the model checkpoint and then overwrite it
        with whatever is provided
    """
    checkpoint_cfg_2_0 = load_2_0_checkpoint_model_config(restore_path)
    if checkpoint_cfg_2_0 is not None:
        checkpoint_cfg = checkpoint_cfg_2_0
    else:
        checkpoint_cfg = load_checkpoint_model_config(restore_path)

    if remove_meta_info:
        checkpoint_cfg.pop("target", None)
        checkpoint_cfg.pop("nemo_version", None)

    if "overwrite_base_config" in model_cfg_to_overwrite:
        remove_overwritten_fields(checkpoint_cfg, model_cfg_to_overwrite.overwrite_base_config)

    merged_cfg = OmegaConf.merge(checkpoint_cfg, model_cfg_to_overwrite)

    # Remove the "overwrite_base_config" key to avoid cluttering the model config.
    merged_cfg.pop("overwrite_base_config", None)

    return merged_cfg


def remove_overwritten_fields(base_config, overwrite_config):
    """
    Remove from `base_config` fields associated to a `True` value in `overwrite_config`.
    """
    for key, value in overwrite_config.items():
        if key not in base_config:
            continue

        if isinstance(value, DictConfig) and isinstance(base_config[key], DictConfig):
            remove_overwritten_fields(base_config[key], value)
        else:
            assert isinstance(value, bool), "the overwrite config can only contain boolean values"
            if value:
                base_config.pop(key)


def surpress_user_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output

    return wrapper


# need to surpress the masked tensor warnings from pytorch
@surpress_user_warnings
def masked_mean(values, mask, dim=None):
    """
    Masks values with mask, and computes the mean of the values using the masked values.
    """
    if dim is None:
        return values[mask.bool()].mean()
    return as_masked_tensor(values, mask.bool()).mean(dim=dim).to_tensor(torch.nan)


# need to surpress the masked tensor warnings from pytorch
@surpress_user_warnings
def masked_std(values, mask, dim=None):
    """
    Masks values with mask, and computes the std of the values using the masked values.
    """
    if dim is None:
        return values[mask.bool()].std()
    return as_masked_tensor(values, mask.bool()).std(dim=dim).to_tensor(torch.nan)


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


def get_global_set(local_data_ids: set):
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, local_data_ids)
    global_set = set().union(*output)

    return global_set


def calculate_response_lengths(tokens, eos_id):
    """calculates the response length of the tokens after padding"""
    return (tokens != eos_id).sum(-1)


def configure_batch_sizes(mbs, gbs, dp=1):
    app_state = AppState()
    reconfigure_num_microbatches_calculator(
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
def offload_distributed_adam(state_dict, force_clear_memory=False):
    """context manager to offload distributed adam states
    """
    # off load onto cpu
    for state_bucket in state_dict["state"]["buckets"]:
        dist_adam_load_state_bucket_into_device(state_bucket, device="cpu")

    # make sure the offloading is finished before returning
    torch.cuda.synchronize()

    if force_clear_memory:
        clear_memory()

    try:
        yield

    finally:
        # onload back onto gpu
        for state_bucket in state_dict["state"]["buckets"]:
            dist_adam_load_state_bucket_into_device(state_bucket, device=torch.cuda.current_device())

        # make sure the onloading is finished before returning
        torch.cuda.synchronize()


def batch_pad_to_fixed_len(batch, max_batch_len, pad_token):
    batch_pad = torch.stack(
        [torch.cat([seq, torch.full((max_batch_len - len(seq),), pad_token, dtype=seq.dtype),]) for seq in batch]
    )

    return batch_pad


def collate_with_batch_max_sequence_length(
    data_batch,
    response_token_length,
    eos_id,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    generate_masks_and_position_ids,
):
    """collate function that batches by max sequence length
    """
    texts = [item["text"] for item in data_batch]
    loss_multipliers = torch.as_tensor([item["loss_multiplier"] for item in data_batch]).view(len(data_batch), 1)
    lengths = torch.as_tensor([item["length"] for item in data_batch])
    batch_max_length = lengths.max()

    texts = batch_pad_to_fixed_len(texts, batch_max_length + response_token_length, eos_id)

    output = {
        "text": texts,
        "length": lengths,
    }

    other = {}
    if generate_masks_and_position_ids:
        # NOTE: the attention mask is 1x1xSxS, which will broadcast on the batch dimension
        attention_masks, loss_masks, position_ids = get_ltor_masks_and_position_ids(
            texts, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss
        )
        other = {
            "attention_mask": attention_masks,
            # to preserve the loss mask from the dataset
            "loss_mask": loss_masks * loss_multipliers,
            "position_ids": position_ids,
        }

    return output | other


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
def copy_model_states_to_cpu(model, cpu_dict=None, megatron_amp_O2=True, sync=True, alias_non_tensor=False):
    """This function mutates the cpu_dict object to throw the model states into preallocated tensors(if they exist)
        for non tensors it will do a deepcopy, unless alias_non_tensor is True
    """
    if cpu_dict is None:
        cpu_dict = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            if name not in cpu_dict:
                cpu_dict[name] = torch.empty(
                    item.size(), dtype=item.dtype, layout=item.layout, device="cpu", pin_memory=True
                )
            cpu_dict[name].copy_(item, non_blocking=sync)
        elif alias_non_tensor:
            cpu_dict[name] = item
        else:
            cpu_dict[name] = deepcopy(item)

    if megatron_amp_O2:
        cpu_dict = convert_to_amp_o2_format(cpu_dict)

    if sync:
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


@contextmanager
def adapter_control(model):
    """Temporarily disable adapters and re-enable them after the operation
    """
    try:
        # Disable adapters before yielding control
        for _, module in model.named_modules():
            if isinstance(module, AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=False)
        yield
    finally:
        # Re-enable adapters after operation
        for _, module in model.named_modules():
            if isinstance(module, AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)


def convert_to_amp_o2_format(state_dict):
    """when amp_o2 is enabled, the model gets wrapped in a Float16Module which changes
        the keys and how it loads need to add module onto it
    """
    new_state_dict = {}

    for key, item in state_dict.items():
        if "model.module." not in key:
            key = key.replace("model.", "model.module.", 1)
        new_state_dict[key] = item

    return new_state_dict


def get_iterator_k_split_list(batch: List[str], num_microbatches: int) -> Iterator:
    """
    Generate an iterator to split a list into microbatches of equal size.

    Args:
        batch (List[str]): The list to be split into microbatches.
        num_microbatches (int): The number of microbatches to split the list into.

    Returns:
        Iterator: An iterator that yields the microbatches.
    """
    assert len(batch) % num_microbatches == 0, "Issue with batch size configuration!"
    batch_size_per_microbatch = len(batch) // num_microbatches
    microbatches = [
        batch[i * batch_size_per_microbatch : (i + 1) * batch_size_per_microbatch] for i in range(num_microbatches)
    ]
    return itertools.chain(microbatches)


def _get_autocast_dtype(precision: str):
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    if precision in [32, "32", "32-true"]:
        return torch.float
    if precision in [16, "16", "16-mixed"]:
        return torch.half
    raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')


# this function uses dataclasses.replace to create ShardedTensors/ShardedObjects from torch.Tensor and IOBytes objects
# based on the TP/PP/DP axis information taken from already existing ShardedTensors/Objects belonging to some input reference parameter
# this is useful for creating TP/PP/DP compliant ShardedDicts where the TP/PP/DP of each sharded tensor can be copied from some
# other model which acts as a reference for providing this TP/PP/DP information. We use this in SPIN
# to ensure that the reference policy inside SPIN is sharded along the correct axis during saving of the checkpoint by reading
# the TP/PP/DP information from the actor policy. The reference_param in the function below refers to the parameter which acts as
# the reference for what the TP/PP/DP information should be. This is not the same thing as the "reference policy" in SPIN/DPO
# NOTE: dataclasses.replace is out-of-place so this is safe
def make_sharded_tensors_from_reference(reference_param, model_param, prefix: str):
    if isinstance(reference_param, ShardedTensorFactory):
        return replace(reference_param, key=f"{prefix}.{reference_param.key}", data=model_param)
    if isinstance(reference_param, ShardedObject):
        return replace(reference_param, key=f"{prefix}.{reference_param.key}", data=model_param)

    assert (
        tuple(model_param.shape) == reference_param.local_shape
    ), f"Model shape ({tuple(model_param.shape)} does not match reference shape ({reference_param.local_shape})"
    return replace(reference_param, key=f"{prefix}.{reference_param.key}", data=model_param, dtype=model_param.dtype)


def log_memory(prefix):
    pyt = torch.cuda.memory_allocated() / (1024 ** 3)
    el = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / (1024 ** 3)
    logging.info(f"Mem Usage (GB) | {prefix} | pytorch:{pyt} total_occupied:{el} | memory_other_than_pyt:{el-pyt}")


def deprecated_in_version(version: str, message: str | None = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Construct the deprecation message
            func_name = func.__name__
            warn_message = (
                f"The function '{func_name}' is deprecated and will be removed in version {version}. "
                f"{message if message else ''}".strip()
            )
            warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
