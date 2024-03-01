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

import re
import warnings
from collections import OrderedDict

# from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel
from contextlib import nullcontext
from typing import Any, Dict, List, Union

import torch
from apex.normalization import MixedFusedRMSNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm  # NOQA
from apex.transformer.layers.layer_norm import FastLayerNorm
from apex.transformer.pipeline_parallel.utils import (
    _reconfigure_microbatch_calculator,
    get_num_microbatches,
    listify_model,
)
from lightning_fabric.utilities.types import Optimizable
from megatron.core import InferenceParams, parallel_state
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.utils import divide, get_attr_wrapped_model
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.modelPT import ModelPT
from nemo.core.optim import MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.core.optim.distributed_adam import _str_to_dtype
from nemo.utils import logging
from nemo_aligner.algorithms.deepsearch import TrainMode
from nemo_aligner.models.nlp.gpt.gpt_hybrid_model import GPTHybridModel
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import configure_batch_sizes, masked_mean, offload_distributed_adam


def compute_masked_per_sample_average(tensor, mask, dim=-1):
    loss = (tensor * mask).sum(dim=dim)
    mask_num = mask.sum(dim=dim)
    return (loss / mask_num).mean()


def find_number_after_prefix(string, prefix):
    # Define the regex pattern to match the prefix followed by a number
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)(.*)")

    # Search for the pattern in the string
    match = pattern.search(string)

    # If a match is found, return the number (group 1 of the match)
    if match:
        return match.group(1), match.group(2)
    else:
        return None


class MegatronGPTHybridModel(MegatronGPTModel):
    """
    Megatron GPT Reward Model Training.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        if self.cfg.pipeline_model_parallel_size > 1 and not self.cfg.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )
        self.automatic_optimization = False

        reward_standardization = cfg.get("reward_standardization")

        self.enable_standardization = reward_standardization.enable if reward_standardization is not None else False

        if self.enable_standardization:
            self.rew_mean = cfg.reward_standardization.mean
            self.rew_std = cfg.reward_standardization.std

        self.to_offload_adam_states = self.cfg.offload_adam_states

        self.distributed_adam_offload_manager = None

        self.value_loss_weight = self.cfg.mcts.train.value_weight
        self.policy_loss_weight = self.cfg.mcts.train.policy_weight

        self.length_params = OmegaConf.to_container(self.cfg.inference.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.cfg.inference.sampling_params, resolve=True)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""

        force_head_dtype = self.cfg.get("force_head_dtype", torch.float32)
        head_dtype = None if force_head_dtype is None else _str_to_dtype(force_head_dtype)
        if self.cfg.get("megatron_amp_O2", False) and (head_dtype is None or torch.finfo(head_dtype).bits < 32):
            logging.warning(
                "When `megatron_amp_O2` is enabled, it is recommended to set `force_head_dtype=32` "
                "to improve the convergence and accuracy of the model"
            )

        if self.cfg.get("share_embeddings_and_output_weights", False):
            logging.warning(
                "`share_embeddings_and_output_weights` is not supported with the reward model since we don't use the "
                "normal output layer. Overriding it to False"
            )
        # this is the hack to get the transformer config for value head
        # first need to backup current cfg
        bak_cfg = self.cfg
        self.cfg = self.cfg.value
        transformer_config_value = self.build_transformer_config()
        # restore the cfg
        self.cfg = bak_cfg

        model = GPTHybridModel(
            config=self.transformer_config,
            head_config=transformer_config_value,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=self.cfg.get("override_vocab_size", self.padded_vocab_size),
            max_sequence_length=self.cfg.get("encoder_seq_length", 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            rotary_percent=self.cfg.get("rotary_percentage", 1.0),
            seq_len_interpolation_factor=self.cfg.get("seq_len_interpolation_factor", None),
            output_sequence=self.cfg.get("output_sequence", False),
            use_avg_pool=self.cfg.get("use_avg_pool", False),
            head_dtype=head_dtype,
            num_attributes=self.cfg.get("regression", {}).get("num_attributes", 1),
            attribute_weights=self.cfg.get("regression", {}).get("attribute_weights", None),
            merge_attributes=self.cfg.get("regression", {}).get("merge_attributes", False),
        )
        return model

    def compute_value_loss(self, batch, values):
        value_loss = torch.tensor([0], dtype=torch.float32, device=torch.cuda.current_device())

        if batch["train_mode"][0] == TrainMode.VALUE_ONLY and (self.value_loss_weight > 0):
            rewards = batch["reward"].view(-1, 1)
            # TODO(geshen); the mask contains the last <extra_id_1> token as well
            # we may not want to include this, but it doesn't really matter
            # because the model is forced to stop by nemo generate anyway
            mask = batch["mcts_mask"]

            # TODO(geshen): change to cross entropy
            value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                values, rewards.broadcast_to(values.shape), reduction="none"
            )
            value_loss = compute_masked_per_sample_average(self.value_loss_weight * value_loss, mask, dim=-1)

        return value_loss

    def compute_policy_loss(self, batch, logits):
        policy_loss = torch.tensor([0], dtype=torch.float32, device=torch.cuda.current_device())
        if batch["train_mode"][0] == TrainMode.POLICY_ONLY and (self.policy_loss_weight > 0):
            # slow on TP
            logits = gather_from_tensor_model_parallel_region(logits)
            logits = logits[..., : self.tokenizer.vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            actions = batch["actions"]
            action_probs = batch["action_probs"]
            rewards = batch["reward"]
            mask = batch["mcts_mask"]

            policy_mask = (rewards > 0) & mask
            policy_loss = (log_probs.gather(dim=-1, index=actions.long()) * action_probs).sum(-1)
            policy_loss = self.policy_loss_weight * policy_loss

            return compute_masked_per_sample_average(policy_loss, policy_mask, dim=-1)

        return policy_loss

    def get_hybrid_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            batch = next(data_iterator)

            tokens = batch["tokens"]

            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self.tokenizer.eos_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )

            batch = batch | {"position_ids": position_ids, "attention_mask": attention_mask}

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids", "train_mode"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("tokens", "actions", "action_probs", "reward", "mcts_mask", "train_mode"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            logits, values = model(batch["tokens"], batch["position_ids"], batch["attention_mask"], labels=None)

            def loss_func(parallel_logits):
                # num_micro_batches = batch["num_micro_batches"]
                amount_of_batches = batch["amount_of_batches"]
                divisor = amount_of_batches

                logits = parallel_logits

                policy_loss = self.compute_policy_loss(batch, logits)
                value_loss = self.compute_value_loss(batch, values)

                loss = (value_loss - policy_loss) / divisor
                reduced_loss, reduced_value_loss, reduced_policy_loss = average_losses_across_data_parallel_group(
                    [loss, value_loss / divisor, policy_loss / divisor]
                )

                return (
                    loss,
                    {
                        "loss": reduced_loss.detach(),
                        "value_loss": reduced_value_loss.detach(),
                        "policy_loss": -reduced_policy_loss.detach(),
                    },
                )

            return logits, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only):
        # TODO(geshen): on the value because the batches are differently
        # amount of num micro batches, we may hang
        gbs_by_dp, sequence_length = batch["tokens"].size()

        if batch["train_mode"] == TrainMode.VALUE_ONLY:
            # TODO(geshen): if this hangs, fix it...
            # hack... this actually tosses out data :(
            # TODO: remove  below
            # divide out because the value might have a weird number of samples
            # that is not divisible by the micro batch size
            num_micro_batches = gbs_by_dp // self.cfg.micro_batch_size
            output = torch.tensor([num_micro_batches], dtype=torch.long, device=torch.cuda.current_device())
            torch.distributed.all_reduce(
                output, op=torch.distributed.ReduceOp.MIN, group=parallel_state.get_data_parallel_group()
            )
            num_micro_batches = output.item()
            # TODO: remove above
        else:
            num_micro_batches = divide(gbs_by_dp, self.cfg.micro_batch_size)

        batch["train_mode"] = torch.as_tensor([batch["train_mode"]] * gbs_by_dp)
        batch["amount_of_batches"] = torch.as_tensor([batch["amount_of_batches"]] * gbs_by_dp)
        batch["num_micro_batches"] = torch.as_tensor([num_micro_batches] * gbs_by_dp)

        batch = {k: v[: num_micro_batches * self.cfg.micro_batch_size] for k, v in batch.items()}

        data_iter = get_iterator_k_split(batch, num_micro_batches)

        set_sync_funcs(self, forward_only)
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_hybrid_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_micro_batches,
            forward_only=forward_only,
            seq_length=sequence_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        metrics = {}

        for key in ["loss", "value_loss", "policy_loss"]:
            if losses_reduced_per_micro_batch:
                metric_mean = torch.stack(
                    [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                ).mean()
            else:
                metric_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            torch.distributed.broadcast(metric_mean, get_last_rank())

            metrics[key] = metric_mean.cpu().item()

        return metrics["loss"], metrics

    def generate(self, inputs, length_params=None, sampling_params=None):
        length_params = length_params or self.length_params
        sampling_params = sampling_params or self.sampling_params

        return super().generate(inputs, length_params, sampling_params)

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                if self.mcore_gpt:
                    # if first step, then clear KV cache, otherwise reuse inference_paarms
                    if set_inference_key_value_memory[0].item():
                        self.inference_params = InferenceParams(
                            max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                        )
                    extra_arg["inference_params"] = self.inference_params
                else:
                    extra_arg["set_inference_key_value_memory"] = set_inference_key_value_memory[0].item()
                    extra_arg["inference_max_sequence_len"] = inference_max_sequence_len[0].item()
            output_tensor, value = model(tokens, position_ids, attention_mask, **extra_arg)
            value = value.sigmoid()

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {"logits": output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def on_load_checkpoint(self, checkpoint) -> None:
        """NOTE: Have to set strict to False because we have a rm head
        """
        # mcore uses distributed checkpointing
        if "state_dict" in checkpoint and checkpoint["state_dict"]:
            for index, module in enumerate(self.get_gpt_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
                else:
                    checkpoint_state_dict = checkpoint["state_dict"]
                # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                checkpoint_state_dict = {
                    key.replace("model.", ""): checkpoint_state_dict.pop(key)
                    for key in list(checkpoint_state_dict.keys())
                }

                layer_offset = len(get_attr_wrapped_model(self.model, "decoder").submodules.layer_specs)

                with torch.no_grad():
                    print("#### PRE LOAD VALUE", self.model.module.value_head.layers[0].mlp.linear_fc1.weight.sum())
                    print("#### PRE LOAD STEM", self.model.module.decoder.layers[0].mlp.linear_fc1.weight.sum())

                modified_dict = {}
                prefix_to_use = "value_head.layers."

                for k, v in checkpoint_state_dict.items():
                    output = find_number_after_prefix(k, prefix=prefix_to_use)

                    if output is not None:
                        num, rest_to_use = output
                        k = "{}{}{}".format(prefix_to_use, int(num) - layer_offset, rest_to_use)
                    modified_dict[k] = v

                module.load_state_dict(modified_dict, strict=self.cfg.from_mcts_trained)
                with torch.no_grad():
                    print("#### POST LOAD VALUE", self.model.module.value_head.layers[0].mlp.linear_fc1.weight.sum())
                    print("#### POST LOAD STEM", self.model.module.decoder.layers[0].mlp.linear_fc1.weight.sum())
        else:
            # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
            # see NLPModel.on_load_checkpoint
            checkpoint["state_dict"] = {}

    def sharded_state_dict(self, prefix: str = "") -> Dict[str, Any]:
        """
        Creates the sharded state dict which is used by dist_checkpoint to save the sharded tensors to disk.
        When given the sharded_stated_dict, dist_checkpoint.load will load the tensors corresponding to
        self.state_dict().
        The sharded tensor mapping is defined in the GPTModel class from mcore.
        """

        if self.mcore_gpt:
            module_prefix = f"{prefix}model."
            sharded_state_dict = self.model.sharded_state_dict(prefix=module_prefix)
            return sharded_state_dict

    def offload_adam_states(self):
        if self.distributed_adam_offload_manager is None:

            self.distributed_adam_offload_manager = (
                offload_distributed_adam(self._optimizer.state_dict(state_dict_format=1, gather_on_root=False))
                if self.to_offload_adam_states and self.with_distributed_adam
                else nullcontext()
            )

            # offload onto cpu
            self.distributed_adam_offload_manager.__enter__()

    def onload_adam_states(self):
        if self.distributed_adam_offload_manager is not None:
            # load back onto GPU
            self.distributed_adam_offload_manager.__exit__(None, None, None)

        self.distributed_adam_offload_manager = None

    def prepare_for_inference(self):
        """normally we would configure the micro batch calculator here
            but the nemo generation already does the configuration"""
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)
        self.offload_adam_states()

    def finish_inference(self):
        # training will onload the adam states, no need to onload it here
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()
        set_train(self)

    def prepare_for_training(self):
        # configure_batch_sizes(
        # mbs=self.cfg.micro_batch_size,
        # gbs=self.cfg.global_batch_size,
        # dp=parallel_state.get_data_parallel_world_size(),
        # )
        self.onload_adam_states()

    def finish_training(self):
        """no need to offload adam states here
        """

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)
