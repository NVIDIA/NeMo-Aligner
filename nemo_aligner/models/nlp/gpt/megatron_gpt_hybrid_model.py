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

import warnings

# from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel
from contextlib import nullcontext
from typing import Any, Dict

import torch
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_num_microbatches
from megatron.core import InferenceParams, parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.optim.distributed_adam import _str_to_dtype
from nemo.utils import logging
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
                    required_keys.update(("tokens", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("tokens", "context_lengths", "actions", "action_probs", "reward"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            parallel_logits = model(batch["tokens"], batch["position_ids"], batch["attention_mask"], labels=None,)

            def loss_func(parallel_logits):
                logits, values = parallel_logits
                # slow on TP
                logits = gather_from_tensor_model_parallel_region(logits)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                #

                lengths = batch["context_lengths"]
                actions = batch["actions"]
                action_probs = batch["action_probs"]
                rewards = batch["reward"]

                # context length has to be subtracted by 1 which is the last state before the action
                idx = (lengths - 1).unsqueeze(-1)

                value_loss = torch.tensor([0], dtype=torch.float32, device=torch.cuda.current_device())
                policy_loss = torch.tensor([0], dtype=torch.float32, device=torch.cuda.current_device())

                if self.value_loss_weight > 0:
                    values = values.gather(dim=-1, index=idx).flatten()
                    value_loss = (values - rewards.flatten()) ** 2

                    value_loss = (self.value_loss_weight * value_loss).mean()

                mask = rewards > 0

                if self.policy_loss_weight > 0 and torch.any(mask):
                    mbs = log_probs.size(0)
                    # get the last prediction and actions
                    log_probs = log_probs[torch.arange(mbs).view(mbs, 1), idx, actions]
                    # mbs x num actions
                    policy_loss = (action_probs * log_probs).sum(-1)

                    # mask ones that didn't get the right answer
                    policy_loss = masked_mean(self.policy_loss_weight * policy_loss, mask)

                loss = value_loss - policy_loss
                reduced_loss, reduced_value_loss, reduced_policy_loss = average_losses_across_data_parallel_group(
                    [loss, value_loss, policy_loss]
                )

                return (
                    loss,
                    {
                        "loss": reduced_loss.detach(),
                        "value_loss": reduced_value_loss.detach(),
                        "policy_loss": reduced_policy_loss.detach(),
                    },
                )

            return parallel_logits, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only):
        gbs_by_dp, sequence_length = batch["tokens"].size()

        num_micro_batches = divide(gbs_by_dp, self.cfg.micro_batch_size)
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
                module.load_state_dict(checkpoint_state_dict, strict=False)
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
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
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
