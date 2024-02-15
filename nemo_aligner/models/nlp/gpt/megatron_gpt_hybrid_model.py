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
from typing import Any, Dict, List, Union

import torch
from apex.normalization import MixedFusedRMSNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm  # NOQA
from apex.transformer.layers.layer_norm import FastLayerNorm
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, listify_model
from megatron.core import InferenceParams, parallel_state
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

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


class FakeOptimizer:
    def __init__(self, policy_optimizer, value_optimizer):
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

    def zero_grad(self):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

    def _finish_bucket_grad_sync(self):
        self.policy_optimizer._finish_bucket_grad_sync()
        self.value_optimizer._finish_bucket_grad_sync()

    def allreduce_main_grads(self):
        self.policy_optimizer.allreduce_main_grads()
        self.value_optimizer.allreduce_main_grads()

    def try_grad_sync(self, params=None):
        if self.policy_optimizer.overlap_grad_sync:
            params = self.policy_optimizer.parameters()
            self.policy_optimizer.try_grad_sync(params)
        if self.value_optimizer.overlap_grad_sync:
            params = self.value_optimizer.parameters()
            self.value_optimizer.try_grad_sync(params)

    @property
    def no_sync(self):
        return self.policy_optimizer.no_sync

    @property
    def overlap_grad_sync(self):
        return self.policy_optimizer.overlap_grad_sync or self.value_optimizer.overlap_grad_sync


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
            value_loss = (values - rewards).pow(2).mul(self.value_loss_weight)
            value_loss = compute_masked_per_sample_average(value_loss, mask, dim=-1)

        return value_loss

    def compute_policy_loss(self, batch, logits):
        policy_loss = torch.tensor([0], dtype=torch.float32, device=torch.cuda.current_device())
        if batch["train_mode"][0] == TrainMode.POLICY_ONLY and (self.policy_loss_weight > 0):
            # slow on TP
            logits = gather_from_tensor_model_parallel_region(logits)
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
                logits = parallel_logits

                policy_loss = self.compute_policy_loss(batch, logits)
                value_loss = self.compute_value_loss(batch, values)

                loss = value_loss - policy_loss
                reduced_loss, reduced_value_loss, reduced_policy_loss = average_losses_across_data_parallel_group(
                    [loss, value_loss, policy_loss]
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

    def get_params_for_weight_decay_optimization(
        self, model: Union[torch.nn.Module, List[torch.nn.Module]], include_value_head=True
    ) -> Dict[str, torch.nn.Parameter]:
        """Divide params into with-weight-decay and without-weight-decay groups.

        Layernorms and biases will have no weight decay but the rest will.
        """
        modules = listify_model(model)
        weight_decay_params = {"params": []}
        no_weight_decay_params = {"params": [], "weight_decay": 0.0}
        weight_decay_params_names = []
        no_weight_decay_params_names = []
        for module in modules:
            for n, p in module.named_parameters():
                if not include_value_head:
                    if "value_head" in n or "rm_head" in n:
                        continue
                else:
                    if "value_head" not in n and "rm_head" not in n:
                        continue
                if "layer_norm" in n or "bias" in n or "layernorm" in n:
                    no_weight_decay_params["params"].append(p)
                    no_weight_decay_params_names.append(n)
                else:
                    weight_decay_params["params"].append(p)
                    weight_decay_params_names.append(n)
        return weight_decay_params, no_weight_decay_params

    def setup_optimizer_param_groups(self):
        if self.setup_policy_optimizer:
            self._optimizer_param_groups = self.get_params_for_weight_decay_optimization(
                self.model, include_value_head=False
            )
        else:
            self._optimizer_param_groups = self.get_params_for_weight_decay_optimization(
                self.model, include_value_head=True
            )

    def configure_optimizers(self):
        # the pytorch lightning hook will call this function to setup the optimizer
        # returns a list of optimizers and a list of lr schedulers

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                modules = self.get_gpt_module_list()
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[0]  # only the first virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get("share_embeddings_and_output_weights", True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_gpt
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get("share_embeddings_and_output_weights", True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_gpt
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, "sequence_parallel", False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets_policy = []
            buckets_value = []
            if self.cfg.get("virtual_pipeline_model_parallel_size", None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    stage_bucket = []
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    for layer in layers:
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, "_disable_overlap_grad_sync", False)
                        )
                    buckets_policy.append(stage_bucket)
                    stage_bucket = []
                    if hasattr(module, "value_head"):
                        layers = module.value_head.layers if self.mcore_gpt else None
                        for layer in layers:
                            stage_bucket.extend(
                                p for p in layer.parameters() if not getattr(p, "_disable_overlap_grad_sync", False)
                            )
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    for layer in layers:
                        buckets_policy.append(
                            [p for p in layer.parameters() if not getattr(p, "_disable_overlap_grad_sync", False)]
                        )
                    if hasattr(module, "value_head"):
                        layers = module.value_head.layers if self.mcore_gpt else None
                        for layer in layers:
                            buckets_value.append(
                                [p for p in layer.parameters() if not getattr(p, "_disable_overlap_grad_sync", False)]
                            )
            buckets_policy.reverse()
            buckets_value.reverse()
            used_params = set()
            for bucket in buckets_value:
                used_params.update(bucket)

            buckets_value_remaining_params = []
            if self.cfg.get("virtual_pipeline_model_parallel_size", None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    if hasattr(module, "value_head"):
                        buckets_value_remaining_params.extend(
                            p for p in module.value_head.parameters() if p not in used_params
                        )
                    if hasattr(module, "rm_head"):
                        buckets_value_remaining_params.extend(
                            p for p in module.rm_head.parameters() if p not in used_params
                        )
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    if hasattr(module, "value_head"):
                        buckets_value_remaining_params.extend(
                            [p for p in module.value_head.parameters() if p not in used_params]
                        )
                    if hasattr(module, "rm_head"):
                        buckets_value_remaining_params.extend(
                            [p for p in module.rm_head.parameters() if p not in used_params]
                        )

            used_params = set()
            # load all the policy bucket parameters
            for bucket in buckets_policy:
                used_params.update(bucket)
            # load all the value bucket parameters
            for bucket in buckets_value:
                used_params.update(bucket)
            # load all the value parameters that are not in the buckets
            used_params.update(buckets_value_remaining_params)
            buckets_policy_remaining_params = [p for p in self.parameters() if p not in used_params]
            if buckets_policy_remaining_params:
                buckets_policy.append(buckets_policy_remaining_params)
            if buckets_value_remaining_params:
                buckets_value.append(buckets_value_remaining_params)

        optim_kwargs = {}
        if self.with_distributed_adam:

            # Allocate contiguous buffer to avoid extra copies
            optim_kwargs["contiguous_grad_buffer"] = True

            # Make sure optimizer state is in FP32
            optim_dtype = torch.float32
            optim_kwargs["dtype"] = optim_dtype

            # Make sure embedding grad reductions are in FP32
            for name, param in self.named_parameters():
                if "word_embedding" in name or "position_embedding" in name or "output_layer" in name:
                    param._with_fp32_optimizer = True

            # Match param allgather with model dtype
            model_dtype = torch.float32
            if self.megatron_amp_O2 and hasattr(self, "autocast_dtype"):
                model_dtype = self.autocast_dtype
            optim_kwargs["param_sync_dtype"] = model_dtype

            # Determine whether to store master params in optimizer
            if optim_dtype == model_dtype:
                optim_kwargs["store_params"] = False
            elif optim_dtype == torch.float32 and model_dtype == torch.bfloat16:
                optim_kwargs["store_params"] = False
                optim_kwargs["store_param_remainders"] = True
            else:
                optim_kwargs["store_params"] = True

        self.setup_policy_optimizer = True
        self.policy_optimizer, self.policy_scheduler = ModelPT.setup_optimization(
            self, optim_config=self.cfg.optim, optim_kwargs=optim_kwargs
        )
        # sanity check, make sure the number of parameters in the policy network is the same as number of parameters in the policy buckets
        if self.with_distributed_adam:
            assert sum([len(i["params"]) for i in self._optimizer_param_groups]) == sum(
                [len(i) for i in buckets_policy]
            )
        self.setup_policy_optimizer = False
        self.value_optimizer, self.value_scheduler = ModelPT.setup_optimization(
            self, optim_config=self.cfg.value.optim, optim_kwargs=optim_kwargs
        )

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_O2 and not self.with_distributed_adam:
            if self.torch_dtype == torch.bfloat16:
                fp32_grad_accum = True
                contiguous_grad_bucket = True
            elif self.torch_dtype == torch.float16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False
                raise ValueError(
                    "fp16 training is not yet supported with O2. Please set megatron_amp_O2 to False in the model config."
                )

            # if using tensor parallel only, we automatically use async grad all-reduce
            # if using pipeline parallel or sequence parallel or gradient accumulation fusion, then we disable it
            if (
                self.cfg.get("pipeline_model_parallel_size", 1) == 1
                and not (
                    self.cfg.get("sequence_parallel", False) or self.cfg.get("gradient_accumulation_fusion", False)
                )
                and self.cfg.get("async_grad_allreduce", True)
            ):
                async_grad_allreduce = True
            else:
                async_grad_allreduce = False

            if async_grad_allreduce:
                # we need this to be configurable until make_nccl_premul_sum is in public PyTorch.
                # currently cannot be imported in PyTorch 1.12.0
                grad_div_ar_fusion = self.cfg.get("grad_div_ar_fusion", False)
            else:
                grad_div_ar_fusion = False

            self.policy_optimizer = MainParamsOptimizerWrapper(
                self.policy_optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
                grad_div_ar_fusion=grad_div_ar_fusion,
                grad_allreduce_chunk_size_mb=self.cfg.get("grad_allreduce_chunk_size_mb", 125),
            )

            self.value_optimizer = MainParamsOptimizerWrapper(
                self.value_optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
                grad_div_ar_fusion=grad_div_ar_fusion,
                grad_allreduce_chunk_size_mb=self.cfg.get("grad_allreduce_chunk_size_mb", 125),
            )

            assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
            if hasattr(self._cfg.optim, "sched"):
                sched_config = self._cfg.optim.sched
                sched_config["max_steps"] = self._trainer.max_steps
                self.policy_scheduler = prepare_lr_scheduler(
                    optimizer=self.policy_scheduler, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

            if hasattr(self._cfg.value.optim, "sched"):
                sched_config = self._cfg.value.optim.sched
                sched_config["max_steps"] = self._trainer.max_steps
                self.value_scheduler = prepare_lr_scheduler(
                    optimizer=self.value_scheduler, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

        if self.with_distributed_adam:
            assert sum([len(i["params"]) for i in self._optimizer_param_groups]) == sum(
                [len(i) for i in buckets_value]
            )

            for bucket in buckets_policy:
                self.policy_optimizer.init_params_bucket(bucket)
            for bucket in buckets_value:
                self.value_optimizer.init_params_bucket(bucket)

            # Make sure all params are initialized so main grads are
            # available
            # Note: Consolidate grads without overlap
            policy_overlap_params = []
            policy_no_overlap_params = []
            value_overlap_params = []
            value_no_overlap_params = []
            for n, p in self.named_parameters():
                if "value_head" in n or "rm_head" in n:
                    if getattr(p, "_disable_overlap_grad_sync", False):
                        value_no_overlap_params.append(p)
                    else:
                        value_overlap_params.append(p)
                else:
                    if getattr(p, "_disable_overlap_grad_sync", False):
                        policy_no_overlap_params.append(p)
                    else:
                        policy_overlap_params.append(p)
            assert len(policy_overlap_params) + len(policy_no_overlap_params) == sum([len(i) for i in buckets_policy])
            assert len(value_overlap_params) + len(value_no_overlap_params) == sum([len(i) for i in buckets_value])
            self.policy_optimizer.init_params(reversed(policy_overlap_params))
            self.policy_optimizer.init_params(reversed(policy_no_overlap_params))
            self.value_optimizer.init_params(reversed(value_overlap_params))
            self.value_optimizer.init_params(reversed(value_no_overlap_params))
            if self.policy_optimizer.contiguous_param_buffer:
                self.policy_optimizer.init_param_buffer()
            if self.value_optimizer.contiguous_param_buffer:
                self.value_optimizer.init_param_buffer()

        self._optimizer = FakeOptimizer(self.policy_optimizer, self.value_optimizer)
