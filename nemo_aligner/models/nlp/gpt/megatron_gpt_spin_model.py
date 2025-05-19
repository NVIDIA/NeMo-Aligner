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
from contextlib import nullcontext

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
#from nemo_aligner.models.nlp.gpt.gpt_inference_model import GPTInferenceModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor, from_parallel_logits_to_logprobs
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import (
    configure_batch_sizes,
    cpu_weight_swap,
    make_sharded_tensors_from_reference,
    offload_distributed_adam,
)


class MegatronGPTSPINModel(MegatronGPTModel, SupervisedInterface):
    """
    Megatron GPT SPIN Model Training
    Adapted from the paper Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (Chen, et al, 2024)
    https://arxiv.org/abs/2401.01335
    
    Our implementation differs in that we do not have a scheduler for the KL divergence parameter and we do not currently
    inject generations from iteration t-1 into iteration t.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        if self.cfg.pipeline_model_parallel_size > 1 and not self.cfg.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )

        self.ref_policy_state_dict = None
        self.distributed_adam_offload_manager = None
        self.to_offload_adam_states = self.cfg.spin.offload_adam_states

        # self.ref_policy_kl_penalty = self.cfg.spin.get("ref_policy_kl_penalty", 0.0)
        self.spin_config = OmegaConf.to_container(self.cfg.spin, resolve=True)
        if isinstance(self.spin_config["ref_policy_kl_penalty"], (float, int)):
            self.ref_policy_kl_penalty = self.spin_config["ref_policy_kl_penalty"]
        elif isinstance(self.spin_config["ref_policy_kl_penalty"], list):
            self.ref_policy_kl_penalty = 0.0
        else:
            raise TypeError(
                f"`ref_policy_kl_penalty` must be a scalar or list, but got {type(self.spin_config['ref_policy_kl_penalty'])}"
            )

        # RPO params
        self.preference_avg_log_probs = self.cfg.spin.get("preference_average_log_probs", False)
        self.sft_avg_log_probs = self.cfg.spin.get("sft_average_log_probs", self.preference_avg_log_probs)

        self.preference_loss_weight = self.cfg.spin.get("preference_loss_weight", 1)
        self.sft_loss_weight = self.cfg.spin.get("sft_loss_weight", 0)
        assert (
            self.preference_loss_weight != 0 or self.sft_loss_weight != 0
        ), "sft loss weight and dpo loss weight cannot both be 0"

        # variants of preference losses, by default DPO.
        self.preference_loss = self.cfg.spin.get("preference_loss", "dpo")
        self.gt_reward_scale = self.cfg.spin.get("gt_reward_scale", 1.0)
        
        # beta-DPO params
        self.dynamic_kl = self.cfg.spin.get("dynamic_kl", False)
        self.M_0 = 1.0
        self.momentum = 0.9
        self.alpha = 0.2
        self.B_0 = 0.1

    @torch.no_grad()
    def gather_and_split_rewards(self, pi_logprobs, ref_logprobs, masks, average_log_probs=False):
        pi_logprobs = pi_logprobs.detach()

        dp_group = parallel_state.get_data_parallel_group()

        batch_logs = self.get_reduced_masked_logps(
            pi_logprobs - ref_logprobs, masks[:, 1:], average_log_probs=average_log_probs
        )

        output_list = [torch.zeros_like(batch_logs) for _ in range(dp_group.size())]

        torch.distributed.all_gather(output_list, batch_logs, group=dp_group)

        split_iter = map(self.split_output_tensor, output_list)

        out_chosen, out_rejected = map(torch.cat, zip(*split_iter))

        return out_chosen.flatten(), out_rejected.flatten()

    def get_forward_output_and_loss_func(self, validation_step=False, logprobs_only=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # there is a problem with apex ignoring the mask on the older models
                # so we will always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("chosen", "rejected", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(
                        (
                            "chosen",
                            "rejected",
                            "ref_policy_log_probs_chosen",
                            "ref_policy_log_probs_rejected",
                            "chosen_mask",
                            "rejected_mask",
                            "chosen_rewards",
                            "rejected_rewards",
                        )
                    )

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            tokens, masks, ref_logprobs, gt_rewards = None, None, None, None
            if batch["chosen"] is not None and batch["rejected"] is not None:
                tokens = torch.cat((batch["chosen"], batch["rejected"]), dim=0)

            if batch["chosen_mask"] is not None and batch["rejected_mask"] is not None:
                masks = torch.cat((batch["chosen_mask"], batch["rejected_mask"]), dim=0)

            if (
                batch.get("ref_policy_log_probs_chosen") is not None
                and batch.get("ref_policy_log_probs_rejected") is not None
            ):
                ref_logprobs = torch.cat(
                    (batch["ref_policy_log_probs_chosen"], batch["ref_policy_log_probs_rejected"]), dim=0
                )

            if batch.get("chosen_rewards") is not None and batch.get("rejected_rewards") is not None:
                gt_rewards = torch.cat((batch["chosen_rewards"], batch["rejected_rewards"]), dim=0)

            # this is necessary if MBS > 1 with the new GBS padding logic, as you may get batch dim > 1 in some configs
            # these two lines ensure your position_ids and attn_mask are always B=1
            # position_ids = batch["position_ids"][0:1]
            attention_mask = batch["attention_mask"][0:1]

            # Model forward pass
            forward_args = {
                "input_ids": tokens,
                "position_ids": batch["position_ids"],
                "attention_mask": attention_mask,
                "labels": None,
                "loss_mask": None,
            }

            # TODO: we can remove this someday when we no longer support legacy models
            if not self.mcore_gpt:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop("loss_mask")
            else:
                forward_args.pop("loss_mask")

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def logprobs_func(output_tensor, non_loss_data=True):
                # This function is expected to be used only when `collect_non_loss_data=True` in the fwd_bwd_function of Megatron-LM.
                # See https://github.com/NVIDIA/Megatron-LM/blob/0bc3547702464501feefeb5523b7a17e591b21fa/megatron/core/pipeline_parallel/schedules.py#L228
                assert non_loss_data
                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor, target=tokens, inference_only=True, higher_stability=True,
                )
                return {"logprobs": logprobs}

            def loss_func(output_tensor):
                if validation_step and not self.cfg.data.validation_ds.get("drop_last", True):
                    raise NotImplementedError(
                        "SPIN does not support validation when `cfg.data.validation_ds.drop_last=False`"
                    )

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=tokens,
                    higher_stability=True,
                    inference_only=validation_step,
                )

                preference_loss, acc_chosen = self.loss_func(
                    per_token_logps,
                    ref_logprobs,
                    masks[:, 1:],
                    gt_rewards,
                    average_log_probs=self.preference_avg_log_probs,
                )

                sft_loss = torch.zeros_like(preference_loss)
                if self.sft_loss_weight != 0:
                    sft_loss = self.sft_loss_func(
                        per_token_logps, masks[:, 1:], average_log_probs=self.sft_avg_log_probs
                    )
                loss = self.preference_loss_weight * preference_loss + self.sft_loss_weight * sft_loss

                (
                    reduced_loss,
                    reduced_preference_loss,
                    reduced_sft_loss,
                    reduced_acc,
                ) = average_losses_across_data_parallel_group([loss, preference_loss, sft_loss, acc_chosen])

                out_chosen, out_rejected = self.gather_and_split_rewards(
                    per_token_logps, ref_logprobs, masks, average_log_probs=self.preference_avg_log_probs
                )

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "avg_sft_loss": reduced_sft_loss,
                        "avg_preference_loss": reduced_preference_loss,
                        "acc": reduced_acc,
                        "out_chosen": out_chosen,
                        "out_rejected": out_rejected,
                    },
                )

            if logprobs_only:
                return output_tensor, logprobs_func
            else:
                return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor):
        chosen_logps, reject_logps = torch.split(output_tensor.float(), len(output_tensor) // 2, dim=0)
        return chosen_logps, reject_logps

    def get_reduced_masked_logps(self, logps, loss_mask, average_log_probs=False):
        assert logps.shape == loss_mask.shape, "logps and loss_mask shape mismatch"

        loss_mask = loss_mask.float()

        if average_log_probs:
            # need to guard against divide by zero in case labels are all -100
            return (logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (logps * loss_mask).sum(-1)
    
    @torch.no_grad()
    def update_kl_across_ranks(self, local_batch_Mi):
        local_batch_Mi = local_batch_Mi.detach()
        
        # average down to scalar
        local_batch_Mi_mean = local_batch_Mi.mean(0)
        
        # all-reduce the scalar means from all other ranks
        tensor_to_accumulate = torch.tensor([local_batch_Mi_mean], dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        
        global_Mi_mean = tensor_to_accumulate[0].item()
        
        self.M_0 = self.momentum * self.M_0 + (1.0 - self.momentum) * global_Mi_mean
        
        self.ref_policy_kl_penalty = (1.0 + self.alpha * (global_Mi_mean - self.M_0)) * self.B_0

    def loss_func(self, pi_logprobs, ref_logprobs, masks, gt_rewards, average_log_probs=False):
        rewards = self.get_reduced_masked_logps(pi_logprobs - ref_logprobs, masks, average_log_probs=average_log_probs)

        chosen_rewards, reject_rewards = self.split_output_tensor(rewards)
        rewards_delta = chosen_rewards - reject_rewards
        
        if self.dynamic_kl:
            self.update_kl_across_ranks(rewards_delta)

        if self.preference_loss == "dpo":
            loss = -torch.nn.functional.logsigmoid(self.ref_policy_kl_penalty * rewards_delta).mean(0)
        elif self.preference_loss == "scale":
            chosen_gt_rewards, reject_gt_rewards = self.split_output_tensor(gt_rewards)
            abs_margin = torch.abs(chosen_gt_rewards - reject_gt_rewards)
            loss = abs_margin * -torch.nn.functional.logsigmoid(self.ref_policy_kl_penalty * rewards_delta).mean(0)
        elif self.preference_loss == "rpo_bwd_kl":
            logbeta_hat_chosen = torch.nn.functional.logsigmoid(self.ref_policy_kl_penalty * rewards_delta)
            logbeta_hat_rejected = torch.nn.functional.logsigmoid(-self.ref_policy_kl_penalty * rewards_delta)

            chosen_gt_rewards, reject_gt_rewards = self.split_output_tensor(gt_rewards)
            gt_rewards_delta = self.gt_reward_scale * (chosen_gt_rewards - reject_gt_rewards)
            logalpha_hat_chosen = torch.nn.functional.logsigmoid(gt_rewards_delta)
            logalpha_hat_rejected = torch.nn.functional.logsigmoid(-gt_rewards_delta)

            loss = (
                torch.exp(logalpha_hat_chosen) * (logalpha_hat_chosen - logbeta_hat_chosen)
                + torch.exp(logalpha_hat_rejected) * (logalpha_hat_rejected - logbeta_hat_rejected)
            ).mean(0)
        elif self.preference_loss == "rpo_fwd_kl":
            logbeta_hat_chosen = torch.nn.functional.logsigmoid(self.ref_policy_kl_penalty * rewards_delta)
            logbeta_hat_rejected = torch.nn.functional.logsigmoid(-self.ref_policy_kl_penalty * rewards_delta)

            chosen_gt_rewards, reject_gt_rewards = self.split_output_tensor(gt_rewards)
            gt_rewards_delta = self.gt_reward_scale * (chosen_gt_rewards - reject_gt_rewards)
            logalpha_hat_chosen = torch.nn.functional.logsigmoid(gt_rewards_delta)
            logalpha_hat_rejected = torch.nn.functional.logsigmoid(-gt_rewards_delta)

            loss = (
                torch.exp(logbeta_hat_chosen) * (logbeta_hat_chosen - logalpha_hat_chosen)
                + torch.exp(logbeta_hat_rejected) * (logbeta_hat_rejected - logalpha_hat_rejected)
            ).mean(0)
        elif self.preference_loss == "ipo":
            loss = torch.mean((chosen_rewards - reject_rewards - 1.0 / (2.0 * self.ref_policy_kl_penalty)) ** 2, 0)
        elif self.preference_loss == "rpo_sq":
            chosen_gt_rewards, reject_gt_rewards = self.split_output_tensor(gt_rewards)
            gt_rewards_delta = self.gt_reward_scale * (chosen_gt_rewards - reject_gt_rewards)

            loss = torch.mean((self.ref_policy_kl_penalty * rewards_delta - gt_rewards_delta) ** 2, 0)
        else:
            raise NotImplementedError(f"preference_loss {self.preference_loss} is not implemented")

        with torch.no_grad():
            comp = chosen_rewards > reject_rewards
            acc_chosen = comp.float().mean()

        return loss, acc_chosen

    def sft_loss_func(self, pi_logprobs, labels, average_log_probs=False):
        logprobs = self.get_reduced_masked_logps(pi_logprobs, labels, average_log_probs=average_log_probs)
        chosen_logprobs, _ = self.split_output_tensor(logprobs)
        return -chosen_logprobs.mean(0)
    
    def loss_func_sft_workaround(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        losses = output_tensor.float()
        is_finite = losses.isfinite()
        loss_mask = loss_mask.view(-1).float()
        loss_mask = loss_mask * is_finite.view(-1)
        # TODO: add nemo version here
        loss = 1.0 * torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub.clamp(min=1)  # sequence level nll
        if parallel_state.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss

    def get_loss_and_metrics(self, batch, forward_only):
        seq_length = batch["chosen"].shape[1]

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only, logprobs_only=False),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size
            * 2,  # each minibatch has 2 comparisons so tensor shape will be mbs * 2
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            rewards_chosen = torch.cat([item["out_chosen"] for item in losses_reduced_per_micro_batch])
            rewards_rejected = torch.cat([item["out_rejected"] for item in losses_reduced_per_micro_batch])

            rewards_all = torch.cat((rewards_chosen, rewards_rejected))
            rewards_chosen_mean = rewards_chosen.mean()
            rewards_rejected_mean = rewards_rejected.mean()
            rewards_all_mean = rewards_all.mean()
            rewards_all_std = rewards_all.std()

            loss_mean = torch.as_tensor(
                [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
            sft_loss_mean = torch.as_tensor(
                [loss_reduced["avg_sft_loss"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
            preference_loss_mean = torch.as_tensor(
                [loss_reduced["avg_preference_loss"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
            acc_mean = torch.as_tensor(
                [loss_reduced["acc"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
        else:

            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            sft_loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            preference_loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            acc_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            rewards_chosen_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_rejected_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(sft_loss_mean, get_last_rank())
        torch.distributed.broadcast(preference_loss_mean, get_last_rank())
        torch.distributed.broadcast(acc_mean, get_last_rank())

        torch.distributed.broadcast(rewards_chosen_mean, get_last_rank())
        torch.distributed.broadcast(rewards_rejected_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "sft_loss": sft_loss_mean,
            "preference_loss": preference_loss_mean,
            "acc": acc_mean,
            "rewards_chosen_mean": rewards_chosen_mean,
            "rewards_rejected_mean": rewards_rejected_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}
        
        if self.dynamic_kl:
            metrics["dynamic_kl_kl_penalty"] = self.ref_policy_kl_penalty
            metrics["dynamic_kl_M_0"] = self.M_0

        return loss_mean.item(), metrics

    def get_loss_and_metrics_vanilla_sft(self, batch, forward_only):
        """Take a local batch as input and returns loss and metrics for a vanilla SFT loss
           meaning the loss for standard SFT and NOT SPIN loss. This is to speed up validation,
           as it can be argued that the goal of SPIN is to produce a quality SFT model so we train
           on SPIN but validate using vanilla SFT loss. This also bypasses needing to do costly generation
           for validation.
           
           TODO: possibly change to SPIN loss for validation once we have TRT-LLM
        """
        seq_length = batch["tokens"].shape[1]
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        set_sync_funcs(self, forward_only)

        orig_loss_func = self.loss_func
        #self.loss_func = super().loss_func
        self.loss_func = self.loss_func_sft_workaround

        fwd_bwd_function = get_forward_backward_func()
        fwd_loss_fn = super().get_forward_output_and_loss_func(forward_only)

        losses_reduced = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            seq_length=seq_length,
        )

        torch.cuda.synchronize()

        self.loss_func = orig_loss_func

        # only the last stages of the pipeline return losses
        if parallel_state.is_pipeline_last_stage():
            # average loss across micro batches
            loss_mean = torch.concat([loss_reduced["avg"] for loss_reduced in losses_reduced]).mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()
        # Logging
        torch.distributed.broadcast(loss_mean, get_last_rank())
        loss_value = loss_mean.detach().item()
        metrics = {"loss": loss_value}

        return loss_value, metrics

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()

    def finish_training_step(self):
        grad_reductions(self)

    def finish_training(self):
        """no need to offload adam states here
        """

    def prepare_for_validation(self):
        configure_batch_sizes(
            mbs=self.cfg.data.validation_ds.micro_batch_size,
            gbs=self.cfg.data.validation_ds.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)

    def finish_validation(self):
        """no need to offload adam states here
        """

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

    def offload_adam_states(self):
        if self.distributed_adam_offload_manager is None:

            self.distributed_adam_offload_manager = (
                offload_distributed_adam(
                    self._optimizer.state_dict(state_dict_format=1, gather_on_root=False), force_clear_memory=True
                )
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

    # Alternative 1: swap in the ref weights and create a custom key which then needs to be handled in the loader
    # this method is confirmed working, and is mothballed here in case we need to use it in the future
    def sharded_state_dict_alt_1(self, prefix: str = ""):
        sharded_state_dict_orig = super().sharded_state_dict(prefix=prefix)

        if self.ref_policy_state_dict is None:
            return sharded_state_dict_orig

        sharded_state_dict_new = {}
        with cpu_weight_swap(self, self.ref_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
            module_prefix = f"{prefix}ref_policy_model."
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    # virtual pipline rank must be set so that GPTModel returns the correct sharded state dict
                    parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict_new[f"ref_policy_model_{index}"] = module_sharded_state_dict
                else:
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict_new.update(module_sharded_state_dict)

            # reset vp rank
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        return sharded_state_dict_orig | sharded_state_dict_new

    # Alternative 2: use make_sharded_optimizer_tensor to link the TP/PP/DP info, but this may break under VP logic
    def sharded_state_dict(self, prefix: str = ""):
        sharded_state_dict_orig = super().sharded_state_dict(prefix=prefix)

        if self.ref_policy_state_dict is not None:
            # Link ref_policy keys with sharded_state_dict to reuse sharding information
            ref_policy_sharded_state_dict = {}
            for k, v in self.ref_policy_state_dict.items():
                if v is None:
                    continue
                key = k.replace("model.module.", "model.", 1) if self.megatron_amp_O2 else k
                assert (
                    key in sharded_state_dict_orig
                ), f"key [ {key} ] exists in ref_policy but not in sharded_state_dict_orig"  # may fail due to nesting?
                ref_policy_sharded_state_dict[k] = make_sharded_tensors_from_reference(
                    sharded_state_dict_orig[key], v, "reference_policy"
                )
            sharded_state_dict_orig["reference_policy"] = ref_policy_sharded_state_dict
        
        if self.dynamic_kl:
            dynamic_kl_dict = {"M_0": self.M_0, "ref_policy_kl_penalty": self.ref_policy_kl_penalty}
            sharded_state_dict_orig["dynamic_kl"] = dynamic_kl_dict

        return sharded_state_dict_orig

    # use this version for alternative 1
    # this method is confirmed working, and is mothballed here in case we need to use it in the future
    def on_load_checkpoint_alt_1(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """

        # mcore uses distributed checkpointing
        if self.mcore_gpt:
            # checkpoint keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
            if "state_dict" in checkpoint and checkpoint["state_dict"]:
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
                        checkpoint_state_dict.update(checkpoint["state_dict"][f"ref_policy_model_{index}"])
                    else:
                        checkpoint_state_dict = checkpoint["state_dict"]
                    # pull out the ref_policy_model parts first so it doesn't conflict with strict=True later
                    ref_state_dict = {
                        key.replace("ref_policy_model.", ""): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                        if "ref_policy_model." in key
                    }
                    # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                    checkpoint_state_dict = {
                        key.replace("model.", ""): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                    }
                    ref_policy = ref_state_dict
                    if ref_policy is not None and len(ref_policy) > 0:
                        # param_mean_ref = sum([v.mean().item() for k,v in ref_policy.items() if isinstance(v, torch.Tensor)])
                        # print(f"*** REF_MEAN_LOAD_RAW: {param_mean_ref}", flush=True)
                        self.ref_policy_state_dict = {
                            ("model." + ("module." if self.megatron_amp_O2 else "") + k): v
                            for k, v in ref_policy.items()
                        }
                    module.load_state_dict(checkpoint_state_dict, strict=True)
            else:
                # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
                # see NLPModel.on_load_checkpoint
                checkpoint["state_dict"] = {}

        # legacy checkpointing no longer supported (sorry)
        else:
            raise RuntimeError("legacy checkpoints are not supported by NeMo-Aligner")

    # use this one for alternative 2
    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """

        # mcore uses distributed checkpointing
        if self.mcore_gpt:
            # checkpoint keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
            if "state_dict" in checkpoint and checkpoint["state_dict"]:
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
                    else:
                        checkpoint_state_dict = checkpoint["state_dict"]
                    # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                    checkpoint_state_dict = {
                        key.replace("model.", ""): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                    }
                    ref_policy = checkpoint_state_dict.pop("reference_policy", None)
                    if ref_policy is not None and len(ref_policy) > 0:
                        # param_mean_ref = sum([v.mean().item() for k,v in ref_policy.items() if isinstance(v, torch.Tensor)])
                        # print(f"*** REF_MEAN_LOAD_RAW: {param_mean_ref}", flush=True)
                        self.ref_policy_state_dict = ref_policy
                    
                    dynamic_kl = checkpoint_state_dict.pop("dynamic_kl", None)
                    if dynamic_kl is not None and len(dynamic_kl) > 0 and self.cfg.spin.get("dynamic_kl", False):
                        self.M_0 = dynamic_kl["M_0"]
                        self.ref_policy_kl_penalty = dynamic_kl["ref_policy_kl_penalty"]
                    module.load_state_dict(checkpoint_state_dict, strict=True)
            else:
                # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
                # see NLPModel.on_load_checkpoint
                checkpoint["state_dict"] = {}

        # legacy checkpointing no longer supported (sorry)
        else:
            raise RuntimeError("legacy checkpoints are not supported by NeMo-Aligner")

    @torch.no_grad()
    def get_logprob_batch(self, batch):
        seq_length = batch["chosen"].shape[1]
        batch_size = batch["chosen"].shape[0]

        num_microbatches = divide(batch_size, min(batch_size, self.cfg.spin.log_prob_forward_micro_batch_size))
        data_iter = get_iterator_k_split(batch, num_microbatches)
        set_sync_funcs(self, forward_only=True)

        fwd_bwd_function = get_forward_backward_func()

        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(logprobs_only=True),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=min(batch_size, self.cfg.spin.log_prob_forward_micro_batch_size) * 2,
            collect_non_loss_data=True,
        )

        if len(logprobs_list) > 0:
            chosen_logprobs_list = []
            rejected_logprobs_list = []
            for item in logprobs_list:
                chosen_logprobs, rejected_logprobs = self.split_output_tensor(item["logprobs"])
                chosen_logprobs_list.append(chosen_logprobs)
                rejected_logprobs_list.append(rejected_logprobs)

            logprobs = torch.cat([torch.cat(chosen_logprobs_list), torch.cat(rejected_logprobs_list)], dim=0)
        else:
            logprobs = None

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            # broadcast it from last PP stage to everything else
            logprobs = broadcast_2d_tensor(
                logprobs,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                parallel_state.get_pipeline_model_parallel_group(),
            )

        return logprobs

    def get_ref_policy_logprobs(self, batch):
        with cpu_weight_swap(self, self.ref_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
            ref_log_probs = self.get_logprob_batch(batch)

        # return in GPU, trainer needs to move to cpu
        return ref_log_probs

    def set_KL_penalty_by_iteration(self, iteration):
        if isinstance(self.spin_config["ref_policy_kl_penalty"], (float, int)):
            return
        elif isinstance(self.spin_config["ref_policy_kl_penalty"], list):
            assert iteration < len(
                self.spin_config["ref_policy_kl_penalty"]
            ), f"iteration [ {iteration} ] is out of bounds for KL schedule {self.spin_config['ref_policy_kl_penalty']}"

            self.ref_policy_kl_penalty = self.spin_config["ref_policy_kl_penalty"][iteration]
