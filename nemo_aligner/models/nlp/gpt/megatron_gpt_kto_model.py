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
from functools import partial

import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils.distributed import broadcast_2d_tensor, from_parallel_logits_to_logprobs
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_sync_funcs,
)
from nemo_aligner.utils.utils import adapter_control, cpu_weight_swap


class MegatronGPTKTOModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    """
    Megatron GPT KTO Model Training.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        if self.cfg.pipeline_model_parallel_size > 1 and not self.cfg.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )
        self.automatic_optimization = False
        self.ref_policy_state_dict = None

        self.ref_policy_kl_penalty = self.cfg.kto.get("ref_policy_kl_penalty", 0.0)
        self.preference_avg_log_probs = self.cfg.kto.get("preference_average_log_probs", False)
        self.sft_avg_log_probs = self.cfg.kto.get("sft_average_log_probs", self.preference_avg_log_probs)

        self.preference_loss_weight = self.cfg.kto.get("preference_loss_weight", 1)
        self.sft_loss_weight = self.cfg.kto.get("sft_loss_weight", 0)
        assert (
            self.preference_loss_weight != 0 or self.sft_loss_weight != 0
        ), "sft loss weight and preference loss weight cannot both be 0"

        self.desirable_loss_weight = self.cfg.kto.get("desirable_loss_weight", 1.0)
        self.undesirable_loss_weight = self.cfg.kto.get("undesirable_loss_weight", 1.0)

    @torch.no_grad()
    def gather_and_split_rewards(self, pi_logprobs, ref_logprobs, labels, preferences, average_log_probs=False):
        pi_logprobs = pi_logprobs.detach()

        dp_group = parallel_state.get_data_parallel_group()

        batch_logs = self.get_reduced_masked_logps(
            pi_logprobs - ref_logprobs, labels[:, 1:], average_log_probs=average_log_probs
        )

        output_list = [torch.zeros_like(batch_logs) for _ in range(dp_group.size())]

        torch.distributed.all_gather(output_list, batch_logs, group=dp_group)

        # split_iter = map(self.split_output_tensor, output_list)
        split_iter = map(partial(self.split_output_tensor, preferences=preferences), output_list)

        out_chosen, out_rejected, kl_rewards = map(torch.cat, zip(*split_iter))

        return out_chosen.flatten(), out_rejected.flatten(), kl_rewards.clamp(min=0)

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
                    required_keys.update(("sample", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("ref_policy_log_probs", "sample_labels",))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            tokens, labels, ref_logprobs = None, None, None
            if batch["samples"] is not None and batch["kl_samples"] is not None:
                tokens = torch.cat((batch["samples"], batch["kl_samples"]), dim=0)

            if batch["sample_labels"] is not None and batch["kl_sample_labels"] is not None:
                labels = torch.cat((batch["sample_labels"], batch["kl_sample_labels"]), dim=0)

            if (
                batch["ref_policy_log_probs_samples"] is not None
                and batch["ref_policy_log_probs_kl_samples"] is not None
            ):
                ref_logprobs = torch.cat(
                    (batch["ref_policy_log_probs_samples"], batch["ref_policy_log_probs_kl_samples"]), dim=0
                )

            if batch["preference"] is not None:
                preferences = batch["preference"]

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
                    vocab_parallel_logits=output_tensor, target=labels, inference_only=True, higher_stability=True,
                )
                return {"logprobs": logprobs}

            def loss_func(output_tensor):
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    raise NotImplementedError("KTO does not support validation when cfg.data.drop_last=False")

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=labels,
                    inference_only=validation_step,
                    higher_stability=True,
                )

                loss, kl_divergence = self.loss_func(
                    per_token_logps,
                    ref_logprobs,
                    labels[:, 1:],
                    preferences,
                    average_log_probs=self.preference_avg_log_probs,
                )

                reduced_loss = average_losses_across_data_parallel_group([loss])

                out_chosen, out_rejected, kl_div = self.gather_and_split_rewards(
                    per_token_logps, ref_logprobs, labels, preferences, average_log_probs=self.preference_avg_log_probs
                )

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "kl": kl_div,
                        "out_chosen": out_chosen,
                        "out_rejected": out_rejected,
                    },
                )

            if logprobs_only:
                return output_tensor, logprobs_func
            else:
                return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor, preferences):
        logps, kl_logps = torch.split(output_tensor.float(), len(output_tensor) // 2, dim=0)

        chosen_idx = torch.where(preferences == 1)[0]
        rejected_idx = torch.where(preferences == 0)[0]
        chosen_logps = logps[chosen_idx, ...]
        reject_logps = logps[rejected_idx, ...]

        return chosen_logps, reject_logps, kl_logps

    def get_reduced_masked_logps(self, logps, labels, average_log_probs=False):
        assert logps.shape == labels.shape, "logps and labels shape mismatch"

        loss_mask = (labels > -1).float()

        if average_log_probs:
            # need to guard against divide by zero in case labels are all -100
            return (logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (logps * loss_mask).sum(-1)

    def loss_func(self, pi_logprobs, ref_logprobs, labels, preferences, average_log_probs=False):

        rewards = self.get_reduced_masked_logps(
            pi_logprobs - ref_logprobs, labels, average_log_probs=average_log_probs
        )
        chosen_rewards, reject_rewards, kl_rewards = self.split_output_tensor(rewards, preferences)

        kl_divergence = average_losses_across_data_parallel_group([kl_rewards.mean().clamp(min=0).detach()])
        if chosen_rewards.shape[0] != 0:
            chosen_losses = 1.0 - torch.nn.functional.sigmoid(
                self.ref_policy_kl_penalty * (chosen_rewards - kl_divergence)
            )
            chosen_rewards = self.ref_policy_kl_penalty * chosen_rewards
        else:
            chosen_losses = torch.Tensor([]).to(rewards.dtype).to(rewards.device)
            chosen_rewards = torch.Tensor([]).to(rewards.dtype).to(rewards.device)

        if reject_rewards.shape[0] != 0:
            reject_losses = 1.0 - torch.nn.functional.sigmoid(
                self.ref_policy_kl_penalty * (kl_divergence - reject_rewards)
            )
            reject_rewards = self.ref_policy_kl_penalty * reject_rewards
        else:
            reject_losses = torch.Tensor([]).to(rewards.dtype).to(rewards.device)
            reject_rewards = torch.Tensor([]).to(rewards.dtype).to(rewards.device)

        loss = torch.cat(
            (self.desirable_loss_weight * chosen_losses, self.undesirable_loss_weight * reject_losses), dim=0
        )

        return loss.mean(), kl_divergence

    def get_loss_and_metrics(self, batch, forward_only):
        seq_length = batch["samples"].shape[1]

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
            kl_div = torch.cat([item["kl"] for item in losses_reduced_per_micro_batch])
            kl_mean = kl_div.mean()

            rewards_all = torch.cat((rewards_chosen, rewards_rejected))
            rewards_chosen_mean = rewards_chosen.mean()
            rewards_rejected_mean = rewards_rejected.mean()
            rewards_all_mean = rewards_all.mean()
            rewards_all_std = rewards_all.std()
            rewards_margins = rewards_chosen_mean.nan_to_num(0) - rewards_rejected_mean.nan_to_num(0)

            # average loss across micro batches
            loss_mean = torch.as_tensor(
                [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
        else:

            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            kl_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            rewards_chosen_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_rejected_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_margins = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(kl_mean, get_last_rank())

        torch.distributed.broadcast(rewards_chosen_mean, get_last_rank())
        torch.distributed.broadcast(rewards_rejected_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())
        torch.distributed.broadcast(rewards_margins, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "kl": kl_mean,
            "rewards_chosen_mean": rewards_chosen_mean,
            "rewards_rejected_mean": rewards_rejected_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
            "rewards_margin": rewards_margins,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

        return loss_mean.item(), metrics

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)

    @torch.no_grad()
    def get_logprob_batch(self, batch):
        seq_length = batch["samples"].shape[1]

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only=True)

        fwd_bwd_function = get_forward_backward_func()

        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(logprobs_only=True),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=self.cfg.kto.log_prob_forward_micro_batch_size,
            collect_non_loss_data=True,
        )

        if len(logprobs_list) > 0:
            sample_logprobs_list = []
            kl_sample_logprobs_list = []
            for item in logprobs_list:
                chosen_logprobs, rejected_logprobs = self.split_output_tensor(item["logprobs"])
                sample_logprobs_list.append(chosen_logprobs)
                kl_sample_logprobs_list.append(rejected_logprobs)

            logprobs = torch.cat([torch.cat(sample_logprobs_list), torch.cat(kl_sample_logprobs_list)], dim=0)
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

        if self.use_peft and self.ref_policy_state_dict is None:
            # when using adapters instead of full-tuning, the actor is reference model + adapters
            with adapter_control(self):
                # With adapters disabled (meaning using the reference model), calculate ref_log_probs
                ref_log_probs = self.get_logprob_batch(batch)
        else:
            with cpu_weight_swap(self, self.ref_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
                ref_log_probs = self.get_logprob_batch(batch)

        # return in GPU, trainer needs to move to cpu
        return ref_log_probs

    # def get_logprob_output_only_func(self, inference_only=True):
    #     fwd_output_only_func = self.get_forward_output_only_func()

    #     def log_prob_output_only_func(dataloader_iter, model):
    #         batch = next(dataloader_iter)
    #         logits, _ = fwd_output_only_func(iter([batch[0:3],]), model)

    #         def id_func(logits, non_loss_data=True):
    #             logprobs = from_parallel_logits_to_logprobs(
    #                 vocab_parallel_logits=logits,
    #                 target=batch[-1].cuda() if len(batch) == 4 else batch[0].cuda(),
    #                 inference_only=inference_only,
    #                 higher_stability=True,
    #             )
    #             return {"logprobs": logprobs}

    #         return logits, id_func

    #     return log_prob_output_only_func

    # @torch.no_grad()
    # def get_logprob_batch(self, global_batch):
    #     set_sync_funcs(self, forward_only=True)

    #     # assumes we pad to seq length before going into the model
    #     # response_tokens = sequences.cuda()
    #     # labels = labels.cuda() if labels is not None else None

    #     dp_size = parallel_state.get_data_parallel_world_size()
    #     local_batch_size, seq_len = global_batch[0].shape
    #     global_batch_size = local_batch_size * dp_size

    #     forward_mbs = self.cfg.kto.log_prob_forward_micro_batch_size
    #     forward_mbs_times_dp = forward_mbs * dp_size

    #     data_iter = get_iterator_k_split(global_batch, global_batch_size // forward_mbs_times_dp)

    #     fwd_bwd_function = get_forward_backward_func()
    #     logprobs_list = fwd_bwd_function(
    #         forward_step_func=self.get_logprob_output_only_func(inference_only=True),
    #         data_iterator=data_iter,
    #         model=self.model,
    #         num_microbatches=global_batch_size // forward_mbs_times_dp,
    #         forward_only=True,
    #         seq_length=seq_len,
    #         micro_batch_size=forward_mbs,
    #         collect_non_loss_data=True,
    #     )

    #     if len(logprobs_list) > 0:
    #         logprobs = torch.cat([item["logprobs"] for item in logprobs_list])
    #     else:
    #         logprobs = None

    #     if parallel_state.get_pipeline_model_parallel_world_size() > 1:
    #         # broadcast it from last PP stage to everything else
    #         logprobs = broadcast_2d_tensor(
    #             logprobs,
    #             parallel_state.get_pipeline_model_parallel_last_rank(),
    #             parallel_state.get_pipeline_model_parallel_group(),
    #         )

    #     return logprobs

    # def get_ref_policy_logprobs(self, list_of_batches):
    #     tokens = torch.cat([torch.cat((b["samples"], b["kl_samples"]), dim=0) for b in list_of_batches], dim=0)
    #     masks = torch.cat(
    #         [torch.cat((b["attention_mask"], b["attention_mask"]), dim=0) for b in list_of_batches], dim=0
    #     )
    #     pos_ids = torch.cat([torch.cat((b["position_ids"], b["position_ids"]), dim=0) for b in list_of_batches], dim=0)
    #     labels = torch.cat(
    #         [torch.cat((b["sample_labels"], b["kl_sample_labels"]), dim=0) for b in list_of_batches], dim=0
    #     )

    #     global_batch = [tokens, masks, pos_ids, labels]
    #     with cpu_weight_swap(self, self.ref_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
    #         ref_log_probs = self.get_logprob_batch(global_batch)

    #     # return in GPU, trainer needs to move to cpu
    #     return ref_log_probs

# def average_rewards_across_data_parallel_group(rewards):
#     """Reduce a tensor of losses across all GPUs."""

#     num_rewards = torch.tensor([(rewards[0].numel())], device=rewards[0].device)
#     averaged_rewards = torch.cat([reward.clone().detach().sum().view(1) for reward in rewards])

#     torch.distributed.all_reduce(averaged_rewards, group=parallel_state.get_data_parallel_group())
#     torch.distributed.all_reduce(num_rewards, group=parallel_state.get_data_parallel_group())

#     averaged_rewards = averaged_rewards / num_rewards.clamp(min=1)

#     return averaged_rewards