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


class MegatronGPTRPOModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    """
    Megatron GPT RPO Model Training.
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

        self.preference_avg_log_probs = self.cfg.rpo.get("preference_average_log_probs", False)
        self.sft_avg_log_probs = self.cfg.rpo.get("sft_average_log_probs", self.preference_avg_log_probs)

        self.preference_loss_weight = float(self.cfg.rpo.get("preference_loss_weight", 1))
        self.sft_loss_weight = float(self.cfg.rpo.get("sft_loss_weight", 0))
        assert (
            self.preference_loss_weight != 0 or self.sft_loss_weight != 0
        ), "sft loss weight and dpo loss weight cannot both be 0"

        # variants of preference losses, by default RPO.
        self.preference_loss = self.cfg.rpo.get("preference_loss", "rpo")
        self.gt_reward_scale = float(self.cfg.rpo.get("gt_reward_scale", 1.0))

        self.beta = float(self.cfg.rpo.get("beta", 0.01))
        self.eta = float(self.cfg.rpo.get("eta", 0.01))
        self.k_len = int(self.cfg.rpo.get("num_responses", 2))

    @torch.no_grad()
    def gather_and_split_rewards(self, pi_logprobs, ref_logprobs, labels, average_log_probs=False):
        pi_logprobs = pi_logprobs.detach()

        dp_group = parallel_state.get_data_parallel_group()

        batch_logs = self.get_reduced_masked_logps(
            pi_logprobs - ref_logprobs, labels[:, 1:], average_log_probs=average_log_probs
        )

        output_list = [torch.zeros_like(batch_logs) for _ in range(dp_group.size())]

        torch.distributed.all_gather(output_list, batch_logs, group=dp_group)

        split_iter = map(self.split_output_tensor, output_list)
        outputs = map(torch.cat, zip(*split_iter))
        flat_outputs = list(map(torch.flatten, outputs))

        return flat_outputs

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
                    required_keys.update(["response_" + str(i) for i in range(1, self.k_len + 1)])
                    required_keys.update(("position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(["response_" + str(i) for i in range(1, self.k_len + 1)])
                    required_keys.update(["ref_policy_log_probs_response_" + str(i) for i in range(1, self.k_len + 1)])
                    required_keys.update(["labels_" + str(i) for i in range(1, self.k_len + 1)])
                    required_keys.update(["rewards_" + str(i) for i in range(1, self.k_len + 1)])

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # creating tokens and labels tensor batches
            tokens, labels, ref_logprobs, gt_rewards = None, None, None, None
            if batch["response_1"] is not None:
                tokens = torch.cat(tuple(batch['response_' + str(i+1)] for i in range(self.k_len)), dim=0)

            if batch["labels_1"] is not None:
                labels = torch.cat(tuple(batch['labels_' + str(i+1)] for i in range(self.k_len)), dim=0)

            if batch["rewards_1"] is not None:
                gt_rewards = torch.cat(tuple(batch['rewards_' + str(i+1)] for i in range(self.k_len)), dim=0)

            if batch.get("ref_policy_log_probs_response_1") is not None:
                ref_logprobs = torch.cat(tuple(batch['ref_policy_log_probs_response_' + str(i+1)] for i in range(self.k_len)), dim=0)

            # this is necessary if MBS > 1 with the new GBS padding logic, as you may get batch dim > 1 in some configs
            # these two lines ensure your position_ids and attn_mask are always B=1
            attention_mask = batch['attention_mask'][0:1]

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
                    raise NotImplementedError("RPO does not support validation when cfg.data.drop_last=False")

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=labels,
                    inference_only=validation_step,
                    higher_stability=True,
                )

                preference_loss, acc_best_resp = self.loss_func(
                    per_token_logps,
                    ref_logprobs,
                    labels[:, 1:],
                    gt_rewards,
                    average_log_probs=self.preference_avg_log_probs,
                )

                sft_loss = torch.zeros_like(preference_loss)
                if self.sft_loss_weight != 0:
                    sft_loss = self.sft_loss_func(
                        per_token_logps, labels[:, 1:], gt_rewards, average_log_probs=self.sft_avg_log_probs
                    )
                loss = self.preference_loss_weight * preference_loss + self.sft_loss_weight * sft_loss


                (
                    reduced_loss,
                    reduced_preference_loss,
                    reduced_sft_loss,
                    reduced_acc,
                ) = average_losses_across_data_parallel_group([loss, preference_loss, sft_loss, acc_best_resp])

                out_responses = self.gather_and_split_rewards(
                    per_token_logps, ref_logprobs, labels, average_log_probs=self.preference_avg_log_probs
                )

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "avg_sft_loss": reduced_sft_loss,
                        "avg_preference_loss": reduced_preference_loss,
                        "acc": reduced_acc,
                        "out_responses": out_responses,
                    },
                )

            if logprobs_only:
                return output_tensor, logprobs_func
            else:
                return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor):
        responses_logps = torch.split(output_tensor.float(), len(output_tensor) // self.k_len, dim=0)
        return responses_logps

    def get_reduced_masked_logps(self, logps, labels, average_log_probs=False):
        assert logps.shape == labels.shape, "logps and labels shape mismatch"

        loss_mask = (labels > -1).float()

        if average_log_probs:
            # need to guard against divide by zero in case labels are all -100
            return (logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (logps * loss_mask).sum(-1)

    def log_sum_exp(self, x):
        max_x = torch.max(x)
        return max_x + torch.log(torch.sum(torch.exp(x - max_x)))

    def loss_func(self, pi_logprobs, ref_logprobs, labels, gt_rewards, average_log_probs=False):
        if self.preference_loss == 'rpo':
            # estimated rewards
            rewards_pred = torch.stack(self.split_output_tensor(self.get_reduced_masked_logps(
                self.beta * (pi_logprobs - ref_logprobs), labels, average_log_probs=average_log_probs
            )))

            # based on GT rewards
            gt_rewards = torch.stack(self.split_output_tensor(gt_rewards))
            p_star = self.eta * gt_rewards
        else:
            raise ValueError("Unknown RPO Loss")

        loss = ( torch.nn.functional.softmax(p_star, dim=0) * (torch.nn.functional.log_softmax( p_star, dim=0 ) - torch.nn.functional.log_softmax( rewards_pred, dim=0 )) ).sum(0).mean(0)
        
        # adding accuracy for the best rewards -> MSE or best accuracy?
        acc_best_resp = (torch.argmax(rewards_pred, dim=0) == torch.argmax(gt_rewards, dim=0)).float().mean()

        return loss, acc_best_resp

    def sft_loss_func(self, pi_logprobs, labels, gt_rewards, average_log_probs=False):
        logprobs = self.get_reduced_masked_logps(pi_logprobs, labels, average_log_probs=average_log_probs) # [16]
        all_log_probs = torch.stack(self.split_output_tensor(logprobs)) # [4, 4] -> each has several responses which we select the best?
        gt_rewards = torch.stack(self.split_output_tensor(gt_rewards)) # same, we split the rewards
        chosen_best = torch.argmax(gt_rewards, dim=0)

        chosen_logprobs = all_log_probs[chosen_best, torch.arange(all_log_probs.size(1))]
        return -chosen_logprobs.mean(0)

    def get_loss_and_metrics(self, batch, forward_only):
        seq_length = batch["response_1"].shape[1]

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
            micro_batch_size=self.cfg.micro_batch_size * self.k_len,  # each minibatch has K comparisons so tensor shape will be mbs * num_responses
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            collected_rewards_per_resp = []
            for i in range(self.k_len):
                collected_rewards_per_resp.append(
                    torch.cat([item["out_responses"][i] for item in losses_reduced_per_micro_batch])
                )

            rewards_all = torch.cat(tuple(collected_rewards_per_resp))
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

            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(sft_loss_mean, get_last_rank())
        torch.distributed.broadcast(preference_loss_mean, get_last_rank())
        torch.distributed.broadcast(acc_mean, get_last_rank())

        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "sft_loss": sft_loss_mean,
            "preference_loss": preference_loss_mean,
            "acc": acc_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
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
        seq_length = batch["response_1"].shape[1]
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
            micro_batch_size=self.cfg.rpo.log_prob_forward_micro_batch_size,
            collect_non_loss_data=True,
        )

        each_response_list = [ [] for _ in range(self.k_len) ]

        if len(logprobs_list) > 0:
            for item in logprobs_list:
                all_log_probs = self.split_output_tensor(item["logprobs"])
                for ind in range(self.k_len):
                    each_response_list[ind].extend(all_log_probs[ind])
            each_response_list = [ torch.stack(b, dim=0) for b in each_response_list ]
            logprobs = torch.cat(each_response_list, dim=0)
            
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
