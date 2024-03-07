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
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
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

    @torch.no_grad()
    def gather_and_split_rewards(self, pi_logprobs, ref_logprobs, masks):
        pi_logprobs = pi_logprobs.detach()

        dp_group = parallel_state.get_data_parallel_group()

        batch_logs = self.get_reduced_masked_logps(pi_logprobs - ref_logprobs, masks[:, 1:])

        output_list = [torch.zeros_like(batch_logs) for _ in range(dp_group.size())]

        torch.distributed.all_gather(output_list, batch_logs, group=dp_group)

        split_iter = map(self.split_output_tensor, output_list)

        out_chosen, out_rejected = map(torch.cat, zip(*split_iter))

        return out_chosen.flatten(), out_rejected.flatten()

    def get_forward_output_and_loss_func(self, validation_step=False):
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
                    required_keys.update(("actual", "generated", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(
                        (
                            "ref_policy_log_probs_actual",
                            "ref_policy_log_probs_generated",
                            "actual_mask",
                            "generated_mask",
                        )
                    )

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            tokens, masks, ref_logprobs = None, None, None
            if batch["actual"] is not None and batch["generated"] is not None:
                tokens = torch.cat((batch["actual"], batch["generated"]), dim=0)

            if batch["actual_mask"] is not None and batch["generated_mask"] is not None:
                masks = torch.cat((batch["actual_mask"], batch["generated_mask"]), dim=0)

            if (
                batch["ref_policy_log_probs_actual"] is not None
                and batch["ref_policy_log_probs_generated"] is not None
            ):
                ref_logprobs = torch.cat(
                    (batch["ref_policy_log_probs_actual"], batch["ref_policy_log_probs_generated"]), dim=0
                )

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

            def loss_func(output_tensor):
                if validation_step and not self.cfg.data.validation_ds.get("drop_last", True):
                    raise NotImplementedError(
                        "SPIN does not support validation when `cfg.data.validation_ds.drop_last=False`"
                    )

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor, target=tokens, higher_stability=True
                )

                loss, acc_chosen = self.loss_func(per_token_logps, ref_logprobs, masks[:, 1:])

                reduced_loss = average_losses_across_data_parallel_group([loss])
                reduced_acc = average_losses_across_data_parallel_group([acc_chosen])

                out_actual, out_generated = self.gather_and_split_rewards(per_token_logps, ref_logprobs, masks)

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "acc": reduced_acc,
                        "out_actual": out_actual,
                        "out_generated": out_generated,
                    },
                )

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

    def loss_func(self, pi_logprobs, ref_logprobs, masks, average_log_probs=False):
        rewards = self.get_reduced_masked_logps(pi_logprobs - ref_logprobs, masks, average_log_probs=average_log_probs)
        chosen_rewards, reject_rewards = self.split_output_tensor(self.ref_policy_kl_penalty * rewards)

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards)

        with torch.no_grad():
            comp = chosen_rewards > reject_rewards
            acc_chosen = comp.float().mean()

        return loss, acc_chosen

    def get_loss_and_metrics(self, batch, forward_only):
        seq_length = batch["actual"].shape[1]

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
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
            rewards_chosen = torch.cat([item["out_actual"] for item in losses_reduced_per_micro_batch])
            rewards_rejected = torch.cat([item["out_generated"] for item in losses_reduced_per_micro_batch])

            rewards_all = torch.cat((rewards_chosen, rewards_rejected))
            rewards_chosen_mean = rewards_chosen.mean()
            rewards_rejected_mean = rewards_rejected.mean()
            rewards_all_mean = rewards_all.mean()
            rewards_all_std = rewards_all.std()

            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
            acc_tensors_list = [loss_reduced["acc"] for loss_reduced in losses_reduced_per_micro_batch]

            if len(acc_tensors_list) == 1:
                acc_tensor = acc_tensors_list[0]
            elif len(acc_tensors_list) > 1:
                acc_tensor = torch.concat(acc_tensors_list)
            acc_mean = acc_tensor.mean()
        else:

            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            acc_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            rewards_chosen_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_rejected_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(acc_mean, get_last_rank())

        torch.distributed.broadcast(rewards_chosen_mean, get_last_rank())
        torch.distributed.broadcast(rewards_rejected_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "acc": acc_mean,
            "rewards_actual_mean": rewards_chosen_mean,
            "rewards_generated_mean": rewards_rejected_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

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
        self.loss_func = super().loss_func

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
                    module.load_state_dict(checkpoint_state_dict, strict=True)
            else:
                # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
                # see NLPModel.on_load_checkpoint
                checkpoint["state_dict"] = {}

        # legacy checkpointing no longer supported (sorry)
        else:
            raise RuntimeError("legacy checkpoints are not supported by NeMo-Aligner")

    def get_logprob_output_only_func(self, inference_only=True):
        fwd_output_only_func = self.get_forward_output_only_func()

        def log_prob_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            logits, _ = fwd_output_only_func(iter([batch[0:3],]), model)

            def id_func(logits, non_loss_data=True):
                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=logits,
                    target=batch[-1].cuda() if len(batch) == 4 else batch[0].cuda(),
                    inference_only=inference_only,
                    higher_stability=True,
                )
                return {"logprobs": logprobs}

            return logits, id_func

        return log_prob_output_only_func

    @torch.no_grad()
    def get_logprob_batch(self, global_batch):
        set_sync_funcs(self, forward_only=True)

        # assumes we pad to seq length before going into the model
        # response_tokens = sequences.cuda()
        # labels = labels.cuda() if labels is not None else None

        dp_size = parallel_state.get_data_parallel_world_size()
        local_batch_size, seq_len = global_batch[0].shape
        global_batch_size = local_batch_size * dp_size

        forward_mbs = self.cfg.spin.log_prob_forward_micro_batch_size
        forward_mbs_times_dp = forward_mbs * dp_size

        data_iter = get_iterator_k_split(global_batch, global_batch_size // forward_mbs_times_dp)

        fwd_bwd_function = get_forward_backward_func()
        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(inference_only=True),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=global_batch_size // forward_mbs_times_dp,
            forward_only=True,
            seq_length=seq_len,
            micro_batch_size=forward_mbs,
            collect_non_loss_data=True,
        )

        if len(logprobs_list) > 0:
            logprobs = torch.cat([item["logprobs"] for item in logprobs_list])
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
        tokens = torch.cat((batch["actual"], batch["generated"]), dim=0)
        masks = torch.cat((batch["attention_mask"], batch["attention_mask"]), dim=0)
        pos_ids = torch.cat((batch["position_ids"], batch["position_ids"]), dim=0)
        global_batch = [tokens, masks, pos_ids]
        with cpu_weight_swap(self, self.ref_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
            ref_log_probs = self.get_logprob_batch(global_batch)

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
