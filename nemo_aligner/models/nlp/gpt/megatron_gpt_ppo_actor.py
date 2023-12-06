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

from contextlib import nullcontext

import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
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
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import AlignableGenerativeInterface
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor,
    calculate_distributed_entropy,
    from_parallel_logits_to_logprobs,
)
from nemo_aligner.utils.train_utils import (
    grad_reductions,
    prepare_for_training_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import (
    calculate_dialogue_response_lengths,
    configure_batch_sizes,
    cpu_weight_swap,
    masked_mean,
    offload_distributed_adam,
)


class MegatronGPTActorModel(MegatronGPTModel, AlignableGenerativeInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        self.init_policy_state_dict = None
        self.distributed_adam_offload_manager = None

        # length parameters for generation
        self._length_params = OmegaConf.to_container(self.cfg.ppo.length_params, resolve=True)
        # sampling parameters for generation
        self._sampling_params = OmegaConf.to_container(self.cfg.ppo.sampling_params, resolve=True)

        # Safety check until https://github.com/NVIDIA/NeMo-Aligner/pull/19 is merged.
        valid_end_strings = ["<|endoftext|>", "<extra_id_1>"]
        for end_string in self._sampling_params["end_strings"]:
            if end_string not in valid_end_strings:
                raise NotImplementedError(
                    "Currently only '<|endoftext|>' and '<extra_id_1>' are allowed in `sampling_params.end_strings`, "
                    f"but found '{end_string}'"
                )

        self.to_offload_adam_states = self.cfg.ppo.offload_adam_states
        self.entropy_bonus = self.cfg.ppo.entropy_bonus
        self.ratio_eps = self.cfg.ppo.ratio_eps
        self.forward_micro_batch_size = self.cfg.ppo.forward_micro_batch_size

    # training calls
    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            batch = next(data_iterator)
            response_tokens = batch["response_tokens"]
            advantages = batch["advantages"]
            mask = batch["mask"]
            prev_logprobs = batch["prev_logprobs"]

            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=response_tokens,
                eod_token=self.tokenizer.eos_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )

            batch = {
                "tokens": response_tokens,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "advantages": advantages,
                "prev_log_probs": prev_logprobs,
                "mask": mask,
            }

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("tokens", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("tokens", "advantages", "mask", "prev_log_probs"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            parallel_logits = model(batch["tokens"], batch["position_ids"], batch["attention_mask"], labels=None,)

            def loss_func(parallel_logits):
                mask = batch["mask"]
                advantages = batch["advantages"]
                prev_log_probs = batch["prev_log_probs"]
                tokens = batch["tokens"]

                curr_log_probs = from_parallel_logits_to_logprobs(vocab_parallel_logits=parallel_logits, target=tokens)

                scaled_entropy = torch.tensor(0.0, dtype=parallel_logits.dtype, device=parallel_logits.device)
                if self.entropy_bonus > 0:
                    scaled_entropy = calculate_distributed_entropy(parallel_logits, mask) * self.entropy_bonus

                # Calculate clipped PPO surrogate loss function.
                ratios = (curr_log_probs - prev_log_probs).exp()
                ratios_clamped = ratios.clamp(1.0 - self.ratio_eps, 1.0 + self.ratio_eps)

                loss1 = -advantages * ratios
                loss2 = -advantages * ratios_clamped
                actor_loss = masked_mean(torch.max(loss1, loss2), mask)
                loss = actor_loss - scaled_entropy

                with torch.no_grad():
                    ppo_ratio = masked_mean(ratios.detach(), mask)
                    ppo_ratio_clamped = masked_mean(ratios_clamped.detach(), mask)
                    scaled_entropy = scaled_entropy.detach()

                reduced_actor_loss = average_losses_across_data_parallel_group([loss])
                return (
                    loss,
                    {
                        "loss": reduced_actor_loss,
                        "ppo_ratio": ppo_ratio,
                        "ppo_ratio_clamped": ppo_ratio_clamped,
                        "scaled_entropy": scaled_entropy,
                    },
                )

            return parallel_logits, loss_func

        return fwd_output_and_loss_func

    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def get_loss_and_metrics(self, batch, forward_only):
        sequence_length = batch["response_tokens"].size(-1)

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_actor_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=sequence_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        metrics = {}

        for key in ["loss", "ppo_ratio", "ppo_ratio_clamped", "scaled_entropy"]:
            if losses_reduced_per_micro_batch:
                metric_mean = torch.stack(
                    [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                ).mean()
            else:
                metric_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            torch.distributed.broadcast(metric_mean, get_last_rank())

            metrics[key] = metric_mean.cpu().item()

        return metrics["loss"], metrics

    def finish_training_step(self):
        grad_reductions(self)

    def finish_training(self):
        """no need to offload adam states here
        """

    # inference calls
    def get_logprob_output_only_func(self, inference_only=True):
        fwd_output_only_func = self.get_forward_output_only_func()

        def log_prob_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            output_tensor, _ = fwd_output_only_func(iter([batch,]), model)

            def id_func(output_tensor, non_loss_data=True):
                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor, target=batch[0], inference_only=inference_only
                )
                return logprobs

            return output_tensor, id_func

        return log_prob_output_only_func

    @torch.no_grad()
    def get_inference_log_probs(self, response_tokens, forward_micro_batch_size):
        set_sync_funcs(self, forward_only=True)

        mbs, seq_length = response_tokens.size()
        num_microbatches = divide(mbs, forward_micro_batch_size)
        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(response_tokens)

        batch_iter = get_iterator_k_split([response_tokens, attention_mask, position_ids], num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(inference_only=True),
            data_iterator=batch_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=forward_micro_batch_size,
            collect_non_loss_data=True,
        )

        logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            # broadcast it from last PP stage to everything else
            logprobs = broadcast_2d_tensor(
                logprobs,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                parallel_state.get_pipeline_model_parallel_group(),
            )

        return logprobs

    def prepare_for_inference(self):
        """normally we would configure the micro batch calculator here
            but the nemo generation already does the configuration"""
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)
        self.offload_adam_states()

    @torch.no_grad()
    def infer(self, inference_batch):
        prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        inputs = (prompt_tokens, prompt_lengths)

        actor_output = self.generate(
            inputs=inputs, length_params=self._length_params, sampling_params=self._sampling_params
        )

        response_tokens = torch.cuda.LongTensor(actor_output["token_ids"])
        response_lengths = calculate_dialogue_response_lengths(
            tokens=response_tokens,
            prompt_lengths=prompt_lengths,
            tokenizer=self.tokenizer,
            end_strings=self._sampling_params["end_strings"],
            max_generation_length=self._length_params["max_length"],
            max_sequence_length=self.cfg.encoder_seq_length,
        )

        # TODO(geshen): get nemo generate to return the unaltered log probs
        log_probs = self.get_inference_log_probs(
            response_tokens, forward_micro_batch_size=self.forward_micro_batch_size
        )

        rollout_batch = {
            "response_tokens": response_tokens,
            "response_lengths": response_lengths,
            "prompt_lengths": prompt_lengths,
            "logprobs": log_probs,
        }

        # return in GPU, trainer needs to move to cpu
        return rollout_batch

    def get_init_policy_logprobs(self, rollout_batches):
        init_log_probs = []
        with cpu_weight_swap(self, self.init_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2):
            for rollout_batch in rollout_batches:
                init_log_prob = self.get_inference_log_probs(
                    rollout_batch["response_tokens"].cuda(), forward_micro_batch_size=self.forward_micro_batch_size
                )
                init_log_probs.append(init_log_prob)

        # return in GPU, trainer needs to move to cpu
        return init_log_probs

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

    def get_ltor_masks_and_position_ids(self, tokens):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.tokenizer.eos_id,
            reset_position_ids=self.cfg.data.get("reset_position_ids", False),
            reset_attention_mask=self.cfg.data.get("reset_attention_mask", False),
            eod_mask_loss=False,  # since we ignore the loss mask here
        )
        attention_mask = attention_mask.expand(tokens.size(0), -1, -1, -1)

        return attention_mask, loss_mask, position_ids
