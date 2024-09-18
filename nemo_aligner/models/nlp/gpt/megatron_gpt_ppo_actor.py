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
import torch.distributed
from megatron.core.num_microbatches_calculator import get_num_microbatches
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
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging
from nemo_aligner.models.alignable_interface import AlignableGenerativeInterface
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor_within_pp,
    calculate_distributed_entropy,
    from_parallel_logits_to_logprobs,
)
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import (
    grad_reductions,
    prepare_for_training_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM
from nemo_aligner.utils.utils import (
    adapter_control,
    clear_memory,
    configure_batch_sizes,
    cpu_weight_swap,
    masked_mean,
    offload_distributed_adam,
)


class MegatronGPTActorModel(NLPAdapterModelMixin, MegatronGPTModel, AlignableGenerativeInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        self.init_policy_state_dict = None
        self.distributed_adam_offload_manager = None

        # length parameters for generation
        self._length_params = OmegaConf.to_container(self.cfg.ppo.length_params, resolve=True)
        # sampling parameters for generation
        self._sampling_params = OmegaConf.to_container(self.cfg.ppo.sampling_params, resolve=True)

        self.to_offload_adam_states = self.cfg.ppo.offload_adam_states and self.with_distributed_adam
        self.entropy_bonus = self.cfg.ppo.entropy_bonus
        self.ratio_eps = self.cfg.ppo.ratio_eps
        self.forward_micro_batch_size = self.cfg.ppo.forward_micro_batch_size

        self.use_trtllm_generation = "trt_llm" in self.cfg.ppo and self.cfg.ppo.trt_llm.enable
        if self.use_trtllm_generation:
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.cfg,
                max_generation_length=self.cfg.ppo.length_params.get("max_length", 1024),
                max_input_len=self.cfg.ppo.trt_llm.get("max_input_len", 1024),
                max_input_tokens=self.cfg.ppo.trt_llm.get("max_input_tokens", 4096),
                generation_batch_size=self.cfg.ppo.get("rollout_micro_batch_size", 4),
                unload_engine_train=self.cfg.ppo.trt_llm.get("unload_engine_train", False),
                trt_model_type=self.cfg.ppo.trt_llm.get("model_type", "llama"),
                end_strings=self.cfg.ppo.sampling_params["end_strings"],
                reshard_model=self.cfg.ppo.trt_llm.get("reshard", False),
                sample_temperature=self.cfg.ppo.sampling_params["temperature"],
                sample_top_k=self.cfg.ppo.sampling_params["top_k"],
                sample_top_p=self.cfg.ppo.sampling_params["top_p"],
                repetition_penalty=self.cfg.ppo.sampling_params["repetition_penalty"],
                use_greedy=self.cfg.ppo.sampling_params.get("use_greedy", False),
                tokenizer=self.tokenizer,
                seed=self.cfg.ppo.trt_llm.get("seed", self.cfg.seed),
            )

    # training calls
    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            batch = next(data_iterator)
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("response_tokens", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("response_tokens", "advantages", "mask", "prev_logprobs", "is_end"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            parallel_logits = model(
                batch["response_tokens"], batch["position_ids"], batch["attention_mask"], labels=None,
            )

            def loss_func(parallel_logits):
                mask = batch["mask"]
                advantages = batch["advantages"]
                prev_log_probs = batch["prev_logprobs"]
                tokens = batch["response_tokens"]
                is_end = batch["is_end"]

                is_end_mask = mask * is_end.view(-1, 1)

                curr_log_probs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=parallel_logits, target=tokens, higher_stability=True
                )

                scaled_entropy = torch.tensor(0.0, dtype=parallel_logits.dtype, device=parallel_logits.device)
                if self.entropy_bonus > 0:
                    scaled_entropy = calculate_distributed_entropy(parallel_logits, is_end_mask) * self.entropy_bonus

                # Calculate clipped PPO surrogate loss function.
                ratios = (curr_log_probs - prev_log_probs).exp()
                ratios_clamped = ratios.clamp(1.0 - self.ratio_eps, 1.0 + self.ratio_eps)

                loss1 = -advantages * ratios
                loss2 = -advantages * ratios_clamped

                if is_end_mask.sum() > 0:
                    actor_loss = masked_mean(torch.max(loss1, loss2), is_end_mask)
                    loss = actor_loss - scaled_entropy
                else:
                    # hack to disable this update since there are no valid tokens
                    loss = loss1.view(-1)[0] * 0

                with torch.no_grad():
                    ppo_ratio = masked_mean(ratios.detach(), mask)
                    ppo_ratio_clamped = masked_mean(ratios_clamped.detach(), mask)
                    scaled_entropy = scaled_entropy.detach()

                (
                    reduced_actor_loss,
                    ppo_ratio,
                    ppo_ratio_clamped,
                    scaled_entropy,
                ) = average_losses_across_data_parallel_group([loss, ppo_ratio, ppo_ratio_clamped, scaled_entropy])
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
        sequence_length = batch["response_tokens"].size(1)

        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(tokens=batch["response_tokens"])
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = position_ids

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_actor_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(data_iter),
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
                    vocab_parallel_logits=output_tensor,
                    target=batch[0],
                    inference_only=inference_only,
                    higher_stability=True,
                )
                return logprobs

            return output_tensor, id_func

        return log_prob_output_only_func

    @torch.no_grad()
    def get_inference_log_probs(self, response_tokens, forward_micro_batch_size=None):
        if forward_micro_batch_size is None:
            forward_micro_batch_size = self.forward_micro_batch_size

        set_sync_funcs(self, forward_only=True)

        mbs, seq_length = response_tokens.size()
        num_microbatches = divide(mbs, forward_micro_batch_size)
        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(response_tokens)

        batch_iter = get_iterator_k_split([response_tokens, attention_mask, position_ids], num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(inference_only=True),
            data_iterator=self._make_data_iterator_list(batch_iter),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=forward_micro_batch_size,
            collect_non_loss_data=True,
        )

        logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None

        # Broadcast it from last PP stage to everything else.
        logprobs = broadcast_2d_tensor_within_pp(logprobs)

        return logprobs

    def prepare_for_inference(self):
        """normally we would configure the micro batch calculator here
            but the nemo generation already does the configuration"""
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)
        self.offload_adam_states()

        if self.use_trtllm_generation:
            # TODO this might be optimized to avoid calling `refit()` twice in a row after a validation step
            self.trtllm_generate.refit(self.model)
            clear_memory()

    @torch.no_grad()
    def infer(self, inference_batch):
        prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        inputs = (prompt_tokens, prompt_lengths)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self, context_lengths=prompt_lengths, max_length=self._length_params["max_length"]
        )

        if self.use_trtllm_generation:
            actor_output = self.trtllm_generate.generate(inputs)
            response_tokens = actor_output["response_tokens"]
            response_lengths = actor_output["response_lengths"]
        else:
            actor_output = self.generate(
                inputs=inputs,
                length_params=self._length_params,
                sampling_params=self._sampling_params,
                strategy=strategy,
            )
            response_tokens = torch.cuda.LongTensor(actor_output["token_ids"]) if actor_output else None
            response_tokens = broadcast_2d_tensor_within_pp(response_tokens, dtype=torch.long)
            response_lengths = strategy.get_lengths()

            max_response_length = response_lengths.max().item()

            # Sanity check to validate response length.
            if max_response_length != response_tokens.size(1):
                # This may actually happen because NeMo does not always stop generation after `max_length` in batch mode
                # => `response_tokens` may contain up to `max_length + max_context_length` tokens.
                # TODO once NeMo fixes this issue we should be able to always raise an exception when the check above fails,
                # and remove the `if` below.
                if (
                    max_response_length >= response_tokens.size(1)
                    or response_tokens.size(1) != prompt_lengths.max().item() + self._length_params["max_length"]
                ):
                    raise AssertionError(
                        f"max response length ({max_response_length}) does not match the size of "
                        f"`response_tokens` ({response_tokens.size(1)})"
                    )

        # sometimes backends like TRT-LLM will generate invalid tokens
        # so we need to also inplace mutate the response_tokens to be within the tokenizer range
        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.tokenizer, self.cfg.ppo.sampling_params["end_strings"]
        )

        rollout_batch = {
            "response_tokens": response_tokens,
            "response_lengths": response_lengths,
            "prompt_lengths": prompt_lengths,
            "is_end": is_valid,
        }

        # return in GPU, trainer needs to move to cpu

        return rollout_batch

    def get_init_policy_logprobs(self, response_tokens):
        use_peft_init_policy = self.use_peft and self.init_policy_state_dict is None

        context_mgr = (
            adapter_control(self)
            if use_peft_init_policy
            else cpu_weight_swap(self, self.init_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2)
        )

        with context_mgr:
            return self.get_inference_log_probs(response_tokens)

    def finish_inference(self):
        # training will onload the adam states, no need to onload it here
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()

        if self.use_trtllm_generation:
            self.trtllm_generate.free()

        set_train(self)

    def offload_adam_states(self):
        if self.distributed_adam_offload_manager is None:

            self.distributed_adam_offload_manager = (
                offload_distributed_adam(
                    self._optimizer.state_dict(state_dict_format=1, gather_on_root=False), force_clear_memory=True
                )
                if self.to_offload_adam_states
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
        position_ids = position_ids.expand(tokens.size(0), -1)

        return attention_mask, loss_mask, position_ids
