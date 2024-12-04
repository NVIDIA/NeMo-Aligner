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
from lightning.pytorch.trainer.trainer import Trainer
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import AlignableGenerativeInterface
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp, from_parallel_logits_to_logprobs
from nemo_aligner.utils.text_generation_utils import TrackLengthGPTModelTextGenerationStrategy
from nemo_aligner.utils.train_utils import (
    grad_reductions,
    prepare_for_training_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import (
    adapter_control,
    configure_batch_sizes,
    cpu_weight_swap,
    masked_mean,
    offload_distributed_adam,
)


class MegatronGPTRSModel(NLPAdapterModelMixin, MegatronGPTModel, AlignableGenerativeInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        self.distributed_adam_offload_manager = None

        # length parameters for generation
        self._length_params = OmegaConf.to_container(self.cfg.rs.length_params, resolve=True)
        # sampling parameters for generation
        self._sampling_params = OmegaConf.to_container(self.cfg.rs.sampling_params, resolve=True)

        self.to_offload_adam_states = self.cfg.rs.offload_adam_states
        self.forward_micro_batch_size = self.cfg.rs.forward_micro_batch_size

    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):

            batch = next(data_iterator)
            response_tokens = batch["response_tokens"]
            mask = batch["mask"]

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

            # TODO: This loss depends on the mbs, which is can lead to inconsistencies. See https://github.com/NVIDIA/NeMo/issues/8343.
            def loss_func(parallel_logits):
                mask = batch["mask"]
                tokens = batch["tokens"]

                curr_log_probs = from_parallel_logits_to_logprobs(vocab_parallel_logits=parallel_logits, target=tokens)
                loss = -1 * masked_mean(curr_log_probs, mask)  # Loss is mean logits on response tokens

                reduced_actor_loss = average_losses_across_data_parallel_group([loss])
                return (
                    loss,
                    {"loss": reduced_actor_loss,},
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

        for key in ["loss"]:
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

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self, context_lengths=prompt_lengths, max_length=self._length_params["max_length"]
        )
        actor_output = self.generate(
            inputs=inputs, length_params=self._length_params, sampling_params=self._sampling_params, strategy=strategy
        )

        response_lengths = strategy.get_lengths()
        max_response_length = response_lengths.max().item()

        response_tokens = torch.cuda.LongTensor(actor_output["token_ids"])

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

        rollout_batch = {
            "response_tokens": response_tokens,
            "response_lengths": response_lengths,
            "prompt_lengths": prompt_lengths,
        }

        # return in GPU, trainer needs to move to cpu
        return rollout_batch

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
