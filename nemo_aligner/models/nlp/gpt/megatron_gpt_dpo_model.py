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
from typing import Any, List, Optional

import torch
from lightning.pytorch.trainer.trainer import Trainer
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group, ## TODO: move to llm collection. This is used in nemo/lightning as well
    get_iterator_k_split, ## TODO: move to llm collection. This is used in nemo/lightning as well
)
#from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin ## peft callback in nemo 2
from nemo.collections.nlp.parts.utils_funcs import get_last_rank ## TODO: copy this fn to aligner (just == torch.distributed.get_world_size() - 1)
from nemo_aligner.data.nlp.config import DPODataConfig
from nemo_aligner.data.nlp.datasets import DPOPackedDataset
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor, from_parallel_logits_to_logprobs
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_sync_funcs,
)
from nemo_aligner.utils.utils import adapter_control, cpu_weight_swap

from nemo_aligner.models.nlp.gpt.nemo2.megatron_gpt_dpo_head import MegatronDPOHead

@dataclass
class DPOConfig:
    # Defaults to model's micro_batch_size
    # This default value ensures there are no numeric differences beween trained and reference policies when computing log probs.
    # A higher value can be used to speed-up log probs computations, but may cause numeric differences.
    log_prob_forward_micro_batch_size: Optional[int] = None
    ref_policy_kl_penalty: float = 0.2
    preference_average_log_probs: bool = False # whether normalizing log probs according to the sequence length in preference_loss
    sft_average_log_probs: Optional[bool] = None # whether normalizing log probs according to the sequence length in sft_loss. If not specified, defaults to preference_average_log_probs
    gt_reward_scale: float = 1. # the scale of the rewards in RPO
    preference_loss: str = "dpo" # the preference loss, we support dpo, ipo, rpo_sq, rpo_bwd_kl, rpo_fwd_kl
    preference_loss_weight: float = 1 # the coefficient of the preference loss
    sft_loss_weight: float = 0 # the coefficient of the SFT loss

## TODO: add peft support
class MegatronGPTDPOModel(GPTModel, SupervisedInterface):
    """
    Megatron GPT DPO Model Training.
    """

    ## optimizer needs to be connected to the model at some point
    ## if not provided during init, needs to be done later via model.optim = optim
    def __init__(
        self,
        gpt_config: GPTConfig,
        dpo_config: DPOConfig,
        data_config: DPODataConfig, ## TODO: this is only needed for mbs
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None, ## TODO: is this needed for us?
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            gpt_config,
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

        ## TODO: put this check elsewhere?
        if self.config.pipeline_model_parallel_size > 1 and not self.config.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )
        
        ## setting default values. Update this following nemo run integration
        if not dpo_config.log_prob_forward_micro_batch_size:
            dpo_config.log_prob_forward_micro_batch_size = data_config.micro_batch_size ## TODO: get this from data config
        if dpo_config.sft_average_log_probs is None:
            dpo_config.sft_average_log_probs = dpo_config.preference_average_log_probs
        
        self.dpo_config = dpo_config
        self.data_config = data_config

        self.head = MegatronDPOHead(
            dpo_config.ref_policy_kl_penalty,
            dpo_config.preference_average_log_probs,
            dpo_config.sft_average_log_probs,
            dpo_config.preference_loss_weight,
            dpo_config.sft_loss_weight,
            dpo_config.preference_loss,
            dpo_config.gt_reward_scale,
        )

    ## TODO: remove this once we switch to Marc's APIs
    def build_model(
        self,
        wrap_with_ddp: bool = True,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        #model_type, ## for simplicity, assume decoder only
        on_cpu: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> List[torch.nn.Module]:
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            parallel_state.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
            for i in range(virtual_pipeline_model_parallel_size):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                model.append(
                    self.configure_model(
                        self.tokenizer,
                        pre_process=parallel_state.is_pipeline_first_stage(),
                        post_process=parallel_state.is_pipeline_last_stage(),
                    )
                )
        else:
            model = self.configure_model(
                self.tokenizer,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            )
        if not isinstance(model, list):
            model = [model]

        for model_module in model:
            model_module.model_type = ModelType.encoder_or_decoder

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model:
            for param in model_module.parameters():
                set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        # Print number of parameters.
        if parallel_state.model_parallel_is_initialized() and parallel_state.get_data_parallel_rank() == 0:
            msg = " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_pipeline_model_parallel_rank(),
                _calc_number_of_params(model),
            )
            logging.info(msg)

        # GPU allocation.
        if not on_cpu:
            for model_module in model:
                model_module.cuda(torch.cuda.current_device())

        if wrap_with_ddp:
            i = torch.cuda.current_device()
            model = [
                torch.nn.parallel.distributed.DistributedDataParallel(
                    model_module,
                    device_ids=[i],
                    output_device=i,
                    process_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
                for model_module in model
            ]
        return model
   
    def get_forward_output_and_loss_func(self, validation_step=False, logprobs_only=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            packed = "input_ids" in batch

            forward_args, loss_args = self.head.data_step(batch)

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
                    vocab_parallel_logits=output_tensor,
                    target=loss_args["labels"],
                    inference_only=True,
                    higher_stability=True,
                    ignore_last=not packed,
                )
                return {"logprobs": logprobs}

            def loss_func(output_tensor):
                ## TODO: handle this
                """if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    raise NotImplementedError("DPO does not support validation when cfg.data.drop_last=False")"""

                return self.head.loss_step(output_tensor, **loss_args)

            if logprobs_only:
                return output_tensor, logprobs_func
            else:
                return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_loss_and_metrics(self, batch, forward_only):
        packed = "input_ids" in batch
        if packed:
            seq_length = batch["input_ids"].shape[1]
        else:
            seq_length = batch["chosen"].shape[1]

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        micro_batch_size = self.data_config.micro_batch_size
        if not packed:
            # each minibatch has 2 comparisons so tensor shape will be mbs * 2
            micro_batch_size *= 2
        else:
            assert micro_batch_size == 1, (
                f"Packed sequence is only supported with micro batch size 1,"
                f" but your micro batch size is {micro_batch_size}."
            )

            ## TODO: we need a new check here since we pass the layer_spec into GPTConfig directly
            """assert self.config.get(
                "transformer_engine", False
            ), "Transformer Engine should be enabled when using sequence packing.""""

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only, logprobs_only=False),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        return self.head.loss_reduce(losses_reduced_per_micro_batch)

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    ## I think this is no longer needed since mcore finalize_model_grads handles it
    def finish_training_step(self):
        pass
        #grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)

    @torch.no_grad()
    def get_logprob_batch(self, batch):
        packed = "input_ids" in batch
        if packed:
            k = "input_ids"
        else:
            k = "chosen"
        seq_length = batch[k].shape[1]
        batch_size = batch[k].shape[0]

        num_microbatches = divide(batch_size, self.dpo_config.log_prob_forward_micro_batch_size)
        micro_batch_size = self.dpo_config.log_prob_forward_micro_batch_size
        if not packed:
            # each minibatch has 2 comparisons so tensor shape will be mbs * 2
            micro_batch_size *= 2
        else:
            assert micro_batch_size == 1, (
                f"Packed sequence is only supported with forward micro batch size 1,"
                f" but your forward micro batch size is {micro_batch_size}."
            )

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
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
        )

        if len(logprobs_list) > 0:
            if not packed:
                chosen_logprobs_list = []
                rejected_logprobs_list = []
                for item in logprobs_list:
                    chosen_logprobs, rejected_logprobs = self.split_output_tensor(item["logprobs"])
                    chosen_logprobs_list.append(chosen_logprobs)
                    rejected_logprobs_list.append(rejected_logprobs)

                logprobs = torch.cat([torch.cat(chosen_logprobs_list), torch.cat(rejected_logprobs_list)], dim=0)
            else:
                logprobs_list = [item["logprobs"] for item in logprobs_list]
                logprobs = torch.cat(logprobs_list, dim=0)
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
