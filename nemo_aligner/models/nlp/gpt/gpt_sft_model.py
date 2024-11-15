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
from typing import List, Optional, Tuple, Union

import hydra
import torch
from megatron.core.num_microbatches_calculator import get_micro_batch_size, get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import configure_batch_sizes, make_sharded_tensors_from_reference, offload_distributed_adam


class GPTSFTModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        self.distributed_adam_offload_manager = None
        self.to_offload_adam_states = self.cfg.get("offload_adam_states", False)

        inference_params = dict(cfg.get("inference", {}))
        # note that this will fail if import path is not available when the model is restored
        # this is by design as it might not be possible to use model correctly without a matching
        # inference strategy
        if "strategy" in inference_params:
            if inference_params["strategy"] is not None:
                inference_params["strategy"] = hydra.utils.instantiate(inference_params["strategy"], model=self)
        self.set_inference_params(**inference_params)

    def set_inference_params(self, length_params=None, sampling_params=None, strategy=None):
        # TODO (igitman): the name self._inference_params is very similar to self.inference_params
        #    that's used by the base model for another purpose. There is also self._inference_config
        #    that has a similar role to the parameters below but is less convenient.
        #    While there is a danger for accidental name collision and this adds confusion, it's ok for now
        #    as we are planning to remove dependence on the MegatronGPTModel after which we can remove this note

        # registering inference parameters or default values
        self._inference_params = {
            "length_params": length_params or get_default_length_params(),
            "sampling_params": sampling_params or get_default_sampling_params(),
            "strategy": strategy,
        }

    def get_inference_params(self):
        return self._inference_params

    def get_loss_and_metrics(self, batch, forward_only):
        """Take a data_iter which is an interator over the microbatches
            and return loss as well as metrics
        """
        _, seq_length = batch["tokens"].shape
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()
        fwd_loss_fn = self.get_forward_output_and_loss_func(forward_only)

        losses_reduced = fwd_bwd_function(
            forward_step_func=fwd_loss_fn,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            micro_batch_size=get_micro_batch_size(),
            seq_length=seq_length,
        )

        torch.cuda.synchronize()

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
    
    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        losses = output_tensor.float()
        is_finite = losses.isfinite()
        loss_mask = loss_mask.view(-1).float()
        loss_mask = loss_mask * is_finite.view(-1)
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub.clamp(min=1)  # sequence level nll
        if parallel_state.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss
    
    def sharded_state_dict(self, prefix: str = ""):
        sharded_state_dict_orig = super().sharded_state_dict(prefix=prefix)

        if hasattr(self, "ref_policy_state_dict") and (self.ref_policy_state_dict is not None) and ("self_taught" in self.cfg) and (self.cfg.self_taught.get("update_ref_policy", False)):
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

    def generate(
        self,
        inputs: Union[List[str], Tuple[torch.Tensor, torch.Tensor]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        *,
        strategy: Optional[TextGenerationStrategy] = None,
    ) -> OutputType:
        """
        Same as base model generate, except the following:

        1. Apply padding to max length.
        2. Add a "predictions" key to the output, which is the model output without the prompt.

        These two additional steps above are only performed for actual generation from the model:
        if `generate()` is called with `compute_logprob=True` then the base model method is used.
        """
        if sampling_params is not None and sampling_params.get("compute_logprob", False):
            return super().generate(
                inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy
            )

        if not isinstance(inputs, (list, tuple)):
            raise NotImplementedError(f"Expected type(inputs)=(list or tuple) but got {type(inputs)=}")

        if isinstance(inputs[0], str):
            # add_EOS=False since it is absent from nemo.collections.nlp.modules.common.text_generation_utils.megatron_gpt_generate
            prompt_tokens, prompt_lengths = tokenize_batch(
                sentences=inputs,
                tokenizer=self.tokenizer,
                max_len=self.cfg.encoder_seq_length,
                add_BOS=sampling_params["add_BOS"],
                add_EOS=False,
            )
        else:
            prompt_tokens, prompt_lengths = inputs

        max_prompt_length = prompt_lengths.max().item()
        max_response_length = length_params["max_length"]
        max_length = max_prompt_length + max_response_length
        # # nemo requires us to pad the response length up before we do anything
        prompt_tokens = torch.nn.functional.pad(prompt_tokens, (0, max_length), value=self.tokenizer.eos_id)
        output = super().generate(
            inputs=(prompt_tokens, prompt_lengths),
            length_params=length_params,
            sampling_params=sampling_params,
            strategy=strategy,
        )
        if output is not None:  # may be `None` for intermediate PP ranks when PP>2
            # adding predictions key which contains only model predictions without the prompt
            output["predictions"] = [
                self.tokenizer.ids_to_text(tokens[length.item() :][:max_response_length])
                for tokens, length in zip(output["token_ids"], prompt_lengths)
            ]
        return output

    @torch.no_grad()
    def infer(self, inference_batch, length_params=None, sampling_params=None, strategy=None):
        prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        default_params = self.get_inference_params()

        self.prepare_for_inference()
        outputs = self.generate(
            (prompt_tokens, prompt_lengths),
            length_params=length_params or default_params["length_params"],
            sampling_params=sampling_params or default_params["sampling_params"],
            strategy=strategy or default_params["strategy"],
        )
        self.finish_inference()

        return outputs
    
    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()
    
    def prepare_for_training_step(self):
        """things to call to preprare for training
        """
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        """things to call to finish training for example grad reductions
        """
        grad_reductions(self)

    def prepare_for_validation_step(self):
        """things to call to prepare for validation
        """
        prepare_for_validation_step(self)
        gbs = int(self.cfg.data.validation_ds.global_batch_size)
        mbs = int(self.cfg.data.validation_ds.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

    def finish_validation_step(self):
        """things to call to prepare for validation
        """
        finish_validation_step(self)
        # restore the batch sizes for training
        gbs = int(self.cfg.data.train_ds.global_batch_size)
        mbs = int(self.cfg.data.train_ds.micro_batch_size)
        dp_size = int(parallel_state.get_data_parallel_world_size())
        configure_batch_sizes(mbs=mbs, gbs=gbs, dp=dp_size)

    def prepare_for_inference(self):
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)
        self.offload_adam_states()

    def finish_inference(self):
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()
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
