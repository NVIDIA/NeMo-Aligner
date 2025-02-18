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

"""utils for training with NeMo Megatron* Models
mostly adapted from https://github.com/NVIDIA/NeMo/blob/8c061debd05837148e86fac19abf024e7210c35d/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L190
"""
from functools import partial

from megatron.core.distributed import finalize_model_grads
from megatron.core.num_microbatches_calculator import get_current_global_batch_size
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

from nemo.collections.nlp.modules.common.megatron.clip_grads import (
    clip_grad_norm_distributed_optimizer,
    clip_grad_norm_fp32,
)
from nemo.collections.nlp.modules.common.megatron.module import Float16Module


# TODO: Delete this once API introduced in NeMo (https://github.com/NVIDIA/NeMo/pull/10803)
def set_sync_funcs(ptl_model, forward_only):
    # handle asynchronous grad reduction
    no_sync_func = None
    grad_sync_func = None
    param_sync_func = None
    if ptl_model.with_distributed_adam:
        if forward_only:
            if ptl_model.validation_param_sync_overlap:
                param_sync_func = ptl_model.sync_overlap_parameters
        elif not ptl_model.use_mcore_dist_optim:
            no_sync_func = partial(ptl_model._optimizer.no_sync, greedy_grad_copy=ptl_model.megatron_amp_O2,)
            grad_sync_func = ptl_model.reduce_overlap_gradients
            param_sync_func = ptl_model.sync_overlap_parameters
        else:
            if ptl_model.cfg.optim.get("overlap_grad_sync", False):
                no_sync_func = [model_chunk.no_sync for model_chunk in ptl_model.model]
                no_sync_func = no_sync_func[0] if len(ptl_model.model) == 1 else no_sync_func

                if ptl_model.cfg.optim.get("align_grad_reduce", True):
                    grad_sync_func = [model_chunk.start_grad_sync for model_chunk in ptl_model.model]
                    grad_sync_func = grad_sync_func[0] if len(ptl_model.model) == 1 else grad_sync_func
            if ptl_model.cfg.optim.get("overlap_param_sync", False) and ptl_model.cfg.optim.get(
                "align_param_gather", False
            ):
                param_sync_func = [model_chunk.start_param_sync for model_chunk in ptl_model.model]
                param_sync_func = param_sync_func[0] if len(ptl_model.model) == 1 else param_sync_func

    # pipeline schedules will get these from self.model.config
    for module in ptl_model.get_model_module_list():
        module.config.no_sync_func = no_sync_func
        module.config.grad_sync_func = grad_sync_func
        module.config.param_sync_func = param_sync_func
        if ptl_model.use_mcore_dist_optim:
            module.config.finalize_model_grads_func = finalize_model_grads


def prepare_for_training_step(ptl_model, zero_grad=True):
    set_train(ptl_model)
    # Initialize userbuffer communicators.
    if ptl_model.initialize_ub:
        ptl_model.initialize_ub_func()

    if ptl_model.rampup_batch_size:
        current_global_batch_size = get_current_global_batch_size()
        # do validation and save the checkpoint when gbs is changed
        if ptl_model.prev_global_batch_size != current_global_batch_size and ptl_model.prev_global_batch_size:
            ptl_model.trainer.should_stop = True

    if ptl_model.use_mcore_dist_optim:
        for model_chunk in ptl_model.model:
            model_chunk.zero_grad_buffer()

    if zero_grad:
        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        ptl_model._optimizer.zero_grad()

    if ptl_model.with_distributed_adam:
        # hack to enable overlapping param sync and forward compute
        # note: the distributed optimizer monkey-patches each
        # parameter's __getattribute__ function so that it can
        # launch parameter all-gathers the first time the
        # parameter is accessed after the optimizer step. However,
        # PyTorch directly passes embedding parameters into a C++,
        # bypassing this process. A quick-and-dirty hack is to
        # manually interact with the parameter.
        modules = ptl_model.model if isinstance(ptl_model.model, list) else [ptl_model.model]
        for module in modules:
            if isinstance(module, (Float16Module, MCoreFloat16Module)):
                module = module.module
            if not ptl_model.mcore_gpt:
                module = module.language_model
            if hasattr(module, "embedding"):
                for param in module.embedding.parameters():
                    param.data_ptr()


# TODO: Delete this once API introduced in NeMo (https://github.com/NVIDIA/NeMo/pull/10803)
# TODO: Update PR to move this logic into staticmethod in nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py
def grad_reductions(ptl_model):
    # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
    if ptl_model.cfg.get("tensor_model_parallel_size", 1) > 1 and ptl_model.cfg.get("sequence_parallel", False):
        # Mcore DistOpt handles this, so we don't have to
        if not ptl_model.use_mcore_dist_optim:
            ptl_model.megatron_timer_start("allreduce_sequence_parallel_gradients", log_level=1)
            ptl_model.allreduce_sequence_parallel_gradients()
            ptl_model.megatron_timer_stop("allreduce_sequence_parallel_gradients")

    ptl_model.megatron_timer_start("gradient_allreduce", log_level=1)
    if ptl_model.use_fsdp:
        # Reduce the gradients omitted from FSDP-sharding
        ptl_model.allreduce_fsdp_sharding_omitted_gradients()
    elif ptl_model.with_distributed_adam:
        if not ptl_model.use_mcore_dist_optim:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            ptl_model._optimizer._finish_bucket_grad_sync()
        # else: Mcore distributed optim calls finalize_model_grads to finish grad sync
    elif ptl_model.megatron_amp_O2:
        # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
        if (
            ptl_model.cfg.get("pipeline_model_parallel_size", 1) > 1
            or ptl_model.cfg.get("sequence_parallel", False)
            or not ptl_model.cfg.get("async_grad_allreduce", True)
        ):
            # main grads are stored in the MainParamsOptimizer wrapper
            ptl_model._optimizer.allreduce_main_grads()
    else:
        # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
        # so we all-reduce gradients after the pipeline
        ptl_model.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)
    ptl_model.megatron_timer_stop("gradient_allreduce")

    if (
        not ptl_model.use_mcore_dist_optim
        and ptl_model.cfg.get("pipeline_model_parallel_size", 1) > 1
        and ptl_model.cfg.get("share_embeddings_and_output_weights", True)
    ):
        ptl_model.megatron_timer_start("allreduce_first_last_embeddings", log_level=1)
        # when using pipeline parallelism the first and last stage must keep embeddings in sync
        ptl_model.allreduce_first_last_embeddings()
        ptl_model.megatron_timer_stop("allreduce_first_last_embeddings")


def prepare_for_validation_step(ptl_model):
    if ptl_model.initialize_ub:
        ptl_model.initialize_ub_func()

    set_eval(ptl_model)


def finish_validation_step(ptl_model):
    set_train(ptl_model)


def set_train(ptl_model):
    if isinstance(ptl_model.model, list):
        for model_module in ptl_model.model:
            model_module.train()
    else:
        ptl_model.train()


def set_eval(ptl_model):
    if isinstance(ptl_model.model, list):
        for model_module in ptl_model.model:
            model_module.eval()
    else:
        ptl_model.eval()


# TODO: adapt the version in /opt/NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py
def clip_gradients(ptl_model, clip_val):
    """PTL hook to configure gradients.
    We use gradient clipping implementation from megatron-lm.
    """
    if clip_val is None:
        return

    clip_val = float(clip_val)
    if clip_val <= 0:
        return

    if ptl_model.with_megatron_fused_adam or ptl_model.use_mcore_dist_optim:
        # Gradient clipping is done in optimizer step
        return

    if ptl_model.grad_clip_pl_default:
        # use the default behavior
        return super().configure_gradient_clipping(*args, **kwargs)

    if ptl_model.with_distributed_adam:
        grad_norm = clip_grad_norm_distributed_optimizer(ptl_model._optimizer, clip_val)
    else:
        if ptl_model.megatron_amp_O2:
            # grep fp32 master parameters for gradient clipping
            parameters = ptl_model._optimizer.get_parameters_with_grad()
        else:
            parameters = ptl_model.get_parameters_with_grad()
        grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val, use_fsdp=ptl_model.use_fsdp,)
    return grad_norm
