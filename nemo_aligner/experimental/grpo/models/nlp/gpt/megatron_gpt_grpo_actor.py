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
import os
import time
import shutil
import gc

import torch
import torch.distributed
from lightning.pytorch.trainer.trainer import Trainer
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

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
from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor_within_pp,
    from_parallel_logits_to_logprobs,
    allgather_cp_sharded_tensor,
)
from nemo_aligner.experimental.grpo.utils.rl_utils import calculate_kl_penalty_joschu2020
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
from nemo_aligner.utils.utils import (
    adapter_control,
    clear_memory,
    configure_batch_sizes,
    cpu_weight_swap,
    masked_mean,
    offload_distributed_adam,
    offload_distributed_parameters,
)
from nemo_aligner.experimental.grpo.inference.utils.utils import parallel_save_cpu_state_dict

from nemo_aligner.experimental.grpo.inference.registry import get_backend, list_available_backends
from nemo_aligner.experimental.grpo.models.nlp.gpt import conversion_dict as CONVERTER

from tensor_comms.shared_tensors import SharedCPUMemoryTensorDict

class MegatronGPTActorModel(NLPAdapterModelMixin, MegatronGPTModel, AlignableGenerativeInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        self.init_policy_state_dict = None
        self.distributed_adam_offload_manager = None
        self.distributed_parameter_offload_manager = None

        # length parameters for generation
        self._length_params = OmegaConf.to_container(self.cfg.grpo.length_params, resolve=True)
        # sampling parameters for generation
        self._sampling_params = OmegaConf.to_container(self.cfg.grpo.sampling_params, resolve=True)

        self.to_offload_adam_states = self.cfg.grpo.offload_adam_states and self.with_distributed_adam
        self.ratio_eps = self.cfg.grpo.ratio_eps
        self.forward_micro_batch_size = self.cfg.grpo.forward_micro_batch_size

        # Initialize the inference backend
        self.inference_backend = None
        self.prepare_for_inference_warmed_up = False
        # Collect backends from the configuration and check which ones are enabled
        enabled_backends = [
            name for name in cfg.grpo.inference_backend.config.keys() 
            if cfg.grpo.inference_backend.config.get(name, {}).get("enable", False)
        ]
        if len(enabled_backends) > 1:
            raise ValueError(
                f"Multiple inference backends enabled ({', '.join(enabled_backends)}). "
                f"Please enable only one inference backend. Available backends: {', '.join(list_available_backends())}."
            )

        if len(enabled_backends) == 0:
            raise ValueError(
                f"No inference backend is enabled. Please enable one backend in the configuration. "
                f"Available backends: {', '.join(list_available_backends())}."
            )
        
        if self.cfg.grpo.inference_backend.enable:
            self.inference_backend = self._initialize_inference_backend(self.cfg.grpo.inference_backend.get("type"))

    def _initialize_inference_backend(self, backend_type):
        """
        Dynamically initialize the appropriate inference backend based on the backend type.
        """
        if backend_type == "trt_llm":
            from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM

            backend = GPTGenerateTRTLLM(
                model_cfg=self.cfg,
                max_generation_length=self.cfg.grpo.length_params.get("max_length", 1024),
                max_input_len=self.cfg.grpo.inference_backend.get("max_input_len", 1024),
                generation_batch_size=self.cfg.grpo.get("generation_rollout_mbs", 4),
                unload_engine_train=self.cfg.grpo.inference_backend.config.trt_llm.get("unload_engine_train", False),
                trt_model_type=self.cfg.grpo.inference_backend.config.trt_llm.get("model_type", "llama"),
                end_strings=self.cfg.grpo.sampling_params["end_strings"],
                reshard_model=self.cfg.grpo.inference_backend.get("reshard", False),
                sample_temperature=self.cfg.grpo.sampling_params["temperature"],
                sample_top_k=self.cfg.grpo.sampling_params["top_k"],
                sample_top_p=self.cfg.grpo.sampling_params["top_p"],
                repetition_penalty=self.cfg.grpo.sampling_params["repetition_penalty"],
                use_greedy=self.cfg.grpo.sampling_params.get("use_greedy", False),
                tokenizer=self.tokenizer,
                seed=self.cfg.grpo.inference_backend.get("seed", self.cfg.seed),
                trt_model_dir=self.cfg.grpo.get("trt_model_dir", "/tmp/trt_llm_model"),
            )
        elif backend_type == "vllm":
            from nemo_aligner.experimental.grpo.inference.vllm.vllm_client import VLLMClient
            sampling_params = {
                "temperature": self.cfg.grpo.sampling_params["temperature"],
                "top_p": self.cfg.grpo.sampling_params["top_p"],
                "top_k": self.cfg.grpo.sampling_params["top_k"],
                "max_tokens": self.cfg.grpo.length_params.get("max_length", 2048),
                "logprobs": 0,
            }
            backend = VLLMClient(
                self.cfg.grpo.inference_backend.config.vllm,
                use_reshard=self.cfg.grpo.inference_backend.get("reshard", False),
                tokenizer=self.tokenizer,
                checkpoint_path=self.cfg.grpo.share_dir,
                sampling_params=sampling_params,
            )
            self.shared_cpu_state_dict = SharedCPUMemoryTensorDict()
        elif backend_type == "trt_llm_pytorch":
            sampling_params = {
                "temperature": self.cfg.grpo.sampling_params["temperature"],
                "top_p": self.cfg.grpo.sampling_params["top_p"],
                "top_k": self.cfg.grpo.sampling_params["top_k"],
                "max_tokens": self.cfg.grpo.length_params.get("max_length", 2048),
                "logprobs": 0,
            }

            from nemo_aligner.experimental.grpo.inference.trtllm_pytorch.trtllm_pytorch_client import TRTLLMPytorchClient 
            backend = TRTLLMPytorchClient(
                self.cfg.grpo.inference_backend.config.trt_llm_pytorch,
                use_reshard=self.cfg.grpo.inference_backend.get("reshard", False),
                tokenizer=self.tokenizer,
                checkpoint_path=self.cfg.grpo.share_dir,
                sampling_params=sampling_params,
            )
            self.shared_cpu_state_dict = SharedCPUMemoryTensorDict()
        else:
            raise ValueError(f"Unsupported inference backend: {backend_type}")

        return backend

    # training calls
    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            _batch = next(data_iterator)
            required_keys = set()
            required_keys.add("attention_mask")

            if parallel_state.is_pipeline_first_stage():
                required_keys.update(("response_tokens", "position_ids"))

            if parallel_state.is_pipeline_last_stage():
                required_keys.update(("response_tokens", "advantages", "mask", "logprobs", "valid_mask", "init_logprobs", "inference_logprobs", "importance_correction"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}

            if parallel_state.get_context_parallel_world_size() > 1:
                _cp_batch = {
                    "response_tokens" : batch['response_tokens'].clone(),
                    "position_ids" : batch['position_ids'].clone(),
                    "attention_mask" : batch['attention_mask'].clone(),
                }
                _cp_batch = self.get_batch_on_this_context_parallel_rank(_cp_batch)
                parallel_logits = model(
                    _cp_batch["response_tokens"], _cp_batch["position_ids"], _cp_batch["attention_mask"], labels=None,
                )

            else:
                parallel_logits = model(
                    batch["response_tokens"], batch["position_ids"], batch["attention_mask"], labels=None,
                )

            def loss_func(parallel_logits):
                mask = batch["mask"]
                advantages = batch["advantages"]
                prev_log_probs = batch["logprobs"]
                init_log_probs = batch["init_logprobs"]
                importance_correction = batch["importance_correction"]
                tokens = batch["response_tokens"]
                is_end = batch["valid_mask"]

                #is_end_mask = mask * is_end.view(-1, 1)
                is_end_mask = mask

                # gather logits along CP seqlen dim and unpad
                # TODO support CP wise from_parallel_logits_to_logprobs
                if parallel_state.get_context_parallel_world_size() > 1:
                    parallel_logits = allgather_cp_sharded_tensor(parallel_logits, seq_dim=1)
                    #remove the 2*cp padding
                    parallel_logits = parallel_logits[:,:_batch['cp_unpadded_seqlen'][0], :]
                    tokens = tokens[:,:_batch['cp_unpadded_seqlen'][0]]

                curr_log_probs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=parallel_logits, target=tokens, higher_stability=True
                )

                kl = self.cfg.grpo.initial_policy_kl_penalty * calculate_kl_penalty_joschu2020(
                    log_probs_policy=curr_log_probs, log_probs_reference=init_log_probs
                ) * importance_correction
                kl = masked_mean(kl, is_end_mask)

                # Calculate clipped GRPO surrogate loss function.
                ratios = (curr_log_probs - prev_log_probs).exp()
                ratios_clamped = ratios.clamp(1.0 - self.ratio_eps, 1.0 + self.ratio_eps)

                loss1 = -advantages * ratios
                loss2 = -advantages * ratios_clamped

                if is_end_mask.sum() > 0:
                    actor_loss = masked_mean(importance_correction * torch.max(loss1, loss2), is_end_mask)
                    loss = actor_loss + kl
                else:
                    # hack to disable this update since there are no valid tokens
                    loss = loss1.view(-1)[0] * 0

                with torch.no_grad():
                    grpo_ratio = masked_mean(ratios.detach(), mask)
                    grpo_ratio_clamped = masked_mean(ratios_clamped.detach(), mask)
                    #print(loss.shape, grpo_ratio.shape, grpo_ratio_clamped.shape, "loss shapes", flush=True)

                (
                    reduced_actor_loss,
                    grpo_ratio,
                    grpo_ratio_clamped,
                ) = average_losses_across_data_parallel_group([loss, grpo_ratio, grpo_ratio_clamped])
                return (
                    loss,
                    {
                        "loss": reduced_actor_loss,
                        "grpo_ratio": grpo_ratio,
                        "grpo_ratio_clamped": grpo_ratio_clamped,
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
        #seq len must be padded to the nearest cp*2 for context parallelism
        if parallel_state.get_context_parallel_world_size() > 1:
            response_tokens = batch['response_tokens']
            cp_unpadded_seqlen = response_tokens.shape[1]
            batch['response_tokens'] = self.pad_tensor_for_cp(response_tokens)

        num_microbatches = get_num_microbatches()
        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(tokens=batch["response_tokens"])
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = position_ids

        if parallel_state.get_context_parallel_world_size() > 1:
            batch['cp_unpadded_seqlen'] = [cp_unpadded_seqlen] * num_microbatches

        data_iter = get_iterator_k_split(batch, num_microbatches)
        set_sync_funcs(self, forward_only)
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_actor_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(data_iter),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=None, #unused
            micro_batch_size=None, #unused
        )

        metrics = {}

        for key in ["loss", "grpo_ratio", "grpo_ratio_clamped"]:
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

    def pad_tensor_for_cp(self, tensor):
        seqlen = tensor.shape[1]
        cp_times_two = parallel_state.get_context_parallel_world_size() * 2
        padded_seqlen =  ((seqlen + cp_times_two - 1) // cp_times_two) * cp_times_two

        padded_tensor = torch.nn.functional.pad(
            tensor, 
            (0,padded_seqlen - seqlen), 
            value=self.tokenizer.eos_id,
        )
        return padded_tensor

    # inference calls
    def get_logprob_output_only_func(self, inference_only=True):
        fwd_output_only_func = self.get_forward_output_only_func()

        def log_prob_output_only_func(dataloader_iter, model):
            _batch = next(dataloader_iter)

            if parallel_state.get_context_parallel_world_size() > 1:
                _cp_batch = {
                    "response_tokens" : _batch['response_tokens'].clone(),
                    "position_ids" : _batch['position_ids'].clone(),
                    "attention_mask" : _batch['attention_mask'].clone(),
                }
                _cp_batch = self.get_batch_on_this_context_parallel_rank(_cp_batch)
                batch = [_cp_batch['response_tokens'], _cp_batch['attention_mask'], _cp_batch['position_ids']]
            else:
                batch = [_batch['response_tokens'], _batch['attention_mask'], _batch['position_ids']]
            output_tensor, _ = fwd_output_only_func(iter([batch,]), model)

            def id_func(output_tensor, non_loss_data=True):
                # gather logits along CP seqlen dim and unpad
                # TODO support CP wise from_parallel_logits_to_logprobs
                if parallel_state.get_context_parallel_world_size() > 1:
                    logits = allgather_cp_sharded_tensor(output_tensor, seq_dim=1)
                    #remove the 2*cp padding
                    logits = logits[:,:_batch['cp_unpadded_seqlen'][0], :]
                    response_tokens = _batch['response_tokens'][:,:_batch['cp_unpadded_seqlen'][0]]
                else:
                    logits = output_tensor
                    response_tokens = _batch['response_tokens']

                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=logits,
                    target=response_tokens,
                    inference_only=inference_only,
                    higher_stability=True,
                )
                return logprobs

            return output_tensor, id_func

        return log_prob_output_only_func

    @torch.no_grad()
    def get_inference_log_probs(self, response_tokens, forward_micro_batch_size=None):
        # we need onload model parameter to calculate log_probs
        self.maybe_onload_parameters()
        if forward_micro_batch_size is None:
            forward_micro_batch_size = self.forward_micro_batch_size

        set_sync_funcs(self, forward_only=True)

        #seq len must be padded to the nearest cp*2 for context parallelism
        if parallel_state.get_context_parallel_world_size() > 1:
            cp_unpadded_seqlen = response_tokens.shape[1]
            response_tokens = self.pad_tensor_for_cp(response_tokens)

        mbs, seq_length = response_tokens.size()
        num_microbatches = divide(mbs, forward_micro_batch_size)
        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(response_tokens)

        batch = {
            'response_tokens': response_tokens,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

        if parallel_state.get_context_parallel_world_size() > 1:
            batch['cp_unpadded_seqlen'] = [cp_unpadded_seqlen] * num_microbatches

        batch_iter = get_iterator_k_split(batch, num_microbatches)

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
        
        # testing sync and save to cpu ramdisk
        start_time = time.time()
        out_dir = self.cfg.grpo.share_dir #"/dev/shm/checkpoint_hf/"

        if torch.cuda.current_device() == 0:
            if os.path.isdir(self.cfg.hf_model_name_or_configs_dir):
                # If a directory is provided, use it directly to obtain all .json files.
                source_hf_jsons_dir = self.cfg.hf_model_name_or_configs_dir
            else:
                # Otherwise, treat it as a HuggingFace model name and download all .json files from the repo.
                source_hf_jsons_dir = snapshot_download(self.cfg.hf_model_name_or_configs_dir, allow_patterns=["*.json"], ignore_patterns=["*.index.json"])
        # os.chmod(out_dir, 0o777)
        class SafeDict(dict):
            def __missing__(self, key):
                return '{' + key + '}'

        with torch.no_grad():
            checksum = 0
            import re  # For computing global keys from layer numbers
            # Initialize pipeline parallel group information.
            pp_group = parallel_state.get_training_pipeline_model_parallel_group()
            pp_world_size = torch.distributed.get_world_size(pp_group)
            my_pp_rank = parallel_state.get_training_pipeline_model_parallel_rank()
            pp_global_rank_ids = parallel_state.get_all_rank_ids_in_group(pp_group)
            
            # Build a mapping on each PP rank from a computed global key to the raw state dict key.
            # The global key is computed by replacing the local layer number (after "layers.")
            # with its corresponding global layer number (if applicable).
            local_map = {}
            for key in self.state_dict().keys():
                local_layer = CONVERTER.get_local_layer_num(key)
                if local_layer is not None:
                    global_layer = CONVERTER.get_global_layer_num(key, self.cfg)
                    # Replace the first occurrence of the digits after "layers." with the global layer number.
                    global_key = re.sub(r'(?<=layers\.)\d+', str(global_layer), key, count=1)
                else:
                    global_key = key
                local_map[global_key] = key

            # Gather the local maps from all PP ranks (only lightweight key info is gathered).
            all_maps = [None] * pp_world_size
            torch.distributed.all_gather_object(all_maps, local_map, group=pp_group)
            
            # Build the union over global keys and assign an owner (the rank with the smallest PP rank).
            union_global_map = {}
            for pp_rank, omap in enumerate(all_maps):
                for gk, raw_key in omap.items():
                    if gk not in union_global_map or pp_global_rank_ids[pp_rank] < union_global_map[gk][0]:
                        union_global_map[gk] = (pp_global_rank_ids[pp_rank], raw_key)
                    else:
                        print(f"WARNING: {gk} already in union_global_map when gathering keys", flush=True)

            #merged_cpu_param_dict = {}
   
            # Process each parameter (by its unique global key) one at a time.
            for gk in sorted(union_global_map.keys()):
                ptime = time.time()
                owner_pp_global_rank, owner_raw_key = union_global_map[gk]

                # Only the owner PP rank has the parameter locally.
                if torch.distributed.get_rank() == owner_pp_global_rank:
                    param = self.state_dict()[owner_raw_key]
                    
                    # Retrieve layer identification info using the conversion helpers.
                    local_layer = CONVERTER.get_local_layer_num(owner_raw_key)
                    global_layer = CONVERTER.get_global_layer_num(owner_raw_key, self.cfg) if local_layer is not None else None
                    format_dict = SafeDict(l=local_layer, gl=global_layer)
                    
                    # Use the conversion dict to get the appropriate recipe for this parameter.
                    formatted_mapping = {k.format_map(format_dict): rec for k, rec in CONVERTER.mcore_te_to_hf_llama.items()}
                    recipe = formatted_mapping.get(owner_raw_key, None)
                    if recipe is None:
                        print(f"WARNING: {owner_raw_key} has no recipe mapping for conversion", flush=True)
                        hf_mapping = {'None': None}
                    else:
                        # If the parameter is TP-sharded, gather its slices on GPU.
                        if recipe.get("tp", None) is not None:
                            tp_group = parallel_state.get_training_tensor_model_parallel_group()
                            tp_world_size = torch.distributed.get_world_size(tp_group)
                            gathered_slices = [torch.empty_like(param) for _ in range(tp_world_size)]
                            torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                            full_param = torch.cat(gathered_slices, dim=recipe["tp"]).to(torch.bfloat16)
                        else:
                            full_param = torch.clone(param).to(torch.bfloat16)
                        
                        # Convert the parameter using the provided function or mapping.
                        if recipe.get("hf_func", None) is not None:
                            hf_mapping = recipe["hf_func"](full_param, self.cfg)
                            hf_mapping = {k.format_map(format_dict): v for k, v in hf_mapping.items()}
                        elif recipe.get("hf", None) is not None:
                            hf_mapping = {recipe["hf"].format_map(format_dict): full_param}
                        else:
                            raise NotImplementedError(f"No conversion recipe found for {owner_raw_key}")
                else:
                    hf_mapping = None  # Non-owner ranks will receive the converted tensors.
                
                # Broadcast the list of target HF parameter keys from the owner.
                if torch.distributed.get_rank() == owner_pp_global_rank:
                    target_keys = [list(hf_mapping.keys())]
                else:
                    target_keys = [None]  # Placeholder to be filled by broadcast.
                
                torch.distributed.broadcast_object_list(target_keys, src=owner_pp_global_rank, group=pp_group)
                if 'None' in target_keys[0]:
                    continue
                
                # For each converted tensor (could be more than one per original parameter), broadcast it individually.
                for target_key in target_keys[0]:
                    if torch.distributed.get_rank() == owner_pp_global_rank:
                        tensor_to_send = hf_mapping[target_key]
                    else:
                        tensor_to_send = None
                    # Broadcast tensor metadata (shape and dtype) to allocate GPU buffer on receiving ranks.
                    meta = [None]
                    if torch.distributed.get_rank() == owner_pp_global_rank:
                        meta[0] = (tensor_to_send.shape, str(tensor_to_send.dtype))
                    torch.distributed.broadcast_object_list(meta, src=owner_pp_global_rank, group=pp_group)
                    shape, dtype_str = meta[0]
                    dtype = getattr(torch, dtype_str.split('.')[-1])
                    if torch.distributed.get_rank() != owner_pp_global_rank:
                        tensor_to_send = torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())
                    torch.distributed.broadcast(tensor_to_send, src=owner_pp_global_rank, group=pp_group)
                    # use shared cpu memory
                    if torch.cuda.current_device() == 0:
                        self.shared_cpu_state_dict[target_key] = tensor_to_send
                        checksum += tensor_to_send.sum().item()
                    del tensor_to_send
                
                # Cleanup on the owner side.
                if torch.distributed.get_rank() == owner_pp_global_rank:
                    if 'full_param' in locals():
                        del full_param
                    if 'param' in locals():
                        del param
                    if 'hf_mapping' in locals():
                        del hf_mapping
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"Time taken to convert {gk} {time.time() - ptime}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            print("Finished parameter-by-parameter gathering over PP with conversion mapping.", flush=True)
            print(f"Checksum: {checksum}", flush=True)
        
            # Copy HF jsons to CPU ramdisk with proper permissions and save the gathered parameters.
            if torch.cuda.current_device() == 0:
                try:
                    os.makedirs(out_dir, exist_ok=True, mode=0o777)
                    for file in os.listdir(source_hf_jsons_dir):
                        if file.endswith('.json') and not file.endswith('.index.json'):
                            src = os.path.join(source_hf_jsons_dir, file)
                            dst = os.path.join(out_dir, file)
                            shutil.copy(src, dst)
                            os.chmod(dst, 0o666)
                except Exception as e:
                    print(f"Error copying HF json files: {e}", flush=True)
                    raise
        
                try:
                    ptime = time.time()
                    if not self.prepare_for_inference_warmed_up:
                        #save_file(self.shared_cpu_state_dict.as_dict(), os.path.join(out_dir, f"params.safetensors"))
                        self.prepare_for_inference_warmed_up = True
                        print(f"Saved to {out_dir} {list(os.listdir(out_dir))} {time.time() - ptime}", flush=True)
                    torch.distributed.broadcast_object_list([self.shared_cpu_state_dict.get_metadata_dict()], src=torch.distributed.get_rank(), group=parallel_state.get_node_group())
                except Exception as e:
                    print(f"Error saving params.safetensors: {e}", flush=True)
                    if os.path.exists(out_dir):
                        try:
                            os.remove(out_dir)
                        except Exception:
                            pass
                    raise
            else:
                recv_metadata = [None]
                torch.distributed.broadcast_object_list(
                    recv_metadata,
                    src=torch.distributed.get_rank() - torch.distributed.get_rank(group=parallel_state.get_node_group()),
                    group=parallel_state.get_node_group()
                )
                self.shared_cpu_state_dict = SharedCPUMemoryTensorDict(
                    communicable_metadata=recv_metadata[0]
                )

        torch.distributed.barrier()
        print(f"MP group: {parallel_state.get_model_parallel_group()}", flush=True)
        torch.distributed.barrier()
        print(f'TIME TO SAVE {time.time() - start_time}', flush=True)

        if self.inference_backend:
            # Refitting or recompiling the inference model for generation
            print(f"Memory free before clear before refit {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB", flush=True)
            print(f"reserved before clear before refit {torch.cuda.memory_reserved() / 1024**3:.2f} GB", flush=True)
            print(f"allocated before clear before refit {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
            clear_memory()
            print(f"Memory free after clear before refit {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB", flush=True)  
            print(f"reserved after clear before refit {torch.cuda.memory_reserved() / 1024**3:.2f} GB", flush=True)
            print(f"allocated after clear before refit {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
            if self.cfg.grpo.inference_backend.type == "vllm" or self.cfg.grpo.inference_backend.type == "trt_llm_pytorch":
                self.inference_backend.refit(self.shared_cpu_state_dict)
            else:
                self.inference_backend.refit(self.model)
            clear_memory()
            print(f"Memory free after clear after refit {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB", flush=True)

    @torch.no_grad()
    def infer(self, inference_batch, use_greedy=False):
        """
        Perform text generation for a batch using the selected inference backend.
        """
        if not self.inference_backend:
            raise ValueError("Inference backend is not initialized or enabled!")

        prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        inputs = (prompt_tokens, prompt_lengths)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self, context_lengths=prompt_lengths, max_length=self._length_params["max_length"]
        )

        if self.inference_backend:
            self.maybe_offload_parameters("Temporarily offload model weights for the shared inference backend to save memory")
            actor_output = self.inference_backend.generate(inputs, use_greedy=use_greedy)
            response_tokens = actor_output["response_tokens"]
            response_lengths = actor_output["response_lengths"]
            response_trt_lps = actor_output["response_logprobs_trt"]
            print(f"response_logprobs_trt: {response_trt_lps.shape}", flush=True)
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
            response_tokens, response_lengths, strategy, self.tokenizer, self.cfg.grpo.sampling_params["end_strings"]
        )

        rollout_batch = {
            "response_tokens": response_tokens,
            "response_lengths": response_lengths,
            "response_trt_lps": response_trt_lps,
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

        if self.inference_backend:
            self.inference_backend.free()


        set_train(self)

    def maybe_offload_parameters(self, msg: str):
        if self.distributed_parameter_offload_manager is None:
            if msg:
                print(msg)

            self.distributed_parameter_offload_manager = (
                offload_distributed_parameters(
                    self.model
                )
            )

            # offload onto cpu
            self.distributed_parameter_offload_manager.__enter__()

    def maybe_onload_parameters(self):
        if self.distributed_parameter_offload_manager is not None:
            # load back onto GPU
            self.distributed_parameter_offload_manager.__exit__(None, None, None)

        self.distributed_parameter_offload_manager = None

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
