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

from typing import List, Optional, Tuple, Union

import hydra
import numpy as np
import secrets
import torch
from megatron.core.utils import divide
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
from nemo_aligner.utils.text_generation_utils import tokenize_batch, TrackLengthGPTModelTextGenerationStrategy, verify_is_valid_and_clamp_range_
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_pp
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import configure_batch_sizes, batch_pad_to_fixed_len, clear_memory
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM
from nemo.utils import AppState, logging
from nemo.collections.common.tokenizers import AutoTokenizer


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    world_size = torch.distributed.get_world_size()
    all_ranks = np.arange(world_size)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    if AppState().use_tp_pp_dp_mapping:
        # [DP, PP, TP]
        all_ranks = all_ranks.reshape(-1, pp_size, tp_size)
        return all_ranks[dp_rank, :, :].min()
    else:
        # [PP, DP, TP]
        all_ranks = all_ranks.reshape(pp_size, -1, tp_size)
        return all_ranks[:, dp_rank, :].min()


class GPTGenRMModel(NLPAdapterModelMixin, MegatronGPTModel, SupervisedInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        inference_params = dict(cfg.get("inference", {}))
        # note that this will fail if import path is not available when the model is restored
        # this is by design as it might not be possible to use model correctly without a matching
        # inference strategy
        if "strategy" in inference_params:
            if inference_params["strategy"] is not None:
                inference_params["strategy"] = hydra.utils.instantiate(inference_params["strategy"], model=self)
        self.set_inference_params(**inference_params)
        
        self.forward_micro_batch_size = self.cfg.get("forward_micro_batch_size", self.cfg.micro_batch_size)
        self.trtllm_generate = None

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
    '''
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
    '''
    
    @torch.no_grad()
    def get_generations_trt(self, inputs, sampling_params, length_params, prepare_for_inference=False):
        if prepare_for_inference:
            self.prepare_for_inference()
            self.trtllm_generate.refit(self)
            clear_memory()

        if isinstance(inputs, list):
            list_of_tensors = [torch.LongTensor(self.tokenizer.text_to_ids(x)) for x in inputs]
            prompt_lengths = torch.LongTensor([len(x) for x in list_of_tensors])

            batch_max_length = prompt_lengths.max().item()
            max_possible_length = min(self.cfg.encoder_seq_length, batch_max_length + length_params["max_length"])
            prompt_tokens = batch_pad_to_fixed_len(list_of_tensors, max_possible_length, pad_token=self.tokenizer.eos_id)
        elif isinstance(inputs, tuple):
            prompt_tokens, prompt_lengths = inputs
        else:
            raise RuntimeError(f"Wrong type received by get_generations: {type(inputs)}")

        prompt_tokens = prompt_tokens.cuda(non_blocking=True)
        prompt_lengths = prompt_lengths.cuda(non_blocking=True)
        inputs = (prompt_tokens, prompt_lengths)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self, context_lengths=prompt_lengths, max_length=length_params["max_length"]
        )

        generations = self.trtllm_generate.generate(inputs)
        response_tokens = generations["response_tokens"]
        response_lengths = generations["response_lengths"]

        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.tokenizer, sampling_params["end_strings"]
        )

        if prepare_for_inference:
            self.finish_inference()
            self.trtllm_generate.free()

        return response_tokens.cpu(), prompt_lengths.cpu(), response_lengths.cpu(), is_valid.cpu()
    
    def infer(
        self,
        inputs: Union[List[str], Tuple[torch.Tensor, torch.Tensor]],
        add_BOS: bool = False,
        add_EOS: bool = False,
    ):
        MAX_GEN_LEN = 4096
        USE_TRT = True
        
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        elif isinstance(inputs, list):
            assert all(isinstance(item, str) for item in inputs), "list must contain all strings in infer function"
            #context_tokens_tensor, context_length_tensor = tokenize_batch(
            #    inputs, self.tokenizer, self.cfg.encoder_seq_length, add_BOS=add_BOS, add_EOS=add_EOS,
            #)
            list_of_tensors = [torch.LongTensor(self.tokenizer.text_to_ids(x)) for x in inputs]
            context_length_tensor = torch.LongTensor([len(x) for x in list_of_tensors])

            batch_max_length = context_length_tensor.max().item()
            max_possible_length = min(self.cfg.encoder_seq_length, batch_max_length + MAX_GEN_LEN)
            context_tokens_tensor = batch_pad_to_fixed_len(list_of_tensors, max_possible_length, pad_token=self.tokenizer.eos_id)
        else:
            raise NotImplementedError(f"{type(inputs)=} is not supported in infer function")
        
        if USE_TRT and self.trtllm_generate is None:
            print("******* TRYING TO INSTANTIATE TRT *******: ", self.cfg.encoder_seq_length, " BS: ", len(context_length_tensor), flush=True)
            '''
            model_parallel_group = parallel_state.get_model_parallel_group()
            src = get_model_parallel_src_rank()
            if torch.distributed.get_rank() == src:
                random_seed = secrets.randbits(32)
                input_info_tensor = torch.cuda.FloatTensor([random_seed])
                torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
            else:
                input_info_tensor = torch.empty(1, dtype=torch.float32, device=torch.cuda.current_device())
                torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
                random_seed = int(input_info_tensor[0].item())
            '''
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.cfg,
                end_strings=["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
                tokenizer=self.tokenizer,
                sample_temperature=1.0,
                sample_top_k=0,
                sample_top_p=1.0,
                repetition_penalty=1.0,
                max_generation_length=MAX_GEN_LEN,
                max_input_len=self.cfg.encoder_seq_length - MAX_GEN_LEN,
                generation_batch_size=4,
                use_greedy=False,
                trt_model_type="llama" if isinstance(self.tokenizer, AutoTokenizer) else "gptnext",
                seed=5831445,
                unload_engine_train=False,
                reshard_model=False,
            )
            
            self.prepare_for_inference()
            self.trtllm_generate.refit(self)
            clear_memory()
        else:
            self.prepare_for_inference()

        context_tokens_tensor = context_tokens_tensor.cuda()
        context_length_tensor = context_length_tensor.cuda()

        inference_batch_size, sequence_length = context_tokens_tensor.size()
        inputs = [context_tokens_tensor, context_length_tensor]

        # if inference batch size is smaller than forward mbs run it at the lower batch size
        forward_micro_batch_size = min(inference_batch_size, self.forward_micro_batch_size)

        num_microbatches = divide(inference_batch_size, forward_micro_batch_size)
        data_iter = get_iterator_k_split(inputs, num_microbatches)

        rewards = self.forward_step_for_rewards(data_iter, forward_micro_batch_size, sequence_length, num_microbatches)

        if parallel_state.is_pipeline_last_stage():
            rewards = torch.cat(rewards)

        rewards = broadcast_2d_tensor_within_pp(rewards)
        
        self.finish_inference()
        
        return rewards

    def forward_step_for_rewards(self, data_iter, micro_batch_size, sequence_length, num_microbatches):
        set_sync_funcs(self, forward_only=True)

        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            #forward_step_func=self.get_forward_output_only_func_no_TRT(),
            forward_step_func=self.get_forward_output_only_funcTRT_K2(),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            # Prevent automatic scaling by the number of micro-batches, as we are not in training mode.
            collect_non_loss_data=True,
        )
        return output_tensor

    def get_forward_output_only_func_no_TRT(self):
        def fwd_output_only_func(batch, model):
            tokens, length = next(batch)
            tokens = tokens.cuda()
            length = length.cuda()
            #default_params = self.get_inference_params()
            #YES_LOC = [9642, 7566] # self.tokenizer.text_to_ids(['Yes', ' Yes'])
            MAX_GEN_LEN = 4096
            batch_max_length = length.max().item()
            #max_possible_length = min(self.cfg.encoder_seq_length, batch_max_length + MAX_GEN_LEN)
            adj_generation_length = min(MAX_GEN_LEN, self.cfg.encoder_seq_length - batch_max_length)
            
            #output_tensor = model(tokens, length, position_ids, attention_mask)
            length_params = {'min_length': 1, 'max_length': adj_generation_length}
            sampling_params = {
                  'use_greedy': True,
                  'temperature': 1.0,
                  'top_k': 0,
                  'top_p': 1.0,
                  'repetition_penalty': 1.0,
                  'add_BOS': False,
            	  'add_EOS': False,
                  'all_probs': False,
                  'compute_logprob': False,
                  'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            
            strategy = TrackLengthGPTModelTextGenerationStrategy(
            	model=self, context_lengths=length, max_length=adj_generation_length,
            )
            
            generations = self.generate(
                (tokens, length),
                length_params=length_params, #default_params["length_params"],
                sampling_params=sampling_params, #default_params["sampling_params"],
                strategy=strategy, #default_params["strategy"],
            )
            batch_max_length = max([len(x) for x in generations['token_ids']])
            local_batch_len = min(self.cfg.encoder_seq_length, batch_max_length + MAX_GEN_LEN)
            xx = torch.cat([batch_pad_to_fixed_len(torch.LongTensor(x).unsqueeze(0), local_batch_len, pad_token=self.tokenizer.eos_id) for x in generations['token_ids']]).cuda()
            yy = torch.LongTensor([len(x) for x in generations['token_ids']]).cuda()
            adj_generation_length = min(MAX_GEN_LEN, self.cfg.encoder_seq_length - batch_max_length)
            sampling_params = {
                  'use_greedy': False,
                  'temperature': 1.0,
                  'top_k': 0,
                  'top_p': 1.0,
                  'repetition_penalty': 1.0,
                  'add_BOS': False,
            	  'add_EOS': False,
                  'all_probs': False,
                  'compute_logprob': True,
                  'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            strategy = TrackLengthGPTModelTextGenerationStrategy(
            	model=self, context_lengths=yy, max_length=adj_generation_length,
            )
            generations = self.generate(
           		inputs=(xx,yy),
           		length_params=length_params | {"max_length": adj_generation_length},
           		sampling_params=sampling_params,
           		strategy=strategy,
           	)
            full_lp_token_ids = [tkl.argmax(-1) for tkl in generations['full_logprob']]
            yes_locations = [max([i if c in self.tokenizer.text_to_tokens(["Yes", " Yes"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
            no_locations = [max([i if c in self.tokenizer.text_to_tokens(["No", " No"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
            final_locations = [max(yes, no) for yes, no in zip(yes_locations, no_locations)]
            output_tensor = torch.stack([max([full_lp[yes, yes_pos].exp() for yes_pos in self.tokenizer.text_to_ids(['Yes', ' Yes'])]) for full_lp, yes in zip(generations['full_logprob'], final_locations)])

            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def id_func(output_tensor, non_loss_data=True):
                return output_tensor

            return output_tensor, id_func

        return fwd_output_only_func
    
    def get_forward_output_only_funcTRT(self):
        def fwd_output_only_func(batch, model):
            tokens, length = next(batch)
            tokens = tokens.cuda()
            length = length.cuda()
            MAX_GEN_LEN = 4096
            
            length_params = {'min_length': 1, 'max_length': MAX_GEN_LEN}
            sampling_params = {
                'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            
            response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations_trt((tokens, length), sampling_params, length_params, prepare_for_inference=False)
            token_ids = []
            for t, s, e, v in zip(response_tokens, prompt_lengths.tolist(), response_lengths.tolist(), is_valid.tolist()):
                ints_list = t[:e].tolist()
                token_ids.append(ints_list)
            
            batch_max_length = max([len(x) for x in token_ids])
            local_batch_len = min(self.cfg.encoder_seq_length, batch_max_length + MAX_GEN_LEN)
            xx = torch.cat([batch_pad_to_fixed_len(torch.LongTensor(x).unsqueeze(0), local_batch_len, pad_token=self.tokenizer.eos_id) for x in token_ids]).cuda()
            yy = torch.LongTensor([len(x) for x in token_ids]).cuda()
            adj_generation_length = min(MAX_GEN_LEN, self.cfg.encoder_seq_length - batch_max_length)
            sampling_params = {
                  'use_greedy': False,
                  'temperature': 1.0,
                  'top_k': 0,
                  'top_p': 1.0,
                  'repetition_penalty': 1.0,
                  'add_BOS': False,
            	  'add_EOS': False,
                  'all_probs': False,
                  'compute_logprob': True,
                  'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            strategy = TrackLengthGPTModelTextGenerationStrategy(
            	model=self, context_lengths=yy, max_length=adj_generation_length,
            )
            generations = self.generate(
           		inputs=(xx,yy),
           		length_params=length_params | {"max_length": adj_generation_length},
           		sampling_params=sampling_params,
           		strategy=strategy,
           	)
            full_lp_token_ids = [tkl.argmax(-1) for tkl in generations['full_logprob']]
            yes_locations = [max([i if c in self.tokenizer.text_to_tokens(["Yes", " Yes"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
            no_locations = [max([i if c in self.tokenizer.text_to_tokens(["No", " No"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
            final_locations = [max(yes, no) for yes, no in zip(yes_locations, no_locations)]
            output_tensor = torch.stack([max([full_lp[yes, yes_pos].exp() for yes_pos in self.tokenizer.text_to_ids(['Yes', ' Yes'])]) for full_lp, yes in zip(generations['full_logprob'], final_locations)])

            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def id_func(output_tensor, non_loss_data=True):
                return output_tensor

            return output_tensor, id_func

        return fwd_output_only_func
    
    def get_forward_output_only_funcTRT_K1(self):
        def fwd_output_only_func(batch, model):
            tokens, length = next(batch)
            tokens = tokens.cuda()
            length = length.cuda()
            MAX_GEN_LEN = 4096
            K = 8
            
            length_params = {'min_length': 1, 'max_length': MAX_GEN_LEN}
            sampling_params = {
                'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            
            resK = []
            for idx in range(K):
                response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations_trt((tokens, length), sampling_params, length_params, prepare_for_inference=False)
                token_ids = []
                for t, s, e, v in zip(response_tokens, prompt_lengths.tolist(), response_lengths.tolist(), is_valid.tolist()):
                    ints_list = t[:e].tolist()
                    token_ids.append(ints_list)
                
                batch_max_length = max([len(x) for x in token_ids])
                local_batch_len = min(self.cfg.encoder_seq_length, batch_max_length + MAX_GEN_LEN)
                xx = torch.cat([batch_pad_to_fixed_len(torch.LongTensor(x).unsqueeze(0), local_batch_len, pad_token=self.tokenizer.eos_id) for x in token_ids]).cuda()
                yy = torch.LongTensor([len(x) for x in token_ids]).cuda()
                adj_generation_length = min(MAX_GEN_LEN, self.cfg.encoder_seq_length - batch_max_length)
                sampling_params = {
                      'use_greedy': False,
                      'temperature': 1.0,
                      'top_k': 0,
                      'top_p': 1.0,
                      'repetition_penalty': 1.0,
                      'add_BOS': False,
                	  'add_EOS': False,
                      'all_probs': False,
                      'compute_logprob': True,
                      'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
                }
                strategy = TrackLengthGPTModelTextGenerationStrategy(
                	model=self, context_lengths=yy, max_length=adj_generation_length,
                )
                generations = self.generate(
               		inputs=(xx,yy),
               		length_params=length_params | {"max_length": adj_generation_length},
               		sampling_params=sampling_params,
               		strategy=strategy,
               	)
                full_lp_token_ids = [tkl.argmax(-1) for tkl in generations['full_logprob']]
                yes_locations = [max([i if c in self.tokenizer.text_to_tokens(["Yes", " Yes"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
                no_locations = [max([i if c in self.tokenizer.text_to_tokens(["No", " No"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list.tolist()))]) for token_list in full_lp_token_ids]
                final_locations = [max(yes, no) for yes, no in zip(yes_locations, no_locations)]
                output_t = torch.stack([max([full_lp[yes, yes_pos].exp() for yes_pos in self.tokenizer.text_to_ids(['Yes', ' Yes'])]) for full_lp, yes in zip(generations['full_logprob'], final_locations)])
                
                resK.append(output_t)
            
            print("**** RES_K: ", resK, flush=True)
            output_tensor = torch.stack(resK).mean(dim=0)

            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def id_func(output_tensor, non_loss_data=True):
                return output_tensor

            return output_tensor, id_func

        return fwd_output_only_func
    
    def get_forward_output_only_funcTRT_K2(self):
        def fwd_output_only_func(batch, model):
            tokens, length = next(batch)
            tokens = tokens.cuda()
            length = length.cuda()
            MAX_GEN_LEN = 4096
            K = 8
            
            length_params = {'min_length': 1, 'max_length': MAX_GEN_LEN}
            sampling_params = {
                'end_strings': ["<|endoftext|>", "<extra_id_1>", "<|eot_id|>"],
            }
            
            resK = []
            for idx in range(K):
                response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations_trt((tokens, length), sampling_params, length_params, prepare_for_inference=False)
                token_ids = []
                for t, s, e, v in zip(response_tokens, prompt_lengths.tolist(), response_lengths.tolist(), is_valid.tolist()):
                    ints_list = t[:e].tolist()
                    token_ids.append(ints_list)
                
                yes_locations = [max([i if c in self.tokenizer.text_to_tokens(["Yes", " Yes"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list))]) for token_list in token_ids]
                no_locations = [max([i if c in self.tokenizer.text_to_tokens(["No", " No"]) else 0 for i,c in enumerate(self.tokenizer.ids_to_tokens(token_list))]) for token_list in token_ids]
                final_locations = [1 if yes > no else 0 for yes, no in zip(yes_locations, no_locations)]
                output_t = torch.FloatTensor(final_locations).unsqueeze(-1).cuda()
                
                resK.append(output_t)
            
            #print("**** RES_K: ", resK, flush=True)
            output_tensor = torch.stack(resK).mean(dim=0)

            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def id_func(output_tensor, non_loss_data=True):
                return output_tensor

            return output_tensor, id_func

        return fwd_output_only_func

    def prepare_for_inference(self):
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)

    def finish_inference(self):
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()
        set_train(self)
