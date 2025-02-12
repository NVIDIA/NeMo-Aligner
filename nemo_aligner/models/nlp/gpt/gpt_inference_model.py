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
import pickle
import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_micro_batch_size, get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.tokenizers import AutoTokenizer
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
from nemo.utils import AppState, logging
from nemo_aligner.models.alignable_interface import Inferrable
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.utils import batch_pad_to_fixed_len, clear_memory
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM


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


class GPTInferenceModel(Inferrable, MegatronGPTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        
        logging.info("********** INIT FOR GPTInferenceModel ********")

        inference_params = dict(cfg.get("inference", {}))
        # note that this will fail if import path is not available when the model is restored
        # this is by design as it might not be possible to use model correctly without a matching
        # inference strategy
        if "strategy" in inference_params:
            if inference_params["strategy"] is not None:
                inference_params["strategy"] = hydra.utils.instantiate(inference_params["strategy"], model=self)
        self.set_inference_params(**inference_params)
        self.trtllm_generate = None
        self.HAVE_TRTLLM = None

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

    @torch.no_grad()
    def get_generations(self, inputs, sampling_params, length_params, prepare_for_inference=False):
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
    
    def tokenise_batch_for_generate(self, inputs, length_params):
        list_of_tensors = [torch.LongTensor(self.tokenizer.text_to_ids(x)) for x in inputs]
        prompt_lengths = torch.LongTensor([len(x) for x in list_of_tensors])

        batch_max_length = prompt_lengths.max().item()
        max_possible_length = min(self.cfg.encoder_seq_length, batch_max_length + length_params["max_length"])
        prompt_tokens = batch_pad_to_fixed_len(list_of_tensors, max_possible_length, pad_token=self.tokenizer.eos_id)
        
        return prompt_tokens, prompt_lengths
    
    def send_generate_info(self, context_tokens_tensor, context_length_tensor, length_params, sampling_params, random_seed=None):
        """
        Needs to be synced up with receive_generate_info
        """
        model_parallel_group = parallel_state.get_model_parallel_group()
        src = get_model_parallel_src_rank()
        if random_seed is None:
            random_seed = -1  # to be able to convert to float
        # Send the sizes of the tensors
        input_info = [
            context_tokens_tensor.size(0),  # batch_size
            context_tokens_tensor.size(1),  # seq_len
            random_seed,
        ]
        input_info_tensor = torch.cuda.FloatTensor(input_info)
        torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
    
        # Send variables to all ranks
        torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
        torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)
    
        # send length_params
        length_params_tensor = torch.as_tensor(
            np.frombuffer(pickle.dumps(length_params), dtype=np.int8), device=torch.cuda.current_device()
        )
        length_params_size = torch.as_tensor([length_params_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
        torch.distributed.broadcast(length_params_size, src, model_parallel_group)
        torch.distributed.broadcast(length_params_tensor, src, model_parallel_group)
        
        # send sampling_params
        sampling_params_tensor = torch.as_tensor(
            np.frombuffer(pickle.dumps(sampling_params), dtype=np.int8), device=torch.cuda.current_device()
        )
        sampling_params_size = torch.as_tensor([sampling_params_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
        torch.distributed.broadcast(sampling_params_size, src, model_parallel_group)
        torch.distributed.broadcast(sampling_params_tensor, src, model_parallel_group)
    
    def receive_generate_info(self):
        """
        Needs to be synced up with send_generate_info
        """
        model_parallel_group = parallel_state.get_model_parallel_group()
        src = get_model_parallel_src_rank()
        input_info_tensor = torch.empty(3, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
        batch_size = int(input_info_tensor[0].item())
        seq_len = int(input_info_tensor[1].item())
        random_seed = int(input_info_tensor[2].item())
        if random_seed == -1:  # was converted to -1 before broadcast
            random_seed = None
    
        context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
        context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())
        # Send variables to all ranks
        torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
        torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)
    
        length_array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(length_array_size, src, model_parallel_group)
        length_params_tensor = torch.empty(length_array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
        torch.distributed.broadcast(length_params_tensor, src, model_parallel_group)
        length_bytes = length_params_tensor.cpu().numpy().tobytes()
        length_params = pickle.loads(length_bytes)
        
        sampling_array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(sampling_array_size, src, model_parallel_group)
        sampling_params_tensor = torch.empty(sampling_array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
        torch.distributed.broadcast(sampling_params_tensor, src, model_parallel_group)
        sampling_bytes = sampling_params_tensor.cpu().numpy().tobytes()
        sampling_params = pickle.loads(sampling_bytes)
    
        return (
            context_tokens_tensor,
            context_length_tensor,
            length_params,
            sampling_params,
            random_seed,
        )
    
    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]] = None,
        length_params: LengthParam = None,
        sampling_params: SamplingParam = None,
        *,
        strategy: Optional[TextGenerationStrategy] = None,
    ) -> OutputType:
        # inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings

        if self.HAVE_TRTLLM == False:
            return super().generate(inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy)

        # check whether the DDP is initialized
        if not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            if self.cfg.get('transformer_engine', False):
                self.setup_transformer_engine_tp_groups()
                self.setup_transformer_engine_cp_groups()
        
        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()
        
        if torch.distributed.get_rank() == get_model_parallel_src_rank():
            random_seed = sampling_params["random_seed"]
            if isinstance(inputs, tuple):
                context_tokens_tensor, context_length_tensor = inputs
            elif isinstance(inputs, list):
                context_tokens_tensor, context_length_tensor = self.tokenise_batch_for_generate(inputs, length_params)
            else:
                raise NotImplementedError(f"unknown type {type(inputs)} is not implemented")
            context_tokens_tensor = context_tokens_tensor.cuda(non_blocking=True)
            context_length_tensor = context_length_tensor.cuda(non_blocking=True)
            
            self.send_generate_info(
                context_tokens_tensor,
                context_length_tensor,
                length_params,
                sampling_params,
                random_seed,
            )
        else:
            (
                context_tokens_tensor,
                context_length_tensor,
                length_params,
                sampling_params,
                random_seed,
            ) = self.receive_generate_info()
        
        inputs = (context_tokens_tensor, context_length_tensor)

        #strategy_args = {} if strategy is None else {"strategy": strategy}
        
        if isinstance(inputs, list):
            batch_size = len(inputs)
        elif isinstance(inputs, tuple):
            batch_size = len(inputs[-1])

        if self.trtllm_generate is None and self.HAVE_TRTLLM is None:
            try:
                logging.info("******* TRYING TO INSTANTIATE TRT *******: ", self.cfg.encoder_seq_length - length_params["max_length"], " BS: ", batch_size)
                self.trtllm_generate = GPTGenerateTRTLLM(
                    model_cfg=self.cfg,
                    end_strings=sampling_params["end_strings"],
                    tokenizer=self.tokenizer,
                    sample_temperature=sampling_params["temperature"],
                    sample_top_k=sampling_params["top_k"],
                    sample_top_p=sampling_params["top_p"],
                    repetition_penalty=sampling_params["repetition_penalty"],
                    max_generation_length=length_params["max_length"],
                    max_input_len=self.cfg.encoder_seq_length - length_params["max_length"],
                    generation_batch_size=batch_size,
                    use_greedy=sampling_params.get("use_greedy", False),
                    trt_model_type="llama" if isinstance(self.tokenizer, AutoTokenizer) else "gptnext",
                    seed=random_seed,
                    unload_engine_train=False,
                    reshard_model=False,
                )
                
                self.prepare_for_inference()
                self.trtllm_generate.refit(self)
                clear_memory()
                
                self.HAVE_TRTLLM = True
            except Exception as e:
                logging.error(f"Error trying to instantiate TRT: {e}")
                self.HAVE_TRTLLM = False
        
        if self.HAVE_TRTLLM == False:
            return super().generate(inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy)
        
        response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations(inputs, sampling_params, length_params, prepare_for_inference=False)
        
        #try:
        #    response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations(inputs, sampling_params, length_params, prepare_for_inference=False)
        #except Exception as e:
        #    logging.error(f"Exception during `get_generations`: {e}")
        #    
        #    return super().generate(inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy)
        
        sentences, tokens, token_ids = [], [], []
        for t, s, e, v in zip(response_tokens, prompt_lengths.tolist(), response_lengths.tolist(), is_valid.tolist()):
            ints_list = t[:e].tolist()
            tokens_list = self.tokenizer.ids_to_tokens(ints_list)
            sent_list = self.tokenizer.tokens_to_text(tokens_list)
            
            sentences.append(sent_list)
            tokens.append(tokens_list)
            token_ids.append(ints_list)
        
        output = {"sentences": sentences, "tokens": tokens, "logprob": None, "full_logprob": None, "token_ids": token_ids, "offsets": None}
        
        return output
