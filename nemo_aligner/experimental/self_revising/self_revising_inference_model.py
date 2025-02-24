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

import jinja2
from jinja2 import meta
jinja2_env = jinja2.Environment()


DEFAULT_CRITIQUE_PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

Please assess the Assistant's Answer inside the brackets above and provide a detailed critique of how helpful you believe this Answer to be in relation to the User's query. Provide a moderate length (2-10 sentences / 50-250 words) justification for your critique of the Answer.

Do not include links used for fact-checking
Avoid first person statements ("I think that...")
Avoid vague statements/lack of specificity
Avoid lists (numbered, bulleted, etc.)
Ensure all sentences are complete and with no grammatical/spelling errors

You should provide negative feedback for Answers which are too verbose or overly long, especially Answers which have repetition of sentences or phrases throughout their response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

DEFAULT_REVISE_PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Below is a conversation between a User and an AI Assistant.

{{ prompt }}

[The start of the Assistant's Answer]
{{ response }}
[The end of the Assistant's Answer]

The Assistant's Answer was sent to {{ num_annotators }} human annotator{{ s_or_not }} to evaluate its quality.
Below are the comments of the annotator{{ s_or_not }}:

[Start of annotator comments]
{{ critique }}
[End of annotator comments]

You must revise the Assistant's Answer to improve it according to the feedback above, only changing what is required to address this feedback.
Reply with the Revised Response only, do not include any additional introductory text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def find_variables_from_jinja_template(template: str):
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)

def exists(v):
    return v is not None

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


class SelfRevisingInferenceModel(Inferrable, MegatronGPTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        
        logging.info("********** INIT FOR SelfRevisingInferenceModel ********")

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
        
        self.num_responses_to_gen = 1
        self.num_critiques_to_gen = self.cfg.spin.get("num_critiques_to_gen", 1) if "spin" in self.cfg else 3
        self.critique_template = DEFAULT_CRITIQUE_PROMPT_LLAMA3 #self.cfg.spin.get("critique_prompt_template")
        self.revise_template = DEFAULT_REVISE_PROMPT_LLAMA3 #self.cfg.spin.get("revise_prompt_template")
        assert find_variables_from_jinja_template(self.critique_template) == {
            "prompt",
            "response",
        }, "critique_prompt_template must include `prompt` and `response` templating variables"
        #assert find_variables_from_jinja_template(self.revise_template) == {
        #    "num_annotators",
        #    "s_or_not",
        #    "critique",
        #}, "revise_prompt_template must include `num_annotators`, `s_or_not`, and `critique` templating variables"
        assert find_variables_from_jinja_template(self.revise_template) == {
            "prompt",
            "response",
            "num_annotators",
            "s_or_not",
            "critique",
        }, "revise_prompt_template must include `prompt`, `response`, `num_annotators`, `s_or_not`, and `critique` templating variables"
        self.critique_template_fn = jinja2_env.from_string(self.critique_template).render
        self.revise_template_fn = jinja2_env.from_string(self.revise_template).render

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
    
    def get_tagged_responses(self, list_of_strings, sampling_params, length_params, prepare_for_inference=False):
        tagged_responses = []
        responses, prompt_lengths, resp_lengths, is_end = self.get_generations(list_of_strings, sampling_params, length_params, prepare_for_inference=prepare_for_inference)
        batch_responses_str = []
        for t, s, e, end in zip(responses, prompt_lengths.tolist(), resp_lengths.tolist(), is_end.tolist()):
            response = self.tokenizer.ids_to_text(t[s:e].tolist())
            batch_responses_str.append(response)
            #if torch.distributed.get_rank() == 0 and torch.distributed.get_rank() == parallel_state.get_data_parallel_src_rank():
            #    print(f"*** TAGGED_PROMPT_AND_RESP: {self.tokenizer.ids_to_text(t[:e].tolist())}")
            #    print(f"*** TAGGED_RESP_ONLY: {self.tokenizer.ids_to_text(t[s:e].tolist())}")
        #tagged_as_str = [regex_fn(resp_str.strip()) for resp_str in batch_responses_str]
        tagged_as_str = [resp_str.replace("<|eot_id|>", "").strip() for resp_str in batch_responses_str]
        #tagged_as_str = [list(set([resp_str.replace(x, "").strip() for x in sampling_params["end_strings"]]))[0] for resp_str in batch_responses_str]
        for idx, (r, end) in enumerate(zip(tagged_as_str, is_end.tolist())):
            tagged_responses.append(
                r if (end and (r is not None)) else None
                #r if (r is not None) else None
            )

        return tagged_responses
    
    def get_self_revision_generations(self, inputs, sampling_params, length_params, prepare_for_inference=False):
        allowed_prompt_length = self.cfg.encoder_seq_length - length_params["max_length"] - 148
        N = len(inputs) if isinstance(inputs, list) else len(inputs[-1])
        candidate_responses_with_critiques = [
            [] for _ in range(N)
        ]
        for _ in range(self.num_responses_to_gen):
            # Generation happens on GPU but returned tensors are on CPU so as not to blow up VRAM due to self.num_responses_to_gen
            gen_tokens_buf, gen_prompt_lengths_buf, gen_lengths_buf, is_end = self.get_generations(inputs, sampling_params, length_params, prepare_for_inference=prepare_for_inference)

            # Transform into batch of LLM-as-judge template samples for reward scoring
            orig_prompts_and_responses, critique_buffer = [], []
            for t, s, e in zip(gen_tokens_buf, gen_prompt_lengths_buf.tolist(), gen_lengths_buf.tolist()):
                
                if isinstance(self.tokenizer, AutoTokenizer):
                    prompt = self.tokenizer.ids_to_text(t[:s].tolist()).replace(
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", ""
                    )
                    response = (
                        self.tokenizer.ids_to_text(t[s:e].tolist()).replace("<|eot_id|>", "").strip()
                    )
                else:
                    prompt = self.tokenizer.ids_to_text(t[:s].tolist()).replace(
                        "<extra_id_0>System\n\n", ""
                    )
                    response = self.tokenizer.ids_to_text(t[s:e].tolist()).replace("\n<extra_id_1>", "")
                
                #prompt, response, last_prompt = self.normalise_prompt(
                #    self.tokenizer.ids_to_text(t[:s].tolist()),
                #    self.tokenizer.ids_to_text(t[s:e].tolist()),
                #    buffer[0]["dataset_mask"],
                #)
                critique_prompt_str = self.critique_template_fn(prompt=prompt, response=response)
                critique_prompt = self.tokenizer.text_to_ids(critique_prompt_str)
                if len(critique_prompt) > allowed_prompt_length:
                    p_list, _, _ = self.extract_prompt_elements(
                        prompt, response, "<|start_header_id|>user<|end_header_id|>"
                    )
                    if len(p_list) == 0:
                        prompt_ft = prompt
                    else:
                        prompt_ft = p_list[-1]
                    response_ft = response
                    critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
                    critique_prompt = self.tokenizer.text_to_ids(critique_prompt_str)

                    while len(critique_prompt) > allowed_prompt_length:
                        overage = len(critique_prompt) - allowed_prompt_length
                        if overage > len(self.tokenizer.text_to_ids(response_ft)):
                            # print(f"*** OVERAGE_NOT_FIT_RESPONSE: {reward_prompt_str}")
                            critique_prompt_str = self.critique_template_fn(
                                prompt="How does one make tea?", response="I have no answer at all."
                            )
                            critique_prompt = self.tokenizer.text_to_ids(critique_prompt_str)
                            break
                        response_ft = self.tokenizer.ids_to_text(
                            self.tokenizer.text_to_ids(response_ft)[:-overage]
                        )
                        critique_prompt_str = self.critique_template_fn(prompt=prompt_ft, response=response_ft)
                        critique_prompt = self.tokenizer.text_to_ids(critique_prompt_str)
                    prompt = prompt_ft
                    response = response_ft
                    assert len(critique_prompt) <= (
                        allowed_prompt_length
                    ), f"truncation of response only failed [ {len(critique_prompt)} ]: {critique_prompt_str}"

                critique_buffer.append(critique_prompt_str)
                orig_prompts_and_responses.append( (prompt, response) )

            
            critiques_list = [
                [] for _ in range(N)
            ]
            for _ in range(self.num_critiques_to_gen):
                critiques_as_str = self.get_tagged_responses(critique_buffer, sampling_params, length_params, prepare_for_inference=prepare_for_inference)
                
                for idx, critique in enumerate(critiques_as_str):
                    critiques_list[idx].append(critique)
            
            revise_buffer, bad_idxs = [], []
            for idx, ((prompt, response), critique_list) in enumerate(zip(orig_prompts_and_responses, critiques_list)):
                print("*** RAW_CRITIQUE_LIST: ", critique_list)
                cl_valid = [*filter(exists, critique_list)]
                if len(cl_valid) == 0:
                    critique = "I have no critique."
                    bad_idxs.append(idx)
                else:
                    cl_valid = list(set(cl_valid))
                    critique = "\n-------------------------------------------\n".join(cl_valid)
                revise_prompt_str = self.revise_template_fn(prompt=prompt, response=response, num_annotators=str(max(1, len(cl_valid))), s_or_not="" if len(cl_valid) <= 1 else "s", critique=critique)
                revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                if len(revise_prompt) > allowed_prompt_length:

                    p_list, r_list, resp_raw = self.extract_prompt_elements(
                        prompt, response, "<|start_header_id|>user<|end_header_id|>"
                    )
                    if len(p_list) == 0:
                        prompt_ft = prompt
                    else:
                        prompt_ft = p_list[-1]
                    response_ft = resp_raw
                    revise_prompt_str = self.revise_template_fn(prompt=prompt_ft, response=response_ft, num_annotators=str(max(1, len(cl_valid))), s_or_not="" if len(cl_valid) <= 1 else "s", critique=critique)
                    revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)

                    while len(revise_prompt) > (
                        allowed_prompt_length
                    ):
                        overage = len(revise_prompt) - allowed_prompt_length
                        if overage > len(self.tokenizer.text_to_ids(prompt_ft)):
                            spillover = overage - len(self.tokenizer.text_to_ids(prompt_ft))
                            response_ft = self.tokenizer.ids_to_text(
                                self.tokenizer.text_to_ids(response_ft)[:-spillover]
                            )
                            revise_prompt_str = self.revise_template_fn(prompt="", response=response_ft, num_annotators=str(max(1, len(cl_valid))), s_or_not="" if len(cl_valid) <= 1 else "s", critique=critique)
                            revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                            break
                        prompt_ft = self.tokenizer.ids_to_text(
                            self.tokenizer.text_to_ids(prompt_ft)[:-overage]
                        )
                        revise_prompt_str = self.revise_template_fn(prompt=prompt_ft, response=response_ft, num_annotators=str(max(1, len(cl_valid))), s_or_not="" if len(cl_valid) <= 1 else "s", critique=critique)
                        revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                    
                    if len(revise_prompt) > allowed_prompt_length:
                        revise_prompt_str = self.revise_template_fn(prompt="", response=response_ft, num_annotators=str(max(1, len(cl_valid))), s_or_not="" if len(cl_valid) <= 1 else "s", critique="I have no critique.")
                        revise_prompt = self.tokenizer.text_to_ids(revise_prompt_str)
                        #bad_idxs.append(idx)
                assert len(revise_prompt) <= (
                        allowed_prompt_length
                    ), f"truncation of revise template failed [ {len(revise_prompt)} ]: {revise_prompt_str}"
            
                revise_buffer.append(revise_prompt_str)
            
            revisions_as_str = self.get_tagged_responses(revise_buffer, sampling_params, length_params, prepare_for_inference=prepare_for_inference)
            revisions_as_str = [("NULL" if i in bad_idxs else x) for i,x in enumerate(revisions_as_str)]

            for idx, (t, s, e, end, op_and_r, critique, revise) in enumerate(
                zip(
                    gen_tokens_buf,
                    gen_prompt_lengths_buf.tolist(),
                    gen_lengths_buf.tolist(),
                    is_end.tolist(),
                    orig_prompts_and_responses,
                    critiques_list,
                    revisions_as_str,
                )
            ):
                candidate_responses_with_critiques[idx].append((t, s, e, op_and_r[0], op_and_r[-1], critique, revise, end))

        sentences, tokens, token_ids = [], [], []
        for cand_list in candidate_responses_with_critiques:
            idx_chosen = 0
            cand_selected = cand_list[idx_chosen]
            
            prompt_len = cand_selected[1]
            #orig_generated_len = cand_selected[2]
            prompt_tokens = cand_selected[0][:prompt_len].cpu()
            chosen_resp_str = (cand_selected[-2] if cand_selected[-2] is not None else "NULL") + "<|eot_id|>"
            chosen_resp_tokens = torch.LongTensor(self.tokenizer.text_to_ids(chosen_resp_str))

            # 1 x max_len tensor
            #chosen_prompt_len = prompt_len
            chosen_token_ids = torch.cat([prompt_tokens, chosen_resp_tokens], dim=0).tolist()
            chosen_tokens = self.tokenizer.ids_to_tokens(chosen_token_ids)
            
            token_ids.append(chosen_token_ids)
            tokens.append(chosen_tokens)
            sentences.append(self.tokenizer.tokens_to_text(chosen_tokens))
        
        return {"sentences": sentences, "tokens": tokens, "logprob": None, "full_logprob": None, "token_ids": token_ids, "offsets": None}
    
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
        
        return self.get_self_revision_generations(inputs, sampling_params, length_params, prepare_for_inference=False)
        
        '''
        try:
            response_tokens, prompt_lengths, response_lengths, is_valid = self.get_generations(inputs, sampling_params, length_params, prepare_for_inference=False)
        except Exception as e:
            logging.error(f"Exception during `get_generations`: {e}")
            
            return super().generate(inputs=inputs, length_params=length_params, sampling_params=sampling_params, strategy=strategy)
        
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
        '''
