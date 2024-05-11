from pathlib import Path
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl
from tensorrt_llm.builder import get_engine_version

import json
from typing import Optional

from transformers import AutoTokenizer, T5Tokenizer
from tensorrt_llm.runtime import ModelRunnerCpp
import torch


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    if model_arch == 'QWenForCausalLM':
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version

def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM' or model_name == 'RecurrentGemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)

    if model_name == 'QWenForCausalLM' and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


class TRTLLMInference:
    
    def __init__(self, cfg, end_id, pad_id) -> None:
        runner_cls = ModelRunnerCpp
        runner_kwargs = dict(engine_dir=cfg.trtllm.engine_path,
                             rank=0,
                             debug_mode=False,
                             gpu_weights_percent=1)
        runner_kwargs.update(
            max_batch_size=cfg.model.mcts.rollout_micro_batch_size,
            max_input_len=3072,
            max_output_len=1,
            max_beam_width=1,
            max_attention_window_size=None,
            sink_token_length=None)
        self.end_id = end_id
        self.pad_id = pad_id
        self.runner = runner_cls.from_dir(**runner_kwargs)

    def evaluate(self, batch_input_ids):
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids,
                max_new_tokens=1,
                max_attention_window_size=None,
                sink_token_length=None,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=0.1,
                top_k=1,
                top_p=0.0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=None)
            torch.cuda.synchronize()
        return outputs


