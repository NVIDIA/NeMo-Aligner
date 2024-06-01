import json
from pathlib import Path
from typing import Optional

import sentencepiece
import tensorrt_llm
import tensorrt_llm.profiler as profiler
import torch
import torch.nn.functional as F
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import get_engine_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner, ModelRunnerCpp
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCppGptSession
from tensorrt_llm.tools.ppl import ppl
from transformers import AutoTokenizer, T5Tokenizer

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.transformer.text_generation import OutputType
import numpy as np
from typing import List
import csv


def to_word_list_format(
    words: List[str],
    tokenizer=None,
    ref_str="green tea icecream",
):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    # The encoding of a single word can't always be trusted. See
    #   https://github.com/NVIDIA/NeMo/blob/bb575b72fd0be51ae10cc77d9f89ddb9e9d3b96d/nemo/collections/nlp/modules/common/text_generation_strategy.py#L229
    ids_ref = tokenizer.encode(ref_str)
    item_flat_ids = []
    item_offsets = []

    for word in words:
        ids = tokenizer.encode(f"{ref_str}{word}")
        if ids[0 : len(ids_ref)] == ids_ref:
            # It worked! We can obtain the token(s) associated to `word` by stripping the prefix tokens.
            ids = ids[len(ids_ref) :]
        else:
            # Unfortunately the prefix was merged with `word`. We could try with a different prefix, but
            # for now we just use the basic encoding since this should be a very rare edge case.
            ids = tokenizer.encode(word)
            logger.warning(f"The encoding of word '{word}' into tokens {ids} might be incorrect")

        if len(ids) == 0:
            continue

        item_flat_ids += ids
        item_offsets.append(len(ids))

    flat_ids.append(np.array(item_flat_ids))
    offsets.append(np.cumsum(np.array(item_offsets)))
    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))   
        
def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", "r") as f:
        config = json.load(f)

    if engine_version is None:
        return config["builder_config"]["name"], None

    model_arch = config["pretrained_config"]["architecture"]
    model_version = None
    if model_arch == "ChatGLMForCausalLM":
        model_version = config["pretrained_config"]["chatglm_version"]
    if model_arch == "QWenForCausalLM":
        model_version = config["pretrained_config"]["qwen_type"]
    return model_arch, model_version


def load_tokenizer(
    tokenizer_dir: Optional[str] = None,
    vocab_file: Optional[str] = None,
    model_name: str = "GPTForCausalLM",
    model_version: Optional[str] = None,
    tokenizer_type: Optional[str] = None,
):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            tokenizer_type=tokenizer_type,
            use_fast=use_fast,
        )
    elif model_name == "GemmaForCausalLM" or model_name == "RecurrentGemmaForCausalLM":
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file, padding_side="left", truncation_side="left", legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        # use sentence piece tokenizer
        # tokenizer = T5Tokenizer(vocab_file=vocab_file, padding_side="left", truncation_side="left", legacy=False)
        tokenizer = sentencepiece.SentencePieceProcessor(vocab_file)
    if model_name == "QWenForCausalLM" and model_version == "qwen":
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config["pad_token_id"]
        end_id = gen_config["eos_token_id"]
        bos_id = gen_config["bos_token_id"]
    elif model_name == "ChatGLMForCausalLM" and model_version == "glm":
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
        bos_id = tokenizer.bos_token_id
    else:
        pad_id = tokenizer.pad_id()
        end_id = tokenizer.eos_id()
        bos_id = tokenizer.bos_id()
        # if tokenizer.pad_token_id is None:
        #     tokenizer.pad_token_id = tokenizer.eos_token_id
        # pad_id = tokenizer.pad_token_id
        # end_id = tokenizer.eos_token_id
        # bos_id = tokenizer.bos_token_id
    return tokenizer, pad_id, end_id, bos_id



class TRTLLMFastGeneration:
    def __init__(self, cfg, end_id, pad_id, bos_id, tokenizer, stop_words=['<extra_id_1>', '\x11']) -> None:
        # runner_cls = ModelRunnerCpp
        runner_cls = ModelRunnerCppGptSession
        runner_kwargs = dict(engine_dir=cfg.trtllm.engine_path, rank=0, debug_mode=False, gpu_weights_percent=1)
        runner_kwargs.update(
            max_batch_size=1,
            max_input_len=1024,
            max_output_len=1024,
            max_beam_width=1,
            max_attention_window_size=None,
            sink_token_length=None,
        )
        self.end_id = end_id
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.runner = runner_cls.from_dir(**runner_kwargs)
        self.tokenizer = tokenizer
        # stop_word_ids = to_word_list_format(stop_words, tokenizer)
        # # self.stop_list = torch.from_numpy(stop_word_ids)[0].tolist()
        # self.stop_list = torch.from_numpy(stop_word_ids).cuda().contiguous()

        self.stop_list =  to_word_list_format(stop_words, tokenizer)

        self.stop_list = torch.from_numpy(self.stop_list).cuda().contiguous()
        # stop_word_ids = [tokenizer.encode(word) for word in stop_words]
        # flat_stop_word_ids = [item for sublist in stop_word_ids for item in sublist]
        # stop_word_lengths = [len(ids) for ids in stop_word_ids]

        # stop_word_lengths = stop_word_lengths + [-1] * (len(flat_stop_word_ids) - len(stop_word_lengths))
        # # self.stop_words_tensor = torch.tensor([flat_stop_word_ids, stop_word_lengths], dtype=torch.int32).cuda()
        # self.stop_words_tensor = [flat_stop_word_ids, stop_word_lengths]


    def evaluate(self, batch_input_ids, max_new_tokens=1):
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=self.end_id,
            pad_id=self.pad_id,  # TODO
            temperature=0.1,
            top_k=1,
            top_p=0.0,
            max_new_tokens=max_new_tokens,
            stop_words_list=self.stop_list,
            return_dict=True,
            output_sequence_lengths=True,
        )
        with torch.no_grad():
            # outputs = self.runner.generate(
            #     batch_input_ids, 
            #      max_new_tokens=max_new_tokens,
            #      max_attention_window_size=None,
            #      sink_token_length=None,
            #      end_id=self.end_id,
            #      pad_id=self.pad_id,
            #      temperature=0.1,
            #      top_k=1,
            #      top_p=0.0,
            #      num_beams=1,
            #      length_penalty=1.0,
            #      early_stopping=True,
            #      repetition_penalty=1.0,
            #      presence_penalty=0.0,
            #      frequency_penalty=0.0,
            #      output_sequence_lengths=True,
            #      return_dict=True,
            #      medusa_choices=None,
            #      stop_words_list=self.stop_list,
            # )
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids, 
                sampling_config=sampling_config, 
                streaming=False
            )
            torch.cuda.synchronize()
        return outputs

    def tokenize_batch(self, sentences, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            sentences (List[str]): list of input sentences in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.tokenizer
        if add_BOS:
            context_tokens = [torch.tensor([self.bos_id] + tokenizer.encode(s)) for s in sentences]
        else:
            context_tokens = [torch.tensor(tokenizer.encode(s)) for s in sentences]
        return context_tokens
