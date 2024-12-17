# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import secrets
from dataclasses import dataclass
from typing import Literal

import tensorrt_llm
import torch
import torch.distributed
from megatron.inference.text_generation import beam_search_and_post_process, generate_and_post_process

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm import tensorrt_llm_run
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import build_tokenizer
from nemo.utils import logging
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor_within_mp,
    broadcast_2d_tensor_within_pp,
    broadcast_tensor_within_pp,
)
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.utils import clear_memory, log_memory

try:
    import tensorrt_llm

    HAVE_TRTLLM = True
except (ImportError, ModuleNotFoundError) as e:
    logging.info(f"got error message {e} when importing trt-llm dependencies, disabling")
    HAVE_TRTLLM = False


def append_and_repad_list(list_of_items, item_to_append, pad_id):
    items = [item for item in list_of_items if item != pad_id]
    items.append(item_to_append)

    # add 1 because we inserted 1 into the list
    if len(items) < len(list_of_items) + 1:
        items += [pad_id] * (len(list_of_items) + 1 - len(items))

    return items


# Use a reserved negative number since there is variation between tokenizers if
#  they (1) have a pad_id (2) don't have a pad_id or (3) have None as the pad_id.
#  This pad_id is replaced with eos_id after generation.
DEFAULT_PAD_ID = -42


@dataclass
class GenerationOutput:
    response_tokens: torch.Tensor
    response_lengths: torch.Tensor
    prompt_lengths: torch.Tensor
    is_valid: torch.Tensor


@dataclass
class InferenceGeneratorBase:

    # Model should be a subclass of MegatronGPTModel
    model: MegatronGPTModel
    end_strings: list[str]
    sample_temperature: float = 1.0
    sample_top_k: int = 0
    sample_top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_input_len: int = 1024
    max_generation_length: int = 1024
    generation_batch_size: int = 4
    use_greedy: bool = False
    seed: int | None = None
    unload_engine_train: bool = False
    reshard_model: bool = False
    refit_model: bool = False

    def __post_init__(self):
        assert isinstance(self.model, MegatronGPTModel), type(self.model)

    def generate(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> GenerationOutput:
        raise NotImplementedError


@dataclass
class TRTLLMGenerator(InferenceGeneratorBase):
    # TODO (add the others)
    trt_model_type: Literal["llama", "gptnext"] = "llama"
    trt_model_dir: str = "/tmp/trt_llm_model"

    def __post_init__(self):
        super().__post_init__()
        assert self.max_input_len > 0
        assert self.max_generation_length > 0
        assert (
            self.max_input_len + self.max_generation_length <= self.model.cfg.encoder_seq_length
        ), f"We require max_input_len ({self.max_input_len}) + max_generation_length ({self.max_generation_length}) <= model_cfg.encoder_seq_length ({self.model.cfg.encoder_seq_length})"

        if not HAVE_TRTLLM:
            raise RuntimeError(
                "You are trying to use NeMo-Aligner's TensorRT-LLM acceleration for LLM generation. Please build the dockerfile to enable this feature: https://github.com/NVIDIA/NeMo-Aligner/blob/main/Dockerfile"
            )

        if self.use_greedy and self.sample_top_k != 1:
            logging.warning(f"'use_greedy=True' overrides {self.sample_top_k=} to 1")
            self.sample_top_k = 1

        self._trt_llm_exporter = TensorRTLLM(self.trt_model_dir, load_model=False)
        self._trtllm_model_compiled = False

        rng_generator = torch.Generator(device="cpu")
        seed = secrets.randbits(32) if self.seed is None else self.seed
        rng_generator.manual_seed(seed)
        self.rng_generator = rng_generator

        self.pad_id = DEFAULT_PAD_ID
        self.eos_id = self.model.tokenizer.eos_id
        end_strings = list(self.end_strings)

        # TRT-LLM uses different logic to compute the response length depending on if we specify stop_list or end_id
        # we want to use the same logic (which is to include the stop token in response length) so we put eos_id into the stop_list
        # manually
        if len(end_strings) == 0:
            ids, offsets = [self.eos_id], [1]
        else:
            assert all(
                "," not in end_string for end_string in end_strings
            ), "detected `,` in a specified end_string. This is will cause an error when converting it into ids for TRT-LLM"

            end_strings = [[",".join(end_strings)]]
            # use an arbitary ref string to obtain end_string ids
            stop_list = tensorrt_llm_run.to_word_list_format(
                end_strings, build_tokenizer(self.model.tokenizer), ref_str="green tea icecream"
            )
            ids, offsets = stop_list[0].tolist()
            # add the eos_id if it doesn't exist
            if self.eos_id not in ids:
                ids = append_and_repad_list(ids, self.eos_id, pad_id=0)
                offsets = append_and_repad_list(offsets, max(offsets) + 1, pad_id=-1)

        assert max(offsets) == len(ids), f"offset and stop token length are mismatched ({max(offsets)=} {len(ids)=})"
        # TRT-LLM expects stop_list to be a numpy array
        stop_list = (
            torch.as_tensor([ids, offsets], dtype=torch.int32, device="cpu")
            .repeat(self.generation_batch_size, 1, 1)
            .numpy()
        )

        self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
            # We use `pad_id` as end token instead of `eos_id` to actually "disable" this end token
            # mechanism. Otherwise, the response length returned by TRT-LLM would *not* count `eos_id`
            # when the model would end its generation with this token. Instead, `stop_words_list` is
            # augmented with `eos_id` which ensures it is counted in the response length (see above).
            end_id=self.pad_id,
            pad_id=self.pad_id,
            temperature=self.sample_temperature,
            top_k=self.sample_top_k,
            top_p=self.sample_top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_generation_length,
            stop_words_list=stop_list,
            return_dict=True,
            output_sequence_lengths=True,
        )

    def refit(self):
        """Refits with model currently passed during initialization."""
        if not self._trtllm_model_compiled:
            log_memory("Before TRT-LLM engine build")
            global_devices = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_devices, torch.cuda.current_device())
            gpus_per_node = max(global_devices) + 1

            self._trt_llm_exporter.build(
                model=self.model,
                model_config=self.model.cfg,
                model_type=self.trt_model_type,
                tokenizer=self.model.tokenizer,
                gpus_per_node=gpus_per_node,
                max_input_len=self.max_input_len,
                max_output_len=self.max_generation_length,
                max_batch_size=self.generation_batch_size,
                use_refit=True,
                reshard_model=self.reshard_model,
            )
            self._trtllm_model_compiled = True
            log_memory("After TRT-LLM engine build")
        else:
            log_memory("Before TRT-LLM engine refit")
            self._trt_llm_exporter.refit(self.model, self.model.cfg)
            log_memory("After TRT-LLM engine refit")

    def _generate(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Internal API to make it easier to validate raw TRT-LLM outputs
        """
        prompt_tokens, prompt_lengths = inputs

        batch_input_ids = []
        for idx in range(prompt_tokens.shape[0]):
            batch_input_ids.append(prompt_tokens[idx][0 : prompt_lengths[idx]].cpu())

        random_seeds = torch.randint(
            0, 2 ** 32, size=(prompt_tokens.shape[0],), dtype=torch.long, generator=self.rng_generator
        )
        self.sampling_config.update(random_seed=random_seeds)

        output_dict = tensorrt_llm_run.tensorrt_llm_worker_context.decoder.generate(
            batch_input_ids=batch_input_ids, sampling_config=self.sampling_config, streaming=False
        )

        # TRTLLM returns the output_ids and sequence_lengths only on the first PP rank, and None otherwise, so we need to broadcast
        output_ids = broadcast_tensor_within_pp(output_dict["output_ids"] if output_dict else None, from_last=False)
        response_lengths = broadcast_tensor_within_pp(
            output_dict["sequence_lengths"] if output_dict else None, from_last=False
        )

        # remove beam dim from output_ids: [mbs, beam_dim, sequence len]
        output_ids = torch.squeeze(output_ids, dim=1).long()
        response_lengths = torch.squeeze(response_lengths, dim=1).long()
        return output_ids, response_lengths

    def generate(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> GenerationOutput:
        _, prompt_lengths = inputs
        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self.model, context_lengths=prompt_lengths, max_length=self.max_generation_length
        )
        response_tokens, response_lengths = self._generate(inputs)
        max_length = response_lengths.max().item()

        # Map pad_id to eos_id in case tokenizer does not have a pad_id
        response_tokens[response_tokens == self.pad_id] = self.eos_id
        response_tokens = response_tokens[..., :max_length].contiguous()
        response_tokens = broadcast_2d_tensor_within_mp(response_tokens, dtype=response_tokens.dtype)

        # sometimes backends like TRT-LLM will generate invalid tokens
        # so we need to also inplace mutate the response_tokens to be within the tokenizer range
        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.model.tokenizer, self.end_strings
        )
        return GenerationOutput(
            response_tokens=response_tokens,
            response_lengths=response_lengths,
            prompt_lengths=prompt_lengths,
            is_valid=is_valid,
        )

    def free(self):
        if not self.unload_engine_train:
            return
        log_memory("Before TRT-LLM engine unload")
        self._trt_llm_exporter.unload_engine()
        clear_memory()
        log_memory("After TRT-LLM engine unload")


@dataclass
class MegatronGenerator(InferenceGeneratorBase):
    def __post_init__(self):
        super().__post_init__()
        assert not self.reshard_model, "reshard_model is not supported"
        assert not self.refit_model, "refit_model is not supported"
        assert not self.unload_engine_train, "unload_engine_train is not supported"

        if not self.model.model.config.flash_decode:
            logging.warning("Flash decode is not enabled, consider adding ++model.flash_decode=True")
        if not self.model.model.config.enable_cuda_graph:
            logging.warning("CudaGraphs is not enabled, consider adding ++model.enable_cuda_graph=True")

        class generation_args:
            max_position_embeddings = self.model.cfg.max_position_embeddings
            # TODO: appears to be a soft check for total num of input tokens, for now don't check
            max_tokens_to_oom = float("inf")
            inference_max_seq_length = self.model.cfg.encoder_seq_length
            enable_cuda_graph = self.model.model.config.enable_cuda_graph
            eos_id = self.model.tokenizer.eos_id
            inference_batch_times_seqlen_threshold = -1
            # TODO: Look into whether it's okay to just use the tokenizer's args (is it padded?)
            padded_vocab_size = self.model.tokenizer.vocab_size

        self._generation_args = generation_args

    def generate(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> GenerationOutput:
        prompt_tokens, prompt_lengths = inputs
        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self.model, context_lengths=prompt_lengths, max_length=self.max_generation_length,
        )
        mcore_result = generate_and_post_process(
            model=self.model.model,
            prompts=(prompt_tokens, prompt_lengths),
            tokens_to_generate=self.max_generation_length,  # Must be 0 if passing prompt_tokens + prompt_lengths
            prevent_newline_after_colon=False,  # Turning this on requires a global tokenizer, so we leave it off
            top_k_sampling=self.sample_top_k,  # 1 == greedy
            top_p_sampling=self.sample_top_p,
            temperature=self.sample_temperature,
            random_seed=self.seed,
            return_detokenize=False,  # detokenize triggers tokenizer logic, which we avoid b/c we BYO
            generation_args=self._generation_args,
        )
        # Mcore inference returns None if not on the first PP rank
        if mcore_result is not None:
            response_tokens, response_lengths = mcore_result.tokens, mcore_result.lengths
            response_tokens = torch.tensor(response_tokens, dtype=torch.long, device="cuda")
            response_lengths = torch.tensor(response_lengths, dtype=torch.long, device="cuda")
        else:
            response_tokens = None
        response_tokens = broadcast_2d_tensor_within_pp(response_tokens, dtype=torch.long, from_last=False)
        response_lengths = broadcast_2d_tensor_within_pp(response_lengths, dtype=torch.long, from_last=False)

        # sometimes backends like TRT-LLM will generate invalid tokens
        # so we need to also inplace mutate the response_tokens to be within the tokenizer range
        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.model.tokenizer, self.end_strings
        )

        return GenerationOutput(
            response_tokens=response_tokens,
            response_lengths=response_lengths,
            prompt_lengths=prompt_lengths,
            is_valid=is_valid,
        )


@dataclass
class NemoGenerator(InferenceGeneratorBase):
    def __post_init__(self):
        super().__post_init__()
        assert not self.reshard_model, "reshard_model is not supported"
        assert not self.refit_model, "refit_model is not supported"
        assert not self.unload_engine_train, "unload_engine_train is not supported"

        if not self.model.model.config.flash_decode:
            logging.warning("Flash decode is not enabled, consider adding ++model.flash_decode=True")
        if not self.model.model.config.enable_cuda_graph:
            logging.warning("CudaGraphs is not enabled, consider adding ++model.enable_cuda_graph=True")

        # length argument for autoregressive sampling
        self._length_params = {
            # max length means max amount of tokens to generate
            "max_length": self.max_generation_length,  # Set to ${int_div:${model.encoder_seq_length}, 2} in ppo
            "min_length": 1,
        }
        # sampling parameters for generation
        self._sampling_params = {
            "use_greedy": self.use_greedy,
            "temperature": self.sample_temperature,
            "top_k": self.sample_top_k,
            "top_p": self.sample_top_p,
            "repetition_penalty": self.repetition_penalty,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
            # will be used in NeMo version > 1.20.0
            # keeping it for now
            "end_strings": self.end_strings,
        }

    def generate(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> GenerationOutput:
        prompt_tokens, prompt_lengths = inputs
        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self.model, context_lengths=prompt_lengths, max_length=self.max_generation_length
        )
        actor_output = self.model.generate(
            inputs=(prompt_tokens, prompt_lengths),
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
                or response_tokens.size(1) != prompt_lengths.max().item() + self.max_generation_length
            ):
                raise AssertionError(
                    f"max response length ({max_response_length}) does not match the size of "
                    f"`response_tokens` ({response_tokens.size(1)})"
                )

        # sometimes backends like TRT-LLM will generate invalid tokens
        # so we need to also inplace mutate the response_tokens to be within the tokenizer range
        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens, response_lengths, strategy, self.model.tokenizer, self.end_strings
        )

        return GenerationOutput(
            response_tokens=response_tokens,
            response_lengths=response_lengths,
            prompt_lengths=prompt_lengths,
            is_valid=is_valid,
        )
