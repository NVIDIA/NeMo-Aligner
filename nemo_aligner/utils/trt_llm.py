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
import secrets
import tensorrt_llm
import torch
import torch.distributed

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm import tensorrt_llm_run
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import build_tokenizer
from nemo.utils import logging
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, broadcast_tensor_within_pp
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


class GPTGenerateTRTLLM:
    # Use a reserved negative number since there is variation between tokenizers if
    #  they (1) have a pad_id (2) don't have a pad_id or (3) have None as the pad_id.
    #  This pad_id is replaced with eos_id after generation.
    DEFAULT_PAD_ID = -42

    def __init__(
        self,
        model_cfg,
        end_strings,
        tokenizer,
        sample_temperature=1.0,
        sample_top_k=0,
        sample_top_p=1.0,
        repetition_penalty=1.0,
        max_generation_length=1024,
        max_input_len=1024,
        generation_batch_size=4,
        use_greedy=False,
        trt_model_type="llama",
        seed=None,
        unload_engine_train=False,
        reshard_model=False,
        trt_model_dir="/tmp/trt_llm_model",
    ):
        if not HAVE_TRTLLM:
            raise RuntimeError(
                "You are trying to use NeMo-Aligner's TensorRT-LLM acceleration for LLM generation. Please build the dockerfile to enable this feature: https://github.com/NVIDIA/NeMo-Aligner/blob/main/Dockerfile"
            )

        assert max_input_len > 0
        assert max_generation_length > 0
        assert (
            max_input_len + max_generation_length <= model_cfg.encoder_seq_length
        ), f"We require max_input_len ({max_input_len}) + max_generation_length ({max_generation_length}) <= model_cfg.encoder_seq_length ({model_cfg.encoder_seq_length})"

        if use_greedy and sample_top_k != 1:
            logging.warning(f"'use_greedy=True' overrides {sample_top_k=} to 1")
            sample_top_k = 1

        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length
        self.max_input_len = max_input_len
        self.generation_batch_size = generation_batch_size
        self.unload_engine_train = unload_engine_train
        self.trt_model_type = trt_model_type
        self.reshard_model = reshard_model
        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self._trtllm_model_compiled = False

        rng_generator = torch.Generator(device="cpu")
        seed = secrets.randbits(32) if seed is None else seed
        rng_generator.manual_seed(seed)
        self.rng_generator = rng_generator

        self.pad_id = GPTGenerateTRTLLM.DEFAULT_PAD_ID
        self.eos_id = tokenizer.eos_id
        end_strings = list(end_strings)

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
                end_strings, build_tokenizer(self.tokenizer), ref_str="green tea icecream"
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
            temperature=sample_temperature,
            top_k=sample_top_k,
            top_p=sample_top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=self.max_generation_length,
            stop_words_list=stop_list,
            return_dict=True,
            output_sequence_lengths=True,
        )

    def refit(self, model):
        if not self._trtllm_model_compiled:
            log_memory("Before TRT-LLM engine build")
            global_devices = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_devices, torch.cuda.current_device())
            gpus_per_node = max(global_devices) + 1

            self.trt_llm_exporter.build(
                model=model,
                model_config=self.model_cfg,
                model_type=self.trt_model_type,
                tokenizer=self.tokenizer,
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
            self.trt_llm_exporter.refit(model, self.model_cfg)
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

    def generate(self, inputs: tuple[torch.Tensor, torch.Tensor]):

        output_ids, response_lengths = self._generate(inputs)
        max_length = response_lengths.max().item()

        # Map pad_id to eos_id in case tokenizer does not have a pad_id
        output_ids[output_ids == self.pad_id] = self.eos_id
        output_ids = output_ids[..., :max_length].contiguous()
        output_ids = broadcast_2d_tensor_within_mp(output_ids, dtype=output_ids.dtype)

        output = {
            "response_tokens": output_ids,
            "response_lengths": response_lengths,
        }

        return output

    def free(self):
        if not self.unload_engine_train:
            return
        log_memory("Before TRT-LLM engine unload")
        self.trt_llm_exporter.unload_engine()
        clear_memory()
        log_memory("After TRT-LLM engine unload")
