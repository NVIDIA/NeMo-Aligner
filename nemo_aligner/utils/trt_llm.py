import secrets

import tensorrt_llm
import torch

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm import tensorrt_llm_run
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import build_tokenizer
from nemo.utils import logging
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp
from nemo_aligner.utils.utils import log_memory

try:
    from tensorrt_llm.bindings import GptSession

    GptSession.refit_engine  # check if TRTLLM Cpp runtime was compiled with engine refitting
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
        max_input_tokens=4096,
        generation_batch_size=4,
        use_greedy=False,
        trt_model_type="GPTForCausalLM",
        seed=None,
        unload_engine_train=False,
        reshard_model=False,
        trt_model_dir="/tmp/trt_llm_model",
    ):
        # If this assert turns out to be a blocker with some tokenizers, potential workarounds could be to:
        #   - add a config option to allow specifying which token we pass as `end_id` to TRT-LLM (should
        #     be a token that the model is guaranteed to never generate)
        #   - pass `end_id=-1` (and possibly also `pad_id=-1`) to TRT-LLM (would require making sure
        #     this works as intended)
        assert (
            tokenizer.pad_id != tokenizer.eos_id
        ), "We require tokenizers to have a different pad_id than eos_id when using TRT-LLM. This is to make sure all code goes into the same path and include the eos_id when the response lengths are computed"

        if use_greedy and sample_top_k != 1:
            logging.warning(f"'use_greedy=True' overrides {sample_top_k=} to 1")
            sample_top_k = 1

        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length
        self.max_input_len = max_input_len
        self.max_input_tokens = max_input_tokens
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

        self.pad_id = tokenizer.pad_id
        end_id = tokenizer.eos_id
        end_strings = list(end_strings)

        # TRT-LLM uses different logic to compute the response length depending on if we specify stop_list or end_id
        # we want to use the same logic (which is to include the stop token in response length) so we put end_id into the stop_list
        # manually
        if len(end_strings) == 0:
            ids, offsets = [end_id], [1]
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
            # add the end_id if it doesn't exist
            if end_id not in ids:
                ids = append_and_repad_list(ids, end_id, pad_id=0)
                offsets = append_and_repad_list(offsets, max(offsets) + 1, pad_id=-1)

        assert max(offsets) == len(ids), "offset and stop token length are mismatched"
        stop_list = torch.as_tensor([ids, offsets], dtype=torch.int32, device=torch.cuda.current_device()).repeat(
            self.generation_batch_size, 1, 1
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
            log_memory("memory before TRT-LLM engine build")
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
            log_memory("memory after TRT-LLM engine build")
        else:
            log_memory("memory before TRT-LLM engine refit")
            self.trt_llm_exporter.refit(model, self.model_cfg)
            log_memory("memory after TRT-LLM engine refit")

    def generate(self, inputs):
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

        # remove beam dim from output_ids: [mbs, beam_dim, sequence len]
        output_ids = torch.squeeze(output_dict["output_ids"], dim=1).long()
        response_lengths = torch.squeeze(output_dict["sequence_lengths"], dim=1).long()
        max_length = response_lengths.max().item()

        # TRTLLM with PP erroneously inserts padding:
        # As an example when we have the input:
        #     [[prompt tok, PAD, PAD], [prompt tok, prompt tok, prompt tok]]
        # The output when PP is enabled becomes:
        #     [[prompt tok, PAD, PAD, resp_tok, resp_tok], [prompt tok, prompt tok, prompt tok, resp_tok, resp_tok]]
        # Therefore we need this logic to get rid of the padding in the middle of the tensor.
        # Furthermore, TRTLLM only produces valid outputs on the source rank, so we can only process it here
        # and rely on the aligner broadcast to get it to the other ranks. Miraculously, the length
        # is still correct on the non src ranks
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_model_parallel_src_rank() == torch.distributed.get_rank()
        ):
            valid_tokens = output_ids != self.pad_id
            # we can't just naively use the response length here
            # because there are cases where the model generates
            # stop strings after it has stopped. so we need to
            # be slightly inefficient and then remove the excess later on
            valid_token_lengths = valid_tokens.sum(-1, keepdims=True)
            max_unpadded_length = valid_token_lengths.max()
            assert max_length <= max_unpadded_length, (
                "max unpadded length should be more or equal to max length. This assertion is probably happening because TRT-LLM considered a "
                "pad tokens in the response length"
            )

            _output_ids = torch.full(
                (response_lengths.size(0), max_unpadded_length),
                fill_value=self.pad_id,
                dtype=output_ids.dtype,
                device=output_ids.device,
            )

            # only fill up to the amount of valid tokens
            src_index_mask = (
                torch.arange(max_unpadded_length, device=response_lengths.device).view(1, -1) < valid_token_lengths
            )

            _output_ids[src_index_mask] = output_ids[valid_tokens]

            invalid_response_mask = torch.arange(max_unpadded_length, device=response_lengths.device).view(
                1, -1
            ) >= response_lengths.view(-1, 1)
            _output_ids[invalid_response_mask] = self.pad_id

            output_ids = _output_ids

        output_ids = output_ids[..., :max_length].contiguous()
        output_ids = broadcast_2d_tensor_within_mp(output_ids, dtype=output_ids.dtype)

        output = {
            "response_tokens": output_ids,
            "response_lengths": response_lengths,
        }

        return output

    def free(self, force_unload=False):
        if force_unload or self.unload_engine_train:
            tensorrt_llm_run.tensorrt_llm_worker_context.decoder = None
            tensorrt_llm_run.tensorrt_llm_worker_context = tensorrt_llm_run.TensorrtLLMWorkerContext()
