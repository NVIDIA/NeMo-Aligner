import secrets

import tensorrt_llm
import torch

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm import tensorrt_llm_run
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import build_tokenizer
from nemo.utils import logging
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp


class GPTGenerateTRTLLM:
    def __init__(
        self,
        model_cfg,
        max_generation_length=1024,
        max_input_len=1024,
        max_input_tokens=4096,
        generation_batch_size=4,
        unload_engine_train=False,
        trt_model_type="GPTForCausalLM",
        end_strings=None,
        reshard_model=False,
        sample_temperature=None,
        sample_top_k=None,
        sample_top_p=None,
        repetition_penalty=1.0,
        use_greedy=False,
        tokenizer=None,
        trt_model_dir="/tmp/trt_llm_model",
        seed=None,
    ):
        if use_greedy and sample_top_k != 1:
            if sample_top_k != 1:
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
        self.pad_id = tokenizer.eos_id

        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self._trtllm_model_compiled = False

        rng_generator = torch.Generator(device="cpu")
        seed = secrets.randbits(32) if seed is None else seed

        if seed is not None:
            rng_generator.manual_seed(seed)

        self.rng_generator = rng_generator

        end_strings = list(end_strings)
        end_strings = [[",".join(end_strings)] for _ in range(self.generation_batch_size)]
        stop_list = tensorrt_llm_run.to_word_list_format(
            end_strings, build_tokenizer(self.tokenizer), ref_str="green tea icecream"
        )
        stop_list = torch.from_numpy(stop_list).cuda().contiguous()

        self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=tokenizer.eos_id,
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
        else:
            self.trt_llm_exporter.refit(model, self.model_cfg)

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

        # TRTLLM with PP erroneously inserts padding so have to remove it here
        # TRTLLM only produces valid outputs on the source rank, so we can only process it here
        # and rely on the aligner broadcast to get it to the other ranks. Miraculously, the length
        # is still correct on the non src ranks
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_model_parallel_src_rank() == torch.distributed.get_rank()
        ):
            valid_tokens = output_ids != self.pad_id
            # we can't just natively use the response length here
            # because there are cases where the model generates
            # stop strings after it has stopped. so we need to
            # be slightly inefficient and then remove the excess later on
            valid_token_lengths = valid_tokens.sum(-1, keepdims=True)
            max_unpadded_length = valid_token_lengths.max()

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
