import tensorrt_llm
import torch

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import build_tokenizer
from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context, to_word_list_format
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
        use_greedy=False,
        tokenizer=None,
        trt_model_dir="/tmp/trt_llm_model",
    ):
        if use_greedy and sample_top_k != 1 and sample_temperature != 0.0:
            if sample_top_k != 1:
                logging.warning(f"'use_greedy=True' overrides {sample_top_k=} to 1")
            if sample_temperature != 0.0:
                logging.warning(f"'use_greedy=True' overrides {sample_temperature=} to 0.0")
            sample_top_k = 1
            sample_temperature = 0.0

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

        end_strings = list(end_strings)
        end_strings = [[",".join(end_strings)] for _ in range(self.generation_batch_size)]
        stop_list = to_word_list_format(end_strings, build_tokenizer(self.tokenizer), ref_str="green tea icecream")
        stop_list = torch.from_numpy(stop_list).cuda().contiguous()

        self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=tokenizer.eos_id,
            pad_id=tokenizer.eos_id,
            temperature=sample_temperature,
            top_k=sample_top_k,
            top_p=sample_top_p,
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

        output_dict = tensorrt_llm_worker_context.decoder.generate(
            batch_input_ids=batch_input_ids, sampling_config=self.sampling_config, streaming=False
        )

        # remove beam dim from output_ids: [mbs, beam_dim, sequence len]
        output_ids = torch.squeeze(output_dict["output_ids"], dim=1).long()
        resp_lens = torch.squeeze(output_dict["sequence_lengths"], dim=1).long()

        # TRTLLM with PP erroneously inserts padding so have to remove it here
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            max_prompt_len = prompt_lengths.max().item()
            _output_ids = torch.full_like(input=output_ids, fill_value=self.tokenizer.eos_id)
            for idx in range(prompt_tokens.shape[0]):
                gen_len = (resp_lens[idx] - prompt_lengths[idx]).item()
                response = output_ids[idx, max_prompt_len : max_prompt_len + gen_len]
                prompt_response = torch.cat((prompt_tokens[idx][: prompt_lengths[idx]], response))
                _output_ids[idx, : prompt_response.size(0)] = prompt_response
            output_ids = _output_ids

        max_len = (prompt_lengths + resp_lens).max().item()
        output_ids = output_ids[..., :max_len]
        output_ids = output_ids.contiguous()
        output_ids = broadcast_2d_tensor_within_mp(output_ids, dtype=output_ids.dtype)

        assert (0 <= output_ids).all(), "TRT-LLM generated tokens that are less than 0"
        assert (
            self.tokenizer.vocab_size > output_ids
        ).all(), "TRT-LLM generated tokens that are greater than the vocab size"

        sentences = [self.tokenizer.ids_to_text(output) for output in output_ids.tolist()]
        output = {
            "response_tokens": output_ids,
            "response_lengths": resp_lens,
            "sentences": sentences,
        }

        return output

    def free(self):
        if not self.unload_engine_train:
            return
        del self.trt_llm_exporter.model_runner.session
