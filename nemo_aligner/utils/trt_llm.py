import tensorrt_llm
import torch

from nemo.export import TensorRTLLM
from nemo.export.trt_llm.nemo.nemo_ckpt_convert import build_tokenizer
from nemo.export.trt_llm.nemo_utils import to_word_list_format
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp

from nemo.utils import logging


class GPTGenerateTRTLLM:
    def __init__(
        self, cfg, tokenizer, trt_model_dir="/tmp/trt_llm_model",
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_generation_length = self.cfg.ppo.length_params.get("max_length", 1024)
        self.max_input_len = self.cfg.ppo.trt_llm.get("max_input_len", 1024)
        self.max_input_tokens = self.cfg.ppo.trt_llm.get("max_input_tokens", 4096)
        self.generation_batch_size = self.cfg.ppo.get("rollout_micro_batch_size", 4)
        self.unload_engine_train = self.cfg.ppo.trt_llm.get("unload_engine_train", False)
        self.trt_model_type = self.cfg.ppo.trt_llm.get("model_type", "LLaMAForCausalLM")

        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self._trtllm_model_compiled = False

        # TODO: Move this logic to nemo.export after TRTLLM0.9 support
        end_strings = list(self.cfg.ppo.sampling_params.get("end_strings"))
        end_strings = [[",".join(end_strings)] for _ in range(self.generation_batch_size)]
        stop_list = to_word_list_format(end_strings, build_tokenizer(self.tokenizer), ref_str="green tea icecream")
        stop_list = torch.from_numpy(stop_list).cuda().contiguous()

        self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=tokenizer.eos_id,
            pad_id=tokenizer.eos_id,  # TODO
            temperature=self.cfg.ppo.sampling_params.get("temperature"),
            top_k=self.cfg.ppo.sampling_params.get("top_k"),
            top_p=self.cfg.ppo.sampling_params.get("top_p"),
            max_new_tokens=self.max_generation_length,
            stop_words_list=stop_list,
            return_dict=True,
            output_sequence_lengths=True,
        )

    def refit(self, model):
        if not self._trtllm_model_compiled:
            self.trt_llm_exporter.build(
                nemo_model=model,
                nemo_model_config=self.cfg,
                trt_model_type=self.trt_model_type,
                tokenizer=self.tokenizer,
                max_input_len=self.max_input_len,
                max_input_tokens=self.max_input_tokens,
                max_output_len=self.max_generation_length,
                max_batch_size=self.generation_batch_size,
                reshard_model=parallel_state.is_trt_llm_reshard(),
            )
            self._trtllm_model_compiled = True
        else:
            self.trt_llm_exporter.refit(
                nemo_model=model, nemo_model_config=self.cfg,
            )

    def generate(self, inputs):
        prompt_tokens, prompt_lengths = inputs

        batch_input_ids = []
        for idx in range(prompt_tokens.shape[0]):
            batch_input_ids.append(prompt_tokens[idx][:prompt_lengths[idx]].cpu())

        output_dict = self.trt_llm_exporter.model_runner.generate(
            batch_input_ids=batch_input_ids, 
            sampling_config=self.sampling_config, 
            streaming=False
        )

        # remove beam dim from output_ids: [mbs, beam_dim, sequence len]
        output_ids = torch.squeeze(output_dict["output_ids"], dim=1).long()
        resp_lens = torch.squeeze(output_dict["sequence_lengths"], dim=1).long()

        #TRTLLM with PP erroneously inserts padding so have to remove it here
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            max_prompt_len = prompt_lengths.max().item()
            _output_ids = torch.full_like(input=output_ids, fill_value=self.tokenizer.eos_id)
            for idx in range(prompt_tokens.shape[0]):
                gen_len = (resp_lens[idx]-prompt_lengths[idx]).item()
                response = output_ids[idx, max_prompt_len:max_prompt_len+gen_len]
                prompt_response = torch.cat((prompt_tokens[idx][:prompt_lengths[idx]],response))
                _output_ids[idx, :prompt_response.size(0)] = prompt_response
            output_ids = _output_ids

        max_len = (prompt_lengths + resp_lens).max().item()
        output_ids = output_ids[..., :max_len]
        output_ids = output_ids.contiguous()

        # broadcast output to all PP ranks
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            output_ids = broadcast_2d_tensor_within_mp(output_ids, dtype=output_ids.dtype)

        max_id = torch.max(output_ids).item()
        if max_id > self.tokenizer.vocab_size:
            logging.warning(f"Generated token id greater than vocab size! \
                Generated token: {max_id}")
            output_ids = torch.clamp(
                        output_ids, max=self.tokenizer.vocab_size - 1)

        min_id = torch.min(output_ids).item()
        if min_id < 0:
            logging.warning(f"Generated token id less than vocab size! \
                Generated token: {min_id}")
            output_ids = torch.clamp(
                        output_ids, max=self.tokenizer.vocab_size - 1)


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
