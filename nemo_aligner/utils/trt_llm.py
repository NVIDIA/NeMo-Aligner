import tensorrt_llm
import torch

from nemo.export import TensorRTLLM
from nemo.export.trt_llm.nemo.nemo_ckpt_convert import build_tokenizer
from nemo.export.trt_llm.nemo_utils import to_word_list_format
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp


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
                reshard_model=self.cfg.ppo.trt_llm.get("reshard", True),
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
            batch_input_ids.append(prompt_tokens[idx][0 : prompt_lengths[idx]].cpu())

        output_dict = self.trt_llm_exporter.model_runner.generate(
            batch_input_ids=batch_input_ids, sampling_config=self.sampling_config, streaming=False
        )

        # remove beam dim from output_ids: [mbs, beam_dim, sequence len]
        output_ids = torch.squeeze(output_dict["output_ids"], dim=1).long()
        resp_lens = torch.squeeze(output_dict["sequence_lengths"], dim=1).long()

        # broadcast output to all PP ranks
        if not self.trt_llm_exporter.reshard_model:
            output_ids = broadcast_2d_tensor_within_mp(output_ids, dtype=output_ids.dtype)

        max_len = (prompt_lengths + resp_lens).max().item()
        output_ids = output_ids[..., :max_len]

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
