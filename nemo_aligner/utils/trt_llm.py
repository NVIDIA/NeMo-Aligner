from megatron.core import parallel_state
import torch 

from nemo_aligner.utils.distributed import broadcast_2d_tensor
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from typing import List



class GPTGenerateTRTLLM():
    def __init__(self, cfg, tokenizer, trt_model_dir="/tmp/trt_llm_model", ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_generation_length = self.cfg.ppo.length_params.get('max_length')
        self.generation_batch_size = self.cfg.ppo.get('rollout_micro_batch_size')
        self.max_context_length = 2048
        self._trtllm_model_compiled = False

        self._import_tensorrt_llm()
        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self.stop_words = self._create_stop_words_()

    def _import_tensorrt_llm(self):        
        from mpi4py import MPI 
        from nemo.export import TensorRTLLM
        from nemo.export.trt_llm.tensorrt_llm_run import forward as trtllm_forward
        import tensorrt_llm

        globals()["TensorRTLLM"] = TensorRTLLM
        globals()["trtllm_forward"] = trtllm_forward
        
    def _create_stop_words_(self):
        # stop_id = self.tokenizer.text_to_ids("<extra_id_1>")
        stop_id = [29966, 17833, 29918, 333, 29918, 29896, 29958]
        eos_id = self.tokenizer.eos_id
        stop_strings = [stop_id]
        stop_tokens = [[eos_id]]

        stop_words = [[],[]]
        for w in (stop_strings+stop_tokens):
            stop_words[0] += w
            stop_words[1].append(len(stop_words[0]))
        stop_words[1] += [-1] * (len(stop_words[0]) - len(stop_words[1]))

        stop_words = torch.IntTensor(stop_words).cuda()
        return stop_words.unsqueeze(0).repeat(self.generation_batch_size,1,1)

    def refit(self, model):
        if not self._trtllm_model_compiled:
            self.trt_llm_exporter.build(
                nemo_model = model, 
                nemo_model_config = self.cfg, 
                tokenizer = self.tokenizer,
                max_input_token=self.max_context_length,
                max_output_token=self.max_generation_length,
                max_batch_size=self.cfg.ppo.get('rollout_micro_batch_size'),
                use_refit=True,
                model_type="llama")
            self._trtllm_model_compiled = True
        else:
            self.trt_llm_exporter.refit(
                nemo_model = model, 
                nemo_model_config = self.cfg, 
            )


    def generate(self, inputs, length_params, sampling_params, stop_words=None):
        self._length_params = length_params
        self._sampling_params = sampling_params

        if stop_words is None:
            stop_words = self.stop_words

        output_ids = self.forward(inputs, stop_words)
        
        if output_ids is not None:
            mbs = output_ids.shape[0]
            if mbs == 1:
                output_ids = output_ids.view([1,output_ids.shape[-1]])
            else:
                output_ids = output_ids.squeeze()
            output_ids = output_ids.to(torch.int64)

        group = parallel_state.get_tensor_model_parallel_group()
        if torch.distributed.get_world_size(group) > 1:
            output_ids = broadcast_2d_tensor(
                output_ids, parallel_state.get_tensor_model_parallel_src_rank(), group, dtype=output_ids.dtype)

        toss_out = torch.tensor([1], dtype=output_ids.dtype, device=output_ids.device)
        ii = len(output_ids[0]) - 1
        while ii >= 0 and output_ids[0, ii] == self.tokenizer.eos_id:
            ii -= 1
        trimmed_output_ids = output_ids[:, :ii + 1]

        if inputs[1] + self.max_generation_length == trimmed_output_ids.shape[-1]: 
            print("&&&&&& NOT FINISHED &&&&&&")
            toss_out = torch.tensor([0], dtype=output_ids.dtype, device=output_ids.device)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&")

        sentences = [self.tokenizer.ids_to_text(output.tolist()) for output in output_ids]
        output_ids = torch.Tensor.tolist(output_ids)

        output = {
            "token_ids" : output_ids,
            "sentences" : sentences,
        }
        
        return output, toss_out

    def forward(self, inputs, stop_ids):
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        sampling_config = tensorrt_llm_worker_context.sampling_config

        prompt_tokens, prompt_lengths = inputs
        prompt_tokens = prompt_tokens[:, :max(prompt_lengths)]
        prompt_tokens = prompt_tokens.to(torch.int32).cuda()
        prompt_lengths = prompt_lengths.to(torch.int32).cuda()

        decoder.setup(
            batch_size=self.generation_batch_size, 
            max_context_length=int(max(prompt_lengths)), 
            max_new_tokens=self.max_generation_length,
            max_attention_window_size=2084
        )

        output_ids = decoder.decode(
            input_ids=prompt_tokens,
            context_lengths=prompt_lengths,
            sampling_config=sampling_config,
            prompt_embedding_table=None,
            tasks=None,
            prompt_vocab_size=None,
            stop_words_list=stop_ids,
        )
        return output_ids

    def free(self):
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        del tensorrt_llm_worker_context.decoder
        torch.cuda.empty_cache()
