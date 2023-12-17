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
        self.max_context_length = 4096 - self.max_generation_length

        from mpi4py import MPI 
        rank = MPI.COMM_WORLD.Get_rank()
        worldsize = MPI.COMM_WORLD.Get_size()
        print(f"Loaded mpi lib {MPI.__file__} successfully; rank={rank}, worldsize={worldsize}")

        from nemo.export import TensorRTLLM
        from nemo.export.trt_llm.nemo_utils import WrappedNemoSentencePiece
        from nemo.export.trt_llm.tensorrt_llm_run import forward as trtllm_forward
        import tensorrt_llm

        globals()["TensorRTLLM"] = TensorRTLLM
        globals()["trtllm_forward"] = trtllm_forward
        print(f"TRTLLM MPI settings {tensorrt_llm.mpi_world_size()} {tensorrt_llm.mpi_rank()}")

        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self.trt_llm_exporter.tokenizer = WrappedNemoSentencePiece(self.tokenizer)

        stop_id = self.tokenizer.text_to_ids("<extra_id_1>")[-1]
        eos_id = self.tokenizer.eos_id
        stop_id = torch.IntTensor([[stop_id, eos_id, -1], [1, 2, -1]])
        self.stop_id = stop_id.unsqueeze(0).repeat(self.cfg.ppo.get('rollout_micro_batch_size'),1,1).cuda()

    def refit(self, model):
        self.model = model
        self.trt_llm_exporter.build(
            self.model, 
            self.cfg, 
            max_input_len=self.max_context_length,
            max_output_len=self.max_generation_length,
            max_batch_size=self.cfg.ppo.get('rollout_micro_batch_size'),
            refit=True,
            model_type="llama")


    def generate(self, inputs, length_params, sampling_params, stopid=None):
        self._length_params = length_params
        self._sampling_params = sampling_params
        prompt_tokens, prompt_lengths = inputs

        mbs = prompt_lengths.shape[0]
        prompt_tokens = prompt_tokens.to(torch.int32)
        prompt_tokens = [sample[0:prompt_lengths[idx]] for idx, sample in enumerate(prompt_tokens)]

        if stopid is not None:
            self.stop_id = stopid

        output_ids = self.forward(
            prompt_tokens, 
            self._length_params["max_length"], 
            self._sampling_params["top_k"], 
            self._sampling_params["top_p"], 
            self._sampling_params["temperature"],
            self.stop_id)
        if output_ids is not None:
            mbs = output_ids.shape[0]
            if mbs == 1:
                output_ids = output_ids.view([1,output_ids.shape[-1]])
            else:
                output_ids = output_ids.squeeze()
            output_ids = output_ids.to(torch.int64)
        group = parallel_state.get_model_parallel_group()
        if torch.distributed.get_world_size(group) > 1:
            output_ids = broadcast_2d_tensor(
                output_ids, get_model_parallel_src_rank(), group, dtype=output_ids.dtype)

        sentences = [self.tokenizer.ids_to_text(output.tolist()) for output in output_ids]
        output_ids = torch.Tensor.tolist(output_ids)

        output = {
            "token_ids" : output_ids,
            "sentences" : sentences,
        }
        
        return output

    def forward(
        self,
        input_tensors: List[torch.IntTensor],
        max_output_len: int,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_ids=None,
    ):
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        sampling_config = tensorrt_llm_worker_context.sampling_config

        pad_id = sampling_config.pad_id
        line_encoded = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tensors, dtype=torch.int32), pad_id
        ).cuda()

        input_lengths = [t.shape[0] for t in input_tensors]
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()
        batch_size = len(input_tensors)
        max_length = max(input_lengths)

        decoder.setup(
            batch_size, 
            max_context_length=max_length, 
            max_new_tokens=self.max_generation_length,
            max_kv_cache_length=self.max_generation_length)
        
        output_ids = decoder.decode(
            input_ids=line_encoded,
            context_lengths=input_lengths,
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
