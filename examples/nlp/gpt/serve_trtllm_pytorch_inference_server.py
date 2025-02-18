import argparse
import os
from flask import Flask, request, jsonify
import multiprocessing as mp
import torch

app = Flask(__name__)

# Global variable to hold the inference server instance
inference_server = None
import torch
import gc
import subprocess

import threading
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig


# Global lock to guard all API accesses
api_lock = threading.Lock()

class TRTLLMPytorchInferenceServer:
    def __init__(self) -> None:
        self.running = False
        self.llm = None
        self.max_seq_len = None
        self.end_id = None
        self.pad_id = None
    
    def clean_cuda_memory(self):
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

    def create_shared_dict_file(self, shared_tensor_dict):
        """
        Creates an in-memory file object containing the JSON representation
        of the metadata dictionary. On Linux, it uses os.memfd_create to get
        a real file descriptor that lives in memory only; on other platforms,
        it falls back to io.BytesIO.
        
        Args:
            shared_tensor_dict (SharedCPUMemoryTensorDict): The instance whose
            metadata we want to share.
            
        Returns:
            A file-like object, opened for reading in binary mode.
        """
        import json
        import io
        from tensor_comms.shared_tensors import SharedCPUMemoryTensorDict
        shared_cpu_state_dict = SharedCPUMemoryTensorDict(communicable_metadata=shared_tensor_dict)

        metadata = shared_cpu_state_dict.get_metadata_dict()
        json_bytes = json.dumps(metadata).encode('utf-8')
        
        try:
            # This creates an anonymous in-memory file (Linux only).
            fd = os.memfd_create("shared_tensor_metadata", flags=os.MFD_CLOEXEC)
            os.write(fd, json_bytes)
            os.lseek(fd, 0, os.SEEK_SET)
            file_obj = os.fdopen(fd, 'rb')
        except AttributeError:
            # If os.memfd_create is not available, fallback to a BytesIO object.
            file_obj = io.BytesIO(json_bytes)
        
        return file_obj

    def get_gpu_memory_usage(self):
        command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_used = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1]
        for idx, mem in enumerate(memory_used):
            print(f"GPU {idx} using {mem} GB", flush=True)


    def start(self, path, tp, state_dict):
        self.get_gpu_memory_usage()

        if self.llm is None:
            print(f"starting llm server")
            pytorch_config = PyTorchConfig(
                use_cuda_graph=False,
                use_rl_temporary_decoder=True,
                enable_iter_perf_stats=False,
                # attn_backend = 'VANILLA',
            )

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            self.max_seq_len = tokenizer.model_max_length
            self.end_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            print(f"Max seq len of model is {self.max_seq_len}")
            print(f"End id of model is {self.end_id}")
            print(f"Pad id of model is {self.pad_id}")

            self.llm = LLM(model=path, tensor_parallel_size=tp, pytorch_backend_config=pytorch_config, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7, enable_block_reuse=True))
            self.llm.free_gpu_resources()
            self.running = True
        #else:
        
        file_obj = self.create_shared_dict_file(state_dict)
        self.llm.load_model(file_obj)
        print(f"TRTLLM Pytorch inference server started.", flush=True)
        self.get_gpu_memory_usage()

    def refit(self, path, state_dict, test_dict):
        file_obj = self.create_shared_dict_file(state_dict)
        self.llm.load_model(file_obj)
        print(f"TRTLLM Pytorch inference server refitted.", flush=True)
        self.get_gpu_memory_usage()

    def shutdown(self):
        self.get_gpu_memory_usage()
        self.llm.free_gpu_resources()
        print("TRTLLM Pytorch gpu resources freed.", flush=True)
        # print the current memory usage for all GPUs
        self.get_gpu_memory_usage()

    def generate(self, batch_tokens, sampling_params):
        from tensorrt_llm import SamplingParams
        from tensorrt_llm.inputs.data import TokensPrompt

        self.get_gpu_memory_usage()


        assert self.max_seq_len is not None, "Max seq len is not set"
        assert self.end_id is not None, "End id is not set"
        sampling_params = SamplingParams(
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_tokens=sampling_params["max_tokens"], #self.max_seq_len,
            detokenize=False,
            # return_log_probs=True,
            return_generation_logits=True,
            end_id=self.end_id,
            pad_id=self.pad_id,
        )

        prompt_tokens = [TokensPrompt(prompt_token_ids=tok_seq) for tok_seq in batch_tokens]
        print(len(prompt_tokens), flush=True)
        outputs = self.llm.generate(prompt_tokens, sampling_params, use_tqdm=True)
        logprobs = []
        out_tokens = []

        # def compute_logprobs(generation_logits, token_ids):
        #     log_probs = torch.log_softmax(generation_logits, dim=-1)  # [seq_len, vocab_size]
        #     if isinstance(token_ids, list):
        #         token_ids = torch.tensor(token_ids, device=log_probs.device)
        #     seq_indices = torch.arange(len(token_ids), device=log_probs.device)
        #     selected_logprobs = log_probs[seq_indices, token_ids]  # [seq_len]    
        #     return selected_logprobs

        for output in outputs:
            # print(output.outputs[0].token_ids, flush=True)
            #lps = output.outputs[0].logprobs
            # generation logits
            generation_logits = output.outputs[0].generation_logits
            # logprobs from generation logits
            # lps = compute_logprobs(generation_logits, output.outputs[0].token_ids)
            lps = generation_logits.tolist()
            out_toks = output.outputs[0].token_ids
            logprobs.append(lps)
            out_tokens.append(out_toks)

        self.get_gpu_memory_usage()

        return out_tokens, logprobs

@app.route('/start', methods=['POST'])
def start_server():
    global inference_server
    with api_lock:
        print(f"start call", flush=True)
        data = request.get_json()

        if inference_server is None:
            print(f"First time start code path", flush=True)
            inference_server = TRTLLMPytorchInferenceServer()
            inference_server.start(data["checkpoint_path"], data["tp"], data["test_state_dict"])
            return jsonify({"status": "started"}), 200
        else:
            print(f"Refit code path", flush=True)
            inference_server.refit(data["checkpoint_path"], data["state_dict"], data["test_dict"])
            return jsonify({"status": "started"}), 200

@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    global inference_server
    with api_lock:
        print(f"shutdown call", flush=True)
        inference_server.shutdown()
        return jsonify({"status": "shutdown complete"}), 200

@app.route('/generate', methods=['POST'])
def generate():
    global inference_server
    with api_lock:
        print(f"Generate call", flush=True)
        
        if inference_server is None or not inference_server.running:
            return jsonify({"error": "inference server not running"}), 400


        import time
        start_time = time.time()
        data = request.get_json()
        tokens = data["tokens"]
        sampling_params = data["sampling_params"]
        end_time = time.time()
        print(f"Request get_json time: {end_time - start_time} seconds")
        if not isinstance(tokens, list):
            return jsonify({"error": "Expected a list of lists of tokens"}), 400

        # Validate that each element in the list is itself a list
        for i, item in enumerate(tokens):
            if not isinstance(item, list):
                return jsonify({"error": f"Element at index {i} is not a list"}), 400

        generations, logprobs = inference_server.generate(tokens, sampling_params)
        response = {
            "response_tokens": generations,
            "response_logprobs": logprobs
        }

        import time
        start_time = time.time()
        response = jsonify(response), 200
        end_time = time.time()
        print(f"Response jsonify time: {end_time - start_time} seconds", flush=True)
        return response

if __name__ == '__main__':
    # Run the Flask app
    mp.set_start_method("spawn", force=True)
  
    parser = argparse.ArgumentParser(description='Flask server for serving TRTLLM Pytorch inference (one TP group)')

    parser.add_argument(
        '--port', 
        type=int, 
        required=True, 
        help='Port number to use (must be an integer)'
    )
    args = parser.parse_args()
    port = args.port
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")

    def print_device_properties():
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Iterate over all available devices
        for i in range(device_count):
            print(f"\nDevice {i}: {torch.cuda.get_device_properties(i)}")
    # Run the function
    print_device_properties()

    app.run(host='0.0.0.0', port=port)
