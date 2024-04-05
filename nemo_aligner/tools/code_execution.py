import os
from queue import Queue
from typing import List, Dict
from threading import Thread

import numpy as np
import torch
from megatron.core import parallel_state

from nemo_aligner.tools.tool_exec import GenerationPipelineStage, GenerationItem
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.code_execution.utils import extract_code_to_execute
# from nemo_skills.inference.server.serve_trt import prepare_stop_words

code_output_template = """
<llm-code-output>
{answer}
</llm-code-output>
"""

class CodeExecutionTool(GenerationPipelineStage):
    """
    This pipeline stage executes code asynchronously via external server calls.
    max_threads controls the maximum number of external server calls
    All calls are made from TP rank 0

    All completed requests are stored in self.tmp_queue until run_batch is called,
    which will communicate the results of code execution to all TP members.
    """
    def __init__(self, tokenizer, max_threads=8):
        super().__init__(synchronous=False)

        self.tokenizer = tokenizer
        host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "localhost")
        port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "1034")
        self.sandbox = LocalSandbox(host=host, port=port)

        self.tmp_queue = Queue()
        self.workers = []
        for _ in range(max_threads):
            self.workers.append(Thread(target=self.work_fn))
            self.workers[-1].setDaemon(True)
            self.workers[-1].start()

    def enqueue(self, inputs: List[GenerationItem]):
        not_run = []
        for item in inputs:
            tokens = item.data["token_ids"]
            text = self.tokenizer.decode(tokens)
            if text.endswith("</llm-code>"):
                item.tool_data["code_exec/code_exec_string"] = text
                self.work_queue.put(item)
            else:
                not_run.append(item)

        return not_run
    
    def run_batch(self):
        """
        Performs TP communication for asynchronously completed code requests.
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()

        # src rank
        if torch.distributed.get_rank() == parallel_state.get_tensor_model_parallel_src_rank():
            completed = []
            id_and_lens = []

            while len(self.tmp_queue) > 0:
                completed.append(self.tmp_queue.get())
            completed = completed.sort(key=lambda x: x.id)
            max_len = max([len(c.data["token_ids"]) for c in completed])

            # communicates the completed tensor size
            num_complete_and_maxlen = torch.cuda.LongTensor([len(completed), max_len])
            torch.distributed.broadcast(num_complete_and_maxlen, tp_group)
            
            # constucts a token_id tensor with padding
            completed_tensor = torch.zeros((len(completed), max_len), dtype=torch.int32)
            for i, c in enumerate(completed):
                str_len = len(c.data["token_ids"])
                completed_tensor[i][:str_len] = torch.IntTensor(c.data["token_ids"])
                id_and_lens.append([c.id, str_len])

            # communicates the token tensor
            completed_tensor = completed_tensor.cuda()
            id_and_lens = torch.IntTensor(id_and_lens).cuda()
            torch.distributed.broadcast(completed_tensor, tp_group)
            torch.distributed.broadcast(id_and_lens, tp_group)
          
        # subscriber ranks
        else:
            # gets all elements in the tmp queue
            buffer = []
            while len(self.tmp_queue) > 0:
                buffer.append(self.tmp_queue.get())
                
            # recieves the shape of the complete token tensor
            num_complete_and_maxlen = torch.cuda.LongTensor([0, 0])
            torch.distributed.broadcast(num_complete_and_maxlen, tp_group)
            num_complete = num_complete_and_maxlen[0]
            max_len = num_complete_and_maxlen[1]
            
            # recieves the token tensor
            completed_tensor = torch.zeros((num_complete, max_len)).cuda()
            ids_and_lens = torch.zeros((num_complete, 2)).cuda()
            torch.distributed.broadcast(completed_tensor, tp_group)
            torch.distributed.broadcast(ids_and_lens, tp_group)
            
            # unpads and converts the token tensor to lists
            ids = ids_and_lens[:, 0].tolist()
            lens = ids_and_lens[:, 1].tolist()
            src_completed = []
            for item in buffer:
                if buffer.id in ids:
                    tensor_idx = ids.index(buffer.id)
                    item.data["token_ids"] = completed_tensor[tensor_idx][:lens[tensor_idx]].tolist()
                    src_completed.append(item)
            src_completed.sort(key=lambda x: x.id)
            assert len(src_completed) == num_complete, f"ERROR: non-src TP found {len(src_completed)} completed items compared to expected {num_complete}"
            
            # puts all incomplete (according to src rank) items back in self.tmp_queue
            for item in filter(lambda x: x.id not in ids, buffer):
                self.tmp_queue.put(item)

            completed = src_completed

        for item in completed:
            self.out_queue.put(item)

       
    def work_fn(self):
        """Performs code server requests"""
        while True:
            item = self.work_queue.get()
            text = item.tool_data["code_exec/code_exec_string"]

            if torch.distributed.get_rank() == parallel_state.get_tensor_model_parallel_src_rank():
                try:
                    code = extract_code_to_execute(text)
                    output, uuid = self.sandbox.execute_code(code)
                    results = output["result"]
                    output_text = code_output_template.format(answer=results)
                    item.data["token_ids"] = self.tokenizer.text_to_ids(text + output_text) 
                    del item.tool_data["code_exec/code_exec_string"]
                    self.tmp_queue.put(item)
                except Exception as e:
                    print("############ Code Environment Error ############")
                    print(text)
                    print(e)
            else:
                del item.tool_data["code_exec/code_exec_string"]
                self.tmp_queue.put(item)
                
    
    def get_stop_words(self):
        return prepare_stop_words(["<llm-code-output>"], self.tokenizer)
    
    def has_work(self):
        return len(self.work_queue) + len(self.tmp_queue) > 0

def prepare_stop_words(stop_words_list, tokenizer):
    # adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/b310ec675145c9ee7668592549f733df4abf1e94/tensorrt_llm/runtime/generation.py#L46
    flat_ids = []
    offsets = []
    for batch_stop_words in stop_words_list:
        item_flat_ids = []
        item_offsets = []

        for word in batch_stop_words:
            # there is a known issue in TensorRT-LLM that word ids are not unique and might change depending on
            # where in the text it appears. In our case we mainly need to stop on ids as they appear in the middle
            # of the text. The following is a workaround to get such ids that works for both <TOKEN> kind of stop
            # words as well as newlines that we commonly use. But note that it's not a universal fix, so this might
            # require refactoring if different stop words are used in the future.
            # Eventually, this needs to be fixed inside TensorRT-LLM itself.
            ids = tokenizer.text_to_ids('magic' + word)
            ids = ids[2:]  # skipping "magic"

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    stop_words = np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
    return torch.Tensor(stop_words).to(torch.int32).contiguous()

