import os

import torch

from nemo_aligner.utils.deep_search.serve_trt_for_treesearch import TensorRTLLMModelClient
from nemo_aligner.utils.trtllm.trtllm_fast_generation import TRTLLMFastGeneration
from nemo_aligner.utils.trtllm.trtllm_inference import TRTLLMInference


class ValueApproximationFunction(TRTLLMInference):
    def __init__(self, cfg, tokenizer, score_fn, terminate_fns, pad_id, add_bos_token=False):

        host = os.getenv("TRTLLM_GEN_HOST", "localhost")
        port = os.getenv("TRTLLM_GEN_PORT", "5000")
        self.infer = TensorRTLLMModelClient(host=host, port=port)
        self.score_fn = score_fn
        self.terminate_fns = terminate_fns
        self.add_bos_token = add_bos_token
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.max_depth_to_explore = 1024
        self.value_cache = {}

    def get_value_and_terminated(self, text, data_id, depth, tokens):
        terminate = False
        for fun in self.terminate_fns:
            if fun(text, depth, tokens):
                terminate = True
                break

        value = 0.0
        if terminate:
            value = self.score_fn.score(text, data_id)
        # check if the text ends properly
        end_properly = False
        for fun in self.terminate_fns:
            if fun.ends_by_end_strings(text, tokens):
                end_properly = True
                break
        has_answer = False
        for fun in self.terminate_fns:
            if fun.has_answer(text):
                has_answer = True
                break
        return value, terminate, end_properly, has_answer

    def __call__(
        self, inputs=None, action=None, context_ids=None, data_ids=None,
    ):
        context_tokens = self.compute_context_tokens(inputs, context_ids, action, self.add_bos_token)
        context_tokens = [c.tolist() for c in context_tokens]

        results = []
        input_ids = []
        for context in context_tokens:
            if tuple(context) in self.value_cache:
                results.append(self.value_cache[tuple(context)])
            else:
                results.append(None)
                input_ids.append(context)

        infer_results = []
        if len(input_ids) != 0:
            context_lengths = [len(c) for c in input_ids]
            out = self.infer.generate(
                batch_input_ids=input_ids,
                input_lengths=context_lengths,
                tokens_to_generate=self.max_depth_to_explore,
                temperature=0.0,
                top_p=0.95,
                top_k=1,
                repetition_penalty=1.0,
                random_seed=0,
                stop_phrases=["<extra_id_1>"],
                remove_stop_phrases=False,
            )

            for i in range(len(out)):
                # text = out[i]['generation']
                generation_ids = out[i]["output_ids"]
                full_ids = input_ids[i] + generation_ids
                text = self.tokenizer.decode(full_ids)
                value, terminate, end_properly, has_answer = self.get_value_and_terminated(
                    text, data_ids[i], i, full_ids
                )
                value = torch.tensor(value)
                self.value_cache[tuple(input_ids[i])] = value
                # cache all subsequent values
                for j in range(len(generation_ids)):
                    self.value_cache[tuple(input_ids[i] + generation_ids[:j])] = value
                infer_results.append(value)
        # replace None with the results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = infer_results.pop(0)
        return results
