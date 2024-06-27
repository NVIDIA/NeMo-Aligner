import os

import torch

from nemo_aligner.utils.deep_search.serve_trt_for_treesearch import TensorRTLLMModelClient
from nemo_aligner.utils.trtllm.trtllm_fast_generation import TRTLLMFastGeneration
from nemo_aligner.utils.trtllm.trtllm_inference import TRTLLMInference


class ValueApproximationFunction(TRTLLMInference):
    def __init__(self, tokenizer, stop_criteria, pad_id, add_bos_token=False):

        host = os.getenv("TRTLLM_GEN_HOST", "localhost")
        port = os.getenv("TRTLLM_GEN_PORT", "5000")
        self.infer = TensorRTLLMModelClient(host=host, port=port)
        self.stop_criteria = stop_criteria
        self.add_bos_token = add_bos_token
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.max_depth_to_explore = 1024
        self.value_cache = {}

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

            try:
                for i in range(len(out)):
                    # text = out[i]['generation']
                    generation_ids = out[i]["output_ids"]
                    full_ids = input_ids[i] + generation_ids
                    text = self.tokenizer.decode(full_ids)
                    value, terminate, end_properly, has_answer = self.stop_criteria.get_value_and_terminated(
                        text, data_ids[i], i, full_ids
                    )
                    value = torch.tensor(value)
                    self.value_cache[tuple(input_ids[i])] = value
                    # cache all subsequent values
                    for j in range(len(generation_ids)):
                        self.value_cache[tuple(input_ids[i] + generation_ids[:j])] = value
                    infer_results.append(value)
            except Exception as e:
                print(f"Error in value estimation: {e}")
                print(out)
                print(len(out))
                print(out[0])
                print(type(out))
                print(type(out[0]))
                import traceback

                traceback.print_exc()
                raise e
        # replace None with the results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = infer_results.pop(0)
        return results
