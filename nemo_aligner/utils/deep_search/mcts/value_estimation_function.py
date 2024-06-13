from nemo_aligner.utils.trtllm.trtllm_inference import TRTLLMInference
from nemo_aligner.utils.trtllm.trtllm_fast_generation import TRTLLMFastGeneration


class ValueApproximationFunction(TRTLLMInference):

    def __init__(self, cfg, tokenizer, score_fn, terminate_fns, pad_id, add_bos_token=False):
        self.infer = TRTLLMFastGeneration(cfg, tokenizer.eos_id(), tokenizer.pad_id(), tokenizer.bos_id(), tokenizer, stop_words=['<extra_id_1>', '\x11'])
        self.score_fn = score_fn
        self.terminate_fns = terminate_fns
        self.add_bos_token = add_bos_token
        self.pad_id = pad_id

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
        self,
        inputs=None,
        action=None,
        context_ids=None,
        data_ids=None,
    ): 
        context_tokens = self.compute_context_tokens(inputs, context_ids, action, self.add_bos_token)
        out = self.infer.evaluate(context_tokens, 1024)
        result = []
        for i in range(len(out['output_ids'])):
            text = self.tokenizer.decode(out['output_ids'][i])
            value, terminate, end_properly, has_answer = self.get_value_and_terminated(text, data_ids[i], i, out['output_ids'][i])
            result.append((value, terminate, end_properly, has_answer)) 
        return result
