class SearchStopCriteria:
    def __init__(self, score_fn, terminate_fns, threshold=1.0):
        self.score_fn = score_fn
        self.terminate_fns = terminate_fns
        self.evaluation_cache = {}
        self.threshold = threshold
        self.terminate = {}
        self.max_value = {}

    def get_value_and_terminated(self, text, data_id, depth, tokens):
        if data_id in self.evaluation_cache and text in self.evaluation_cache[data_id]:
            return self.evaluation_cache[data_id][text][0]
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
        result = value, terminate, end_properly, has_answer
        if data_id not in self.evaluation_cache:
            self.evaluation_cache[data_id] = {}
        if terminate:
            # only save the terminated results
            self.evaluation_cache[data_id][text] = (result, tokens)
        if value >= self.threshold:
            self.terminate[data_id] = True
        if data_id not in self.max_value:
            self.max_value[data_id] = value
        else:
            self.max_value[data_id] = max(self.max_value[data_id], value)
        return result

    def reset(self):
        self.terminate = {}
        self.evaluation_cache = {}
        self.max_value = {}

    def __str__(self):
        return self.__class__.__name__
