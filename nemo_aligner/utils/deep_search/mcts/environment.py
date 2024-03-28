import os
import re

import pandas as pd
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.utils import extract_code_to_execute
from nemo_skills.code_execution.sandbox import LocalSandbox

from nemo_aligner.utils.deep_search.mcts.mcts import Node


class Environment(object):
    def __init__(self):
        pass

    def state_transtion(self, node, past_tokens):
        """
        Given the current node and the past context tokens, it will
        1. whether we need to add more tokens to the context from envrionment
        2. if yes, add the obersation tokens to the node state and update the node action
        """
        raise NotImplementedError

class SimpleEnvironment(Environment):
    
    def state_transtion(self, node, past_tokens):
        """
        no new observation tokens are added. do nothing
        """
        pass


code_output_template = """
<llm-code-output>
{answer}
</llm-code-output>
"""

class CodeExecutionEnvironment(Environment):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # local_rank = os.getenv("local_rank", "0")
        host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "localhost")
        port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "1034")
        self.sandbox = LocalSandbox(host=host, port=port)

    def state_transtion(self, node, past_tokens):
        """
        Given the current node and the past context tokens, it will
        1. whether we need to add more tokens to the context from envrionment
        2. if yes, add the obersation tokens to the node state and update the node action
        """
        # combine the context tokens with the current node state
        past_tokens = past_tokens + node.state
        past_text = self.tokenizer.decode(past_tokens)
        if past_text.endswith("</llm-code>"):
            try:
                code = extract_code_to_execute(past_text)
                output, uuid = self.sandbox.execute_code(code)
                results = output['result']
                output_text = code_output_template.format(answer=results)
                output_tokens = self.tokenizer.encode(past_text + output_text)
                output_tokens = output_tokens[len(past_tokens):]
                # modify the node action
                # it merges the node's state with the output tokens
                # and set the last token as the new action
                node.action = node.state + output_tokens
                node.state = node.state + output_tokens
            except Exception as e:
                print("############ Code Environment Error ############")
                print(past_text)
                print(e)
        else:
            # do nothing
            pass