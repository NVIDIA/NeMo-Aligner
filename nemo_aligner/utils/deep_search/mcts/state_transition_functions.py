import os

import numpy as np
from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.code_execution.utils import extract_code_to_execute
from pytriton.client import ModelClient

from nemo_aligner.utils.deep_search.search_callables import encode_context_data
from nemo_aligner.utils.deep_search.text_gen_utils import dp_search

code_output_template = """
<llm-code-output>
{answer}
</llm-code-output>
"""


class StateTransitionFunction:
    def __call__(self, sentences=None, action=None, context_ids=None, session_info=None):
        raise NotImplementedError("StateTransitionFunction must be implemented by the subclass.")


class LocalStateTransitionFunction(StateTransitionFunction):
    def __init__(self, model, top_k, max_depth, add_bos_token, **strategy_args):
        self.model = model
        self.top_k = top_k
        self.max_depth = max_depth
        self.add_bos_token = add_bos_token
        self.strategy_args = strategy_args

    def __call__(self, sentences=None, actions=None, context_ids=None, session_info=None):
        output = dp_search(
            self.model,
            inputs=sentences,
            action=actions,
            context_ids=context_ids,
            session_info=session_info,
            tokens_to_generate=self.max_depth,  # max search depth
            top_k=self.top_k,
            add_bos_token=self.add_bos_token,
            **self.strategy_args,
        )
        threshold = 0.01  # min probability threshold
        probablities = output["policy"]
        actions = output["action"]
        update_probablities = []
        update_actions = []
        for prob, one_actions in zip(probablities, actions):
            selected = prob >= threshold
            if sum(selected) > 0:
                # not empty
                select_prob = prob[selected]
                select_action = one_actions[selected].tolist()
                update_probablities.append(select_prob)
                update_actions.append(select_action)
            else:
                # if all the probablities are less than the threshold
                # use all the probablities
                update_probablities.append(prob)
                update_actions.append(one_actions.tolist())
        output["policy"] = update_probablities
        output["action"] = update_actions
        return output


class EnvironmentStateTransitionFunction(LocalStateTransitionFunction):
    def interact_with_environment(self, action, past_text, past_tokens):
        """base on the context text/tokens, decide whether need to augment the input action with environment tokens

        Args:
            action (int): the next action to take
            past_text (str):  context text
            past_tokens (List[int]): context tokens

        Returns:
            Union[List[int], int]: updated actions or original action
        """
        raise NotImplementedError("EnvironmentStateTransitionFunction must be implemented by the subclass.")

    def __call__(self, sentences=None, actions=None, context_ids=None, session_info=None):
        result_dict = super().__call__(sentences, actions, context_ids, session_info)
        next_actions = result_dict["action"]
        # based on the current context ids state and the action taken
        # It will transite to the next top_k states
        # the following will decide whether to interact with the code environment
        # and add the output to the node state
        if actions is None:
            actions = [[]] * len(context_ids)
        else:
            # only need the action that is not the pad token
            pad_id = self.model.tokenizer.pad_id
            actions = [action[action != pad_id].tolist() for action in actions]
        update_actions = []
        for next_action, context_id_tuple, action_taken in zip(next_actions, context_ids, actions):
            new_actions = []
            actions_list = next_action.tolist()
            for i, action in enumerate(actions_list):
                # combine the context tokens with the current node state
                past_tokens = list(context_id_tuple) + action_taken + [action]
                past_text = self.model.tokenizer.ids_to_text(past_tokens)
                new_action = self.interact_with_environment(action, past_text, past_tokens)
                new_actions.append(new_action)
            update_actions.append(new_actions)
        result_dict["action"] = update_actions
        return result_dict


class MathtoolLocalStateTransitionFunction(EnvironmentStateTransitionFunction):
    def __init__(self, model, top_k, max_depth, add_bos_token, **strategy_args):
        super().__init__(model, top_k, max_depth, add_bos_token, **strategy_args)
        host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "localhost")
        port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "1034")
        self.sandbox = LocalSandbox(host=host, port=port)

    def interact_with_environment(self, action, past_text, past_tokens):
        """base on the context text/tokens, decide whether need to augment the input action with environment tokens

        Args:
            action (int): the next action to take
            past_text (str):  context text
            past_tokens (List[int]): context tokens

        Returns:
            Union[List[int], int]: updated actions or original action
        """
        if past_text.endswith("</llm-code>"):
            try:
                code = extract_code_to_execute(past_text)
                output, uuid = self.sandbox.execute_code(code)
                results = output["result"]
                output_text = code_output_template.format(answer=results)
                output_tokens = self.model.tokenizer.text_to_ids(past_text + output_text)
                output_tokens = output_tokens[len(past_tokens) :]
                # modify the node action
                # it merges the node's state with the output tokens
                # and set the last token as the new action
                action = [action] + output_tokens
            except Exception as e:
                print("############ Code Environment Error ############")
                print(past_text)
                print(e)
        return action


class RemoteStateTransitionFunction(StateTransitionFunction):
    def __init__(self, host_url, model_name, inference_timeout_s=600):
        self.client = ModelClient(host_url, model_name, inference_timeout_s=inference_timeout_s)

    def __call__(self, sentences=None, actions=None, context_ids=None, session_info=None):
        context_ids = encode_context_data(context_ids)
        if sentences is not None:
            str_ndarray = np.array(sentences)[..., np.newaxis]
            input_data = np.char.encode(str_ndarray, "utf-8")
            result_dict = self.client.infer_batch(
                sentences=input_data, context_ids=context_ids, parameters={"session": session_info}
            )
        else:
            result_dict = self.client.infer_batch(
                action=actions, context_ids=context_ids, parameters={"session": session_info}
            )
        return result_dict
