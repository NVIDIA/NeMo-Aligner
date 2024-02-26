import numpy as np
import sentencepiece
from pytriton.client import ModelClient

from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.mcts import DeepSearch, MCTSParallel, ParallelSearch
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.search_callables import decode_context_data, encode_context_data

# set numpy seed
np.random.seed(42)
# set pytorch seed

# tokenizer = sentencepiece.SentencePieceProcessor('/dataset/models/unpack_843m_mcore/adfd4c68d8444aa790c2e65eab362a9f_a184c0997f35446cac66e8e2d63f7853_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model')

tokenizer = sentencepiece.SentencePieceProcessor(
    "/datasets/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model"
)


args = {
    "C": 2,
    "num_searches": 20,
    "rollout_micro_batch_size": 1,
    "num_epochs": 4,
    "temperature": 0.2,  # use low temperature for more greedy search
    "dirichlet_epsilon": 0.0,  # turn off the dirichlet noise
    "dirichlet_alpha": 0.3,
    "max_depth": 250,
    "save_timer": 135000,
    "turn_off_value": True,
    "oracle": True,
}


steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""
# steerlm_template="""<extra_id_0>System
# A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# <extra_id_1>User
# {prompt}
# Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
# <extra_id_1>Assistant
# <extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
# """


termination_condition = TerminationCondition(args["max_depth"], end_strings=["<extra_id_1>"])
score_fun = GSK8KFeedbackHF("train")
offset = 1765
ps = [
    ParallelSearch(tokenizer.encode(steerlm_template.format(prompt=score_fun.ds["train"][i]["question"])), i)
    for i in range(offset, offset + args["rollout_micro_batch_size"])
]


client = ModelClient("localhost:2323", "search", inference_timeout_s=600)


def get_client_fun():
    def trion_infer_batch(sentences=None, action=None, context_ids=None, session_info=None):
        context_ids = encode_context_data(context_ids)
        if sentences is not None:
            str_ndarray = np.array(sentences)[..., np.newaxis]
            input_data = np.char.encode(str_ndarray, "utf-8")
            result_dict = client.infer_batch(
                sentences=input_data, context_ids=context_ids, parameters={"session": session_info}
            )
        else:
            result_dict = client.infer_batch(
                action=action, context_ids=context_ids, parameters={"session": session_info}
            )
        return result_dict

    return trion_infer_batch


mcts = MCTSParallel(
    args,
    tokenizer,
    session_info="test_selfplay",
    score_fn=score_fun,
    terminate_fns=[termination_condition],
    client_fun=get_client_fun(),
)

ds = DeepSearch(mcts, args["max_depth"], args["temperature"], None, args["save_timer"], None)

# mcts.search(ps)
buffer = ds.search(ps, "test")
