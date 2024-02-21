from megatron.core import InferenceParams, parallel_state

from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackHF
from nemo_aligner.utils.deep_search.mcts.mcts import DeepSearch, MCTSParallel, ParallelSearch
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.text_gen_utils import dp_search
from nemo_aligner.utils.deep_search.text_generation_strategy import HybridGPTSearchTextGenerationStrategy


def run_mcts(batch, filename, ptl_model, score_fn):
    mcts_cfg = ptl_model.cfg.mcts

    strategy = HybridGPTSearchTextGenerationStrategy(ptl_model)
    strategy_args = {"strategy": strategy}

    def get_client_fun(model, top_k, max_depth, **strategy_args):
        # one token at a time
        def native_dp_search(sentences=None, action=None, context_ids=None, session_info=None):
            return dp_search(
                model,
                inputs=sentences,
                action=action,
                context_ids=context_ids,
                session_info=session_info,
                tokens_to_generate=max_depth,  # max search depth
                top_k=top_k,
                **strategy_args,
            )

        return native_dp_search

    termination_condition = TerminationCondition(mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings)

    mcts = MCTSParallel(
        mcts_cfg,
        ptl_model.tokenizer.tokenizer,
        session_info="test_selfplay",
        score_fn=score_fn,
        terminate_fns=[termination_condition],
        client_fun=get_client_fun(ptl_model, mcts_cfg.top_k, mcts_cfg.max_depth, **strategy_args),
    )

    ds = DeepSearch(mcts, mcts_cfg.max_depth, mcts_cfg.temperature, strategy, mcts_cfg.save_timer, mcts_cfg.cache_dir)

    ps = []

    for question, answer, data_id in zip(batch["question"], batch["answer"], batch["data_id"]):
        ps.append(ParallelSearch(ptl_model.tokenizer.text_to_ids(question,), data_id,))

    output = ds.search(ps, filename)
    return output
