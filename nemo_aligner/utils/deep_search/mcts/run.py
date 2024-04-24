from nemo_aligner.utils.deep_search.mcts.mcts import DeepSearch, MCTSParallel, ParallelSearch
from nemo_aligner.utils.deep_search.mcts.state_transition_functions import (
    LocalStateTransitionFunction,
    MathtoolLocalStateTransitionFunction,
)
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.text_generation_strategy import (
    GPTSearchTextGenerationStrategy,
    HybridGPTSearchTextGenerationStrategy,
    NoKVCacheGPTSearchTextGenerationStrategy,
    NoKVCacheHybridGPTSearchTextGenerationStrategy,
)


def run_mcts(batch, filename, ptl_model, score_fn, inference_only=False, has_value=True, use_cpu=False):
    mcts_cfg = ptl_model.cfg.mcts

    if has_value:
        if mcts_cfg.turn_off_kv_cache:
            strategy = NoKVCacheHybridGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
        else:
            strategy = HybridGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    else:
        if mcts_cfg.turn_off_kv_cache:
            strategy = NoKVCacheGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
        else:
            strategy = GPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    strategy_args = {"strategy": strategy}

    if mcts_cfg.environment == "code":
        client_fun = MathtoolLocalStateTransitionFunction(
            ptl_model, mcts_cfg.top_k, mcts_cfg.max_depth, mcts_cfg.add_bos_token, **strategy_args
        )
    else:
        client_fun = LocalStateTransitionFunction(
            ptl_model, mcts_cfg.top_k, mcts_cfg.max_depth, mcts_cfg.add_bos_token, **strategy_args
        )

    termination_condition = TerminationCondition(
        mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings, end_tokens=[ptl_model.tokenizer.eos_id]
    )

    mcts = MCTSParallel(
        mcts_cfg,
        ptl_model.tokenizer.tokenizer,
        session_info="test_selfplay",
        score_fn=score_fn,
        terminate_fns=[termination_condition],
        client_fun=client_fun,
        has_value=has_value,
    )

    ds = DeepSearch(
        mcts,
        mcts_cfg.max_depth,
        mcts_cfg.temperature,
        strategy,
        mcts_cfg.save_timer,
        mcts_cfg.cache_dir,
        inference_only=inference_only,
    )

    ps = []

    for question, data_id in zip(batch["question"], batch["data_id"]):
        if mcts_cfg.add_bos_token:
            ps.append(
                ParallelSearch([ptl_model.tokenizer.bos_id] + ptl_model.tokenizer.text_to_ids(question,), data_id,)
            )
        else:
            ps.append(ParallelSearch(ptl_model.tokenizer.text_to_ids(question,), data_id,))

    output = ds.search(ps, filename)
    return output
