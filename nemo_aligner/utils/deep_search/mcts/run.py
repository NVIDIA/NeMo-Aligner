import json
from collections import namedtuple

from nemo_aligner.utils.deep_search.mcts.mcts import DeepSearch, MCTSParallel, ParallelSearch
from nemo_aligner.utils.deep_search.mcts.search_stop_criteria import SearchStopCriteria
from nemo_aligner.utils.deep_search.mcts.state_transition_functions import (
    LocalStateTransitionFunction,
    MathtoolLocalStateTransitionFunction,
    TRTLLMLocalStateTransitionFunction,
)
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.mcts.value_estimation_function import ValueApproximationFunction
from nemo_aligner.utils.deep_search.text_generation_strategy import (
    GPTSearchTextGenerationStrategy,
    HybridGPTSearchTextGenerationStrategy,
    NoKVCacheGPTSearchTextGenerationStrategy,
    NoKVCacheHybridGPTSearchTextGenerationStrategy,
)


def dict_to_namedtuple(d):
    return json.loads(json.dumps(d), object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))


def run_mcts(
    batch, filename, ptl_model, score_fn, eos_id, bos_id, pad_id, inference_only=False, has_value=True, use_cpu=False,
):
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
            ptl_model,
            mcts_cfg.top_k,
            mcts_cfg.max_depth,
            mcts_cfg.add_bos_token,
            mcts_cfg.child_threshold,
            **strategy_args
        )
    else:
        client_fun = LocalStateTransitionFunction(
            ptl_model,
            mcts_cfg.top_k,
            mcts_cfg.max_depth,
            mcts_cfg.add_bos_token,
            mcts_cfg.child_threshold,
            **strategy_args
        )

    termination_condition = TerminationCondition(
        mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings, end_tokens=[ptl_model.tokenizer.eos_id]
    )

    stop_criteria = SearchStopCriteria(score_fn, [termination_condition], threshold=mcts_cfg.value_threshold)

    value_estimation_function = None
    if mcts_cfg.simulate_value:
        value_estimation_function = ValueApproximationFunction(
            ptl_model.tokenizer.tokenizer, stop_criteria, pad_id, mcts_cfg.add_bos_token
        )

    mcts = MCTSParallel(
        mcts_cfg,
        ptl_model.tokenizer.tokenizer,
        pad_id,
        session_info="test_selfplay",
        stop_criteria=stop_criteria,
        client_fun=client_fun,
        has_value=has_value,
        value_estimation_function=value_estimation_function,
    )

    ds = DeepSearch(
        mcts,
        mcts_cfg.max_depth,
        mcts_cfg.temperature,
        strategy,
        mcts_cfg.save_timer,
        mcts_cfg.max_wall_time,
        mcts_cfg.top_k,
        mcts_cfg.cache_dir,
        inference_only=inference_only,
    )

    ps = []

    for question, data_id in zip(batch["question"], batch["data_id"]):
        if mcts_cfg.add_bos_token and bos_id is not None:
            ps.append(ParallelSearch([bos_id] + ptl_model.tokenizer.text_to_ids(question), data_id))
        else:
            ps.append(ParallelSearch(ptl_model.tokenizer.text_to_ids(question), data_id))

    output = ds.search(ps, filename)
    return output


def run_trtllm_mcts(
    batch, filename, trtllm_infer, mcts_cfg, score_fn, eos_id, bos_id, pad_id, inference_only=False, has_value=True,
):

    if mcts_cfg.environment == "code":
        raise NotImplementedError("Code environment is not supported for TRTLLM model")
    else:
        client_fun = TRTLLMLocalStateTransitionFunction(
            trtllm_infer, mcts_cfg.top_k, mcts_cfg.max_depth, mcts_cfg.add_bos_token, mcts_cfg.child_threshold,
        )

    termination_condition = TerminationCondition(
        mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings, end_tokens=[eos_id]
    )

    stop_criteria = SearchStopCriteria(score_fn, [termination_condition], threshold=mcts_cfg.value_threshold)

    value_estimation_function = None
    if mcts_cfg.simulate_value:
        value_estimation_function = ValueApproximationFunction(
            trtllm_infer.tokenizer, stop_criteria, pad_id, mcts_cfg.add_bos_token
        )

    mcts = MCTSParallel(
        mcts_cfg,
        trtllm_infer.tokenizer,
        pad_id,
        session_info="test_selfplay",
        stop_criteria=stop_criteria,
        client_fun=client_fun,
        has_value=has_value,
        value_estimation_function=value_estimation_function,
    )

    ds = DeepSearch(
        mcts,
        mcts_cfg.max_depth,
        mcts_cfg.temperature,
        None,
        mcts_cfg.save_timer,
        mcts_cfg.max_wall_time,
        mcts_cfg.top_k,
        mcts_cfg.cache_dir,
        inference_only=inference_only,
    )

    ps = []

    for question, data_id in zip(batch["question"], batch["data_id"]):
        if mcts_cfg.add_bos_token and bos_id is not None:
            ps.append(ParallelSearch([bos_id] + trtllm_infer.tokenizer.encode(question), data_id))
        else:
            ps.append(ParallelSearch(trtllm_infer.tokenizer.encode(question), data_id))

    output = ds.search(ps, filename)
    return output
