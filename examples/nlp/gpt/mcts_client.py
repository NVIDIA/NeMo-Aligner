import numpy as np
import sentencepiece
from pytriton.client import ModelClient

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo_aligner.utils.deep_search.mcts.feedback_functions import (
    DummyScore,
    GSK8KFeedbackDataset,
    GSK8KFeedbackHF,
    HelpSteerFeedback,
    SteerLMFeedback,
)
from nemo_aligner.utils.deep_search.mcts.mcts import DeepSearch, MCTSParallel, ParallelSearch
from nemo_aligner.utils.deep_search.mcts.run import dict_to_namedtuple
from nemo_aligner.utils.deep_search.mcts.search_stop_criteria import SearchStopCriteria
from nemo_aligner.utils.deep_search.mcts.state_transition_functions import (
    LocalStateTransitionFunction,
    MathtoolLocalStateTransitionFunction,
    RemoteStateTransitionFunction,
    TRTLLMLocalStateTransitionFunction,
)
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.mcts.value_estimation_function import ValueApproximationFunction
from nemo_aligner.utils.deep_search.search_callables import decode_context_data, encode_context_data
from nemo_aligner.utils.deep_search.text_generation_strategy import (
    GPTSearchTextGenerationStrategy,
    HybridGPTSearchTextGenerationStrategy,
    NoKVCacheGPTSearchTextGenerationStrategy,
    NoKVCacheHybridGPTSearchTextGenerationStrategy,
)

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""


@hydra_runner(config_path="conf", config_name="gpt_hybrid_train")
def main(cfg) -> None:
    # set numpy seed
    np.random.seed(42)

    # library = "huggingface"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # tokenizer_model = None

    library = "sentencepiece"
    model_name = "solar"
    tokenizer_model = (
        "/datasets/models/unpack_10b_solar_steerlm/0c96894aab214922922f717b00c1a8e4_solar_tokenizer.model"
    )
    tokenizer = get_nmt_tokenizer(
        library=library, model_name=model_name, tokenizer_model=tokenizer_model, use_fast=True
    )
    tokenizer = tokenizer
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id

    mcts_cfg = cfg.model.mcts
    has_value = cfg.pretrained_checkpoint.has_value_head

    # args = {
    #     "C": 2,
    #     "num_searches": 20,
    #     "rollout_micro_batch_size": 1,
    #     "num_epochs": 4,
    #     "temperature": 0.2,  # use low temperature for more greedy search
    #     "dirichlet_epsilon": 0.0,  # turn off the dirichlet noise
    #     "dirichlet_alpha": 0.3,
    #     "max_depth": 250,
    #     "save_timer": 135000,
    #     "turn_off_value": False,
    #     "oracle": True,
    #     "environment": "simple",
    #     "feedback": "helpsteer",
    #     "reward_weights": [0.0, 0.0, 0.0, 0.0, 0.65, 0.8, 0.45, 0.0, 0.0], # reward weights for the helpsteer reward function
    #     "add_bos_token": False,
    #     "end_strings": ["<extra_id_1>"],
    #     "value_threshold": 0.9,
    #     "simulate_value": True,
    #     "top_k": 50,
    #     "cache_dir": None,
    #     "child_threshold": 0.01,
    # }

    # has_value = True
    # # convert args into a named tuple
    # mcts_cfg = dict_to_namedtuple(args)

    # if has_value:
    #     if mcts_cfg.turn_off_kv_cache:
    #         strategy = NoKVCacheHybridGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    #     else:
    #         strategy = HybridGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    # else:
    #     if mcts_cfg.turn_off_kv_cache:
    #         strategy = NoKVCacheGPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    #     else:
    #         strategy = GPTSearchTextGenerationStrategy(ptl_model, use_cpu=use_cpu)
    # strategy_args = {"strategy": strategy}

    if mcts_cfg.environment == "code":
        raise NotImplementedError("Code environment not implemented")
    else:
        client_fun = RemoteStateTransitionFunction(
            "localhost:2323", "search", mcts_cfg.child_threshold, inference_timeout_s=600
        )

    termination_condition = TerminationCondition(
        mcts_cfg.max_depth, end_strings=mcts_cfg.end_strings, end_tokens=[eos_id]
    )

    if mcts_cfg.feedback == "math":
        ds = None
        score_fn = GSK8KFeedbackDataset(ds)
    elif mcts_cfg.feedback == "steerlm":
        score_fn = SteerLMFeedback()
    elif mcts_cfg.feedback == "helpsteer":
        score_fn = HelpSteerFeedback(mcts_cfg.reward_weights)
    elif mcts_cfg.feedback == "dummy":
        score_fn = DummyScore()
    else:
        raise ValueError(f"Invalid feedback function {mcts_cfg.feedback}")

    stop_criteria = SearchStopCriteria(score_fn, [termination_condition], threshold=mcts_cfg.value_threshold)

    value_estimation_function = None
    if mcts_cfg.simulate_value:
        value_estimation_function = ValueApproximationFunction(
            tokenizer.tokenizer, stop_criteria, pad_id, mcts_cfg.add_bos_token
        )

    mcts = MCTSParallel(
        mcts_cfg,
        tokenizer.tokenizer,
        tokenizer.bos_id,
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
        mcts_cfg.top_k,
        mcts_cfg.cache_dir,
        inference_only=False,
    )

    prompts = ["why the sky is blue?", "what's the capital of France?"]
    ps = [
        ParallelSearch(tokenizer.tokenizer.encode(steerlm_template.format(prompt=prompts[i])), i)
        for i in range(len(prompts))
    ]

    # mcts.search(ps)
    buffer = ds.search(ps, "test")

    print(buffer)


if __name__ == "__main__":
    main()
