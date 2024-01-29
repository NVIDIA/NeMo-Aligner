# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.megatron_gpt_hybrid_model import MegatronGPTHybridModel
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedback
from nemo_aligner.utils.deep_search.mcts.mcts import MCTSParallel, ParallelSearch, deep_search
from nemo_aligner.utils.deep_search.mcts.termination_condition import TerminationCondition
from nemo_aligner.utils.deep_search.text_gen_utils import dp_search
from nemo_aligner.utils.deep_search.text_generation_strategy import HybridGPTSearchTextGenerationStrategy
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)

steerlm_template = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""
#


@hydra_runner(config_path="conf", config_name="gpt_hybrid_infer")
def main(cfg) -> None:
    """
    Binary ranking reward models use comparison based objective similar to the one found in the
    InstructGPT paper: https://arxiv.org/pdf/2203.02155.pdf and have no explicit labels.
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    hybrid_model_cls = MegatronGPTHybridModel

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    cfg.model.value = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model.value)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    # exp_manager(trainer, cfg.exp_manager)
    # logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        hybrid_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    # # pull values from checkpoint
    # trainer_restore_path = trainer.ckpt_path
    # if trainer_restore_path is not None:
    #     custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
    #     consumed_samples = custom_trainer_state_dict["consumed_samples"]
    # else:
    #     custom_trainer_state_dict = None
    #     consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    # ptl_model.save_to("hybrid_model.nemo")

    dp_size = parallel_state.get_data_parallel_world_size()
    max_batch_size = cfg.inference.micro_batch_size * dp_size

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    # strategy_args = {"strategy": strategy}
    strategy = HybridGPTSearchTextGenerationStrategy(ptl_model)
    strategy_args = {"strategy": strategy}

    def get_client_fun(model, sampling_params, length_params, **strategy_args):
        # one token at a time
        tokens_to_generate = length_params["max_length"]
        top_k = sampling_params["top_k"]
        end_strings = sampling_params["end_strings"]

        def native_dp_search(sentences=None, action=None, context_ids=None, session_info=None):
            return dp_search(
                model,
                inputs=sentences,
                action=action,
                context_ids=context_ids,
                session_info=session_info,
                tokens_to_generate=tokens_to_generate,  # max search depth
                top_k=top_k,
                end_strings=end_strings,
                **strategy_args,
            )

        return native_dp_search

    args = {
        "C": 2,
        "num_searches": 500,
        "num_selfPlay_iterations": 10,
        "num_parallel_searchs": 1,
        "num_epochs": 4,
        "batch_size": 128,
        "temperature": 0.2,  # use low temperature for more greedy search
        "dirichlet_epsilon": 0.0,  # turn off the dirichlet noise
        "dirichlet_alpha": 0.3,
        "max_depth": 200,
    }

    termination_condition = TerminationCondition(args["max_depth"], end_strings=["<extra_id_1>"])
    score_fun = GSK8KFeedback("train-00000-of-00001.parquet")

    dp_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()

    ps = [
        ParallelSearch(
            ptl_model.tokenizer.text_to_ids(
                steerlm_template.format(
                    prompt=score_fun.gsk8k.iloc[i + dp_rank * args["num_parallel_searchs"]]["question"]
                )
            ),
            i + dp_rank * args["num_parallel_searchs"],
        )
        for i in range(args["num_parallel_searchs"])
    ]

    mcts = MCTSParallel(
        args,
        ptl_model.tokenizer.tokenizer,
        session_info="test_selfplay",
        score_fn=score_fun,
        terminate_fns=[termination_condition],
        client_fun=get_client_fun(
            ptl_model, sampling_params=sampling_params, length_params=length_params, **strategy_args
        ),
    )

    buffer = deep_search(ps, mcts, args["max_depth"], args["temperature"])


if __name__ == "__main__":
    main()
