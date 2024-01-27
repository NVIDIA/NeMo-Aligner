import numpy as np
import torch
from megatron.core import InferenceParams
import pytest

from nemo_aligner.utils.deep_search.mcts.mcts import Node, State
from nemo_aligner.utils.deep_search.mcts.search_db import SearchDB, get_kv_cache

from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo_aligner.utils.deep_search.text_generation_strategy import GPTSearchTextGenerationStrategy

from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytriton.model_config import ModelConfig
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.utils.deep_search.search_callables import SearchCallable
from nemo_aligner.utils.deep_search.text_gen_utils import search
from nemo_aligner.utils.deep_search.text_generation_strategy import GPTSearchTextGenerationStrategy
from nemo_aligner.utils.train_script_utils import init_distributed

try:
    from megatron.core import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

@pytest.mark.run_only_on("GPU")
class TestSearch:

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        GPUS = 1
        trainer = Trainer(strategy=NLPDDPStrategy(), devices=GPUS, accelerator='gpu', num_nodes=1, logger=None, precision='bf16')

        save_restore_connector = NLPSaveRestoreConnector()
        path = '/dataset/models/unpack_843m_mcore/'

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path='/dataset/models/unpack_843m_mcore/',
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
        cls.model = MegatronGPTModel.restore_from(
            restore_path=path,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f"cuda:{trainer.local_rank}",  # map_location is needed for converted models
        )
        cls.model.freeze()
        init_distributed(trainer, cls.model, False)


    @pytest.mark.run_only_on("GPU")
    def test_search_infer(self):
        strategy = GPTSearchTextGenerationStrategy(self.model)
        strategy_args = {"strategy": strategy}
        tokens_to_generate = 10
        top_k = 50
        end_strings = ['']

        def infer_fn(inputs=None, action=None, context_ids=None, session_info=None):
            return search(
                    self.model,
                    inputs,
                    action,
                    context_ids,
                    session_info,
                    tokens_to_generate=tokens_to_generate,  # max search depth
                    top_k=top_k,
                    end_strings=end_strings,
                    **strategy_args,
                    )
        batched_inputs = ['hello?', 'how are you?']
        tokenizer = self.model.tokenizer
        context_ids = [tuple(tokenizer.text_to_ids(input)) for input in batched_inputs]
        token_lengths = [len(input) for input in context_ids]
        output = infer_fn(inputs=batched_inputs, action=None, context_ids=context_ids, session_info='session1')
        assert strategy.search_db.get_inference_params('session1').sequence_len_offset == max(token_lengths) - 1
        assert 'session1' in strategy.search_db.db
        assert len(strategy.search_db.db['session1']) == 2 + top_k * 2 # root + children
        for action in output['action'][0]:
            assert (context_ids[0], action) in strategy.search_db.db['session1']
        for action in output['action'][1]:
            assert (context_ids[1], action) in strategy.search_db.db['session1']
        assert (context_ids[0], -1) in strategy.search_db.db['session1']
        assert (context_ids[1], -1) in strategy.search_db.db['session1']

        old_k_caches = []
        batch_size = len(batched_inputs)
        for i in range(batch_size):
            old_k_caches.append(strategy.search_db.get_inference_params('session1').key_value_memory_dict[1][0][:, i, 0, 0])
        for i in range(batch_size):
            assert old_k_caches[i].shape[0] == max(token_lengths) + 1
        
        for step in range(10):
            # next inference call 
            depths = np.array([[step + 1], [step + 1]], dtype=np.int32)
            actions = output['action'][:, 0:1] # greedy sampling
            output = infer_fn(inputs=None, action=actions, context_ids=context_ids, session_info='session1')
            assert strategy.search_db.get_inference_params('session1').sequence_len_offset == max(token_lengths) + step
            context_ids = [parent_ids + (child.item(),) for parent_ids, child in zip(context_ids, actions)]
            for action in output['action'][0]:
                assert (context_ids[0], action) in strategy.search_db.db['session1']
            for action in output['action'][1]:
                assert (context_ids[1], action) in strategy.search_db.db['session1']
            new_k_caches = []
            for i in range(batch_size):
                new_k_caches.append(strategy.search_db.get_inference_params('session1').key_value_memory_dict[1][0][:, i, 0, 0])
            # test if the k_cache_b0 is close to k_cache_b0_v2
            context_length = np.array(token_lengths) + depths[:, 0]
            for i in range(batch_size):
                # make sure the context k cache matches for the new inference 
                assert torch.allclose(old_k_caches[i][0:context_length[i] -1], new_k_caches[i][0:context_length[i] -1])
            old_k_caches = new_k_caches
