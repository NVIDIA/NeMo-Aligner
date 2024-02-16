import itertools
import sys

from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.utils import logging

try:

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPDDPStrategyNotebook


class CustomNLPDDPStrategy(NLPDDPStrategy):
    def optimizer_sharded_state_dict(self):
        """
        Sharded state dictionary for an MainParamsOptimizerWrapper.
        Used to save and load the optimizer state when training with distributed_checkpoint.
        Returns:
            dict: The sharded state dictionary for the optimizer
        Raises:
            ValueError: If a parameter ID does not match any model sharded parameter.
        """

        optimizer = self.lightning_module.optimizers(use_pl_optimizer=False)  # MainParamsOptimizerWrapper

        model_sharded_state_dict = self.lightning_module.sharded_state_dict()

        # remove _extra_state
        model_sharded_state_dict = {
            key: value for key, value in model_sharded_state_dict.items() if not key.endswith("_extra_state")
        }

        if not isinstance(optimizer.policy_optimizer, MainParamsOptimizerWrapper):
            return optimizer.sharded_state_dict(model_sharded_state_dict)

        optimizer_state_dict = optimizer.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=itertools.chain.from_iterable([g for g in optimizer.policy_optimizer.float16_groups]),
        )

        assert len(optimizer_state_dict[("policy", "fp32_from_fp16_params")]) == len(
            optimizer_state_dict[("policy", "optimizer")]["param_groups"]
        )

        assert len(optimizer_state_dict[("value", "fp32_from_fp16_params")]) == len(
            optimizer_state_dict[("value", "optimizer")]["param_groups"]
        )

        def get_safe(param_id):
            try:
                return id_to_sharded_param_map[param_id]
            except KeyError as e:
                raise ValueError(f"Param id {param_id} does not match any model sharded param") from e

        optimizer_state_dict[("policy", "fp32_from_fp16_params")] = [
            [
                make_sharded_optimizer_tensor(get_safe(param_id), fp32_param, prefix=f"optimizer.state.fp32_param")
                for param_id, fp32_param in zip(state_group["params"], fp32_group)
            ]
            for fp32_group, state_group in zip(
                optimizer_state_dict[("policy", "fp32_from_fp16_params")],
                optimizer_state_dict[("policy", "optimizer")]["param_groups"],
            )
        ]

        # Convert state
        optim_state_to_sharding_state(optimizer_state_dict[("policy", "optimizer")], id_to_sharded_param_map)

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=itertools.chain.from_iterable([g for g in optimizer.value_optimizer.float16_groups]),
        )

        optimizer_state_dict[("value", "fp32_from_fp16_params")] = [
            [
                make_sharded_optimizer_tensor(get_safe(param_id), fp32_param, prefix=f"optimizer.state.fp32_param")
                for param_id, fp32_param in zip(state_group["params"], fp32_group)
            ]
            for fp32_group, state_group in zip(
                optimizer_state_dict[("value", "fp32_from_fp16_params")],
                optimizer_state_dict[("value", "optimizer")]["param_groups"],
            )
        ]

        # Convert state
        optim_state_to_sharding_state(optimizer_state_dict[("value", "optimizer")], id_to_sharded_param_map)

        return optimizer_state_dict


class CustomMegatronTrainerBuilder(MegatronTrainerBuilder):
    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        # check interactive environment
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)
        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            logging.info("Detected interactive environment, using NLPDDPStrategyNotebook")
            return NLPDDPStrategyNotebook(no_ddp_communication_hook=True, find_unused_parameters=False,)

        return CustomNLPDDPStrategy(
            no_ddp_communication_hook=True,
            gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )
