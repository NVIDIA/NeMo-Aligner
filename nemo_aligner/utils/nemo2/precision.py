from dataclasses import fields
from typing import Literal

import torch
from torch.nn import Module
from torch.optim import Optimizer

from nemo.lightning.pytorch.plugins.mixed_precision import DtypeConfig
from nemo.utils import logging

## TODO: why doesn't our optimizer have a mcore_optimizer attribute?
def get_optim_config(optimizer: Optimizer):
    extract_config = lambda x: x.config
    #try:
    from megatron.core.optimizer import ChainedOptimizer

    if isinstance(optimizer, ChainedOptimizer):
        opts = optimizer.chained_optimizers
    else:
        opts = [optimizer]
    yield from map(extract_config, opts)
    #except:
    #    raise ValueError("Failed to extract optimizer config from module.")

## modified version of https://github.com/NVIDIA/NeMo/blob/6d90758fa36c1c0d4383d67c44785ac227990a4b/nemo/lightning/pytorch/plugins/mixed_precision.py
## without PTL dependency
class MegatronMixedPrecision:
    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed", "32"],
        params_dtype: torch.dtype = None,
        pipeline_dtype: torch.dtype = None,
        autocast_dtype: torch.dtype = None,
        autocast_enabled: bool = False,
        grad_reduce_in_fp32: bool = True,
        # fp8 related,
        fp8: str = None,
        fp8_margin: int = 0,
        fp8_amax_history_len: int = 1,
        fp8_amax_compute_algo: str = "most_recent",
        fp8_wgrad: bool = True,
        fp8_dot_product_attention: bool = False,
        fp8_multi_head_attention: bool = False,
        fp8_params: bool = False,
        fp16_loss_scale: float = None,
        fp16_initial_loss_scale: float = 4294967296,
        fp16_min_loss_scale: float = 1.0,
        fp16_loss_scale_window: int = 1000,
        fp16_hysteresis: int = 2,
    ) -> None:

        if isinstance(precision, int):
            precision = str(precision)

        fp8_param_gather = False
        if fp8 is not None:
            te_fp8, HAVE_TE = safe_import("transformer_engine.pytorch.fp8")
            assert HAVE_TE, "FP8 precision requires transformer engine."
            if fp8_params:
                te_fp8.FP8GlobalStateManager.FP8_PARAMETERS = True
                fp8_param_gather = True

        dtype = torch.bfloat16 if precision in ['bf16', 'bf16-mixed'] else torch.float32
        self.dtype_config = DtypeConfig(
            fp32=precision in ['fp32', '32'],
            fp16=precision in ['fp16', 'fp16-mixed', '16', '16-mixed'],
            bf16=precision in ['bf16', 'bf16-mixed'],
            params_dtype=params_dtype or torch.float32,
            pipeline_dtype=pipeline_dtype or dtype,
            autocast_dtype=autocast_dtype or dtype,
            autocast_enabled=autocast_enabled,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            fp8=fp8,
            fp8_margin=fp8_margin,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_wgrad=fp8_wgrad,
            fp8_dot_product_attention=fp8_dot_product_attention,
            fp8_multi_head_attention=fp8_multi_head_attention,
            fp8_param_gather=fp8_param_gather,
            # fp16 loss scale
            loss_scale=fp16_loss_scale,
            initial_loss_scale=fp16_initial_loss_scale,
            min_loss_scale=fp16_min_loss_scale,
            loss_scale_window=fp16_loss_scale_window,
            hysteresis=fp16_hysteresis,
        )
        #super().__init__()
        if self.dtype_config.fp16:
            self.precision = "16-mixed"
        elif self.dtype_config.bf16:
            self.precision = "bf16-mixed"
        else:
            self.precision = "32-true"

    def convert_module(self, module: Module) -> Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config

        if self.dtype_config.fp16 or self.dtype_config.bf16:
            # Patch config options
            config = get_model_config(module.module)
            config.fp16 = self.dtype_config.fp16
            config.bf16 = self.dtype_config.bf16
            if hasattr(module, 'module'):
                module.module = Float16Module(config, module.module)
            else:
                module = Float16Module(config, module)

        return module

    def verify_dtype(self, optimizer: Optimizer) -> Optimizer:
        """Convert the optimizer parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        for optim_config in get_optim_config(optimizer):
            assert optim_config.bf16 == self.dtype_config.bf16, "BF16 model/optim config mismatch"
            assert optim_config.fp16 == self.dtype_config.fp16, "FP16 model/optim config mismatch"

    def update_config_with_dtype_overrides(self, config):
        if hasattr(config, "__io__"):
            config.__io__ = self.update_config_with_dtype_overrides(config.__io__)
        for field in fields(self.dtype_config):
            if not hasattr(config, field.name):
                continue
            # If we overwrote a value, log a debug message.
            old_val = getattr(config, field.name)
            new_val = getattr(self.dtype_config, field.name)
            if old_val != new_val:
                setattr(config, field.name, new_val)
                logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
        return config
