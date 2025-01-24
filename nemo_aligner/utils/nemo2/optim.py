# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
from abc import abstractmethod
from typing import Callable, Optional

from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig
from megatron.core.utils import get_model_config

from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.lightning._strategy_lib import setup_megatron_optimizer

class LRScheduler:
    
    @abstractmethod
    def init_scheduler(self, model, optimizers):
        """Abstract method to define the learning rate scheduler.

        Args:
            model: The model for which the scheduler is being defined.
            optimizers: The optimizers for which the scheduler is being defined.

        Returns:
            OptimizerLRScheduler: The learning rate scheduler.
        """
        raise NotImplementedError("The scheduler method should be implemented by subclasses.")


    def __call__(self, model, optimizer): ## maybe rename this to setup? 

        """self.connect(model, optimizers)

        self._scheduler = self.init_scheduler(model, optimizers)

        if not isinstance(self._scheduler, (dict, tuple)):
            return optimizers, self._scheduler

        return self._scheduler"""

        return self.init_scheduler(model, optimizer)

## copied from nemo
class CosineAnnealingScheduler(LRScheduler):
    def __init__(
        self,
        max_steps: int = 10,
        warmup_steps: int = 750,
        constant_steps: int = 80000,
        min_lr: float = 6e-5,
        interval: str = "step",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def init_scheduler(self, model, optimizer):
        from nemo.core.optim.lr_scheduler import CosineAnnealing

        lr_scheduler = CosineAnnealing(
            optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            constant_steps=self.constant_steps,
            min_lr=self.min_lr,
        )

        ## TODO: verify that this is all we need
        """return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.interval, ## TODO: verify that this is only used in the case of automatic optimization
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": self.frequency, ## How often to increment the LR scheduler. Is this needed?
            },
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor, ## do we need this?
        }"""
        return lr_scheduler

## copied from https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/_strategy_lib.py#L600, 
## with some minor modifications to remove the dep on strategy
def setup_megatron_optimizer(
    model,
    config,
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from nemo.core.optim import McoreDistributedOptimizer

    assert isinstance(config, OptimizerConfig), f"Expected OptimizerConfig, got {type(config)}"

    class McoreOpt(McoreDistributedOptimizer):
        def sharded_state_dict(
            self,
            model_sharded_state_dict,
            optimizer_state_dict=None,
            is_loading=False,
            sharding_type='fully_sharded_model_space',
        ):
            mcore_optimizer_sig = inspect.signature(self.mcore_optimizer.sharded_state_dict).parameters
            distrib_optim_kwargs = {}
            if "sharding_type" in mcore_optimizer_sig:
                distrib_optim_kwargs["sharding_type"] = sharding_type
            state_dict = self.mcore_optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading=is_loading, **distrib_optim_kwargs
            )
            return state_dict

    # TODO: update once we migrate to nemo2
    ddp_modules = model
    #ddp_modules = [m.module for m in model]
    mcore_opt = get_megatron_optimizer(
        config,
        ddp_modules,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )

    """if getattr(model.ddp_config, "overlap_param_gather", False) and getattr(
        model.ddp_config, "align_param_gather", False
    ):
        param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        param_sync_func = param_sync_func[0] if len(model) == 1 else param_sync_func
        for module in model:
            module.config.param_sync_func = param_sync_func"""

    return McoreOpt(mcore_opt)

class MegatronOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
        lr_scheduler: Optional[LRScheduler] = None, ## TODO: type hint
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.no_weight_decay_cond = no_weight_decay_cond
        self.scale_lr_cond = scale_lr_cond
        self.lr_mult = lr_mult

        ## add finalize_model_grads func to config
        ## todo: make sure the args passed to this function are correct
        #self.config.finalize_model_grads_func = finalize_model_grads

    def connect(self, model):
        model.optim = self

        """if hasattr(self, "__io__") and hasattr(model, "__io__"):
            if hasattr(model.__io__, "optim"):
                model.__io__.optim = deepcopy(self.__io__)"""

    def setup_optimizer_and_lr_schedule(self, model):
        
        ## TODO: change this when we migrate from nemo1
        #get_model_config(model).finalize_model_grads_func = finalize_model_grads
        get_model_config(model.model[0]).finalize_model_grads_func = finalize_model_grads

        self.optimizer = setup_megatron_optimizer(
            model.model,
            self.config,
            no_weight_decay_cond=self.no_weight_decay_cond,
            scale_lr_cond=self.scale_lr_cond,
            lr_mult=self.lr_mult,
        )

        self.scheduler = None
        if self.lr_scheduler is not None:
            self.scheduler = self.lr_scheduler(model, self.optimizer)
        
    
    def step(self):
        assert hasattr(self, "optimizer"), "make sure to call setup_optimizer_and_lr_schedule first"

        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups