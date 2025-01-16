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

from typing import Callable

from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig

from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.lightning._strategy_lib import setup_megatron_optimizer

## copied from nemo, just removed dep on PTL
class LRScheduler:
    
   def connect(self, model, optimizer) -> None:
        """Sets up the learning rate scheduler.

        Args:
            model: The model for which the scheduler is being set up.
            optimizer: The optimizer for which the scheduler is being set up.
        """
        ...

    @abstractmethod
    def scheduler(self, model, optimizers) -> OptimizerLRScheduler:
        """Abstract method to define the learning rate scheduler.

        Args:
            model: The model for which the scheduler is being defined.
            optimizers: The optimizers for which the scheduler is being defined.

        Returns:
            OptimizerLRScheduler: The learning rate scheduler.
        """
        raise NotImplementedError("The scheduler method should be implemented by subclasses.")


    def __call__(self, model, optimizer): ## maybe rename this to setup? 

        ## TODO: understand, refactor
        self.connect(model, optimizers)

        self._scheduler = self.scheduler(model, optimizers)

        if not isinstance(self._scheduler, (dict, tuple)):
            return optimizers, self._scheduler

        ## returns a dict
        return self._scheduler

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

    def scheduler(self, model, optimizer):
        from nemo.core.optim.lr_scheduler import CosineAnnealing

        lr_scheduler = CosineAnnealing(
            optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            constant_steps=self.constant_steps,
            min_lr=self.min_lr,
        )

        ## TODO: this is PTL's way of doing things. convert this to pure PTL so we can call scheduler directly
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": self.frequency,
            },
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor,
        }

class MegatronOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
        lr_scheduler: Optional["LRScheduler"] = None, ## TODO: type hint
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
    self.config.finalize_model_grads_func = finalize_model_grads

    def setup_optimizer_and_lr_schedule(self, model):
        ## TODO: make sure this supports models that are not instances of megatron_parallel
        optimizer = setup_megatron_optimizer(
            model,
            self.config,
            no_weight_decay_cond=self.no_weight_decay_cond,
            scale_lr_cond=self.scale_lr_cond,
            lr_mult=self.lr_mult,
        )

        if self.lr_scheduler is not None:
            optimizer = self.lr_scheduler(model, optimizer)
        
        self.optimizer = optimizer
    
    def step(self):
        assert hasattr(self.optimizer), "make sure to call setup_optimizer_and_lr_schedule first"

        ## TODO: how does this work if we return a dict with the LR scheduler?
        self.optimizer.step()
