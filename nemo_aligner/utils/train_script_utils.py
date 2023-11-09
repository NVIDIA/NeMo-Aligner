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

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn

from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils.exp_manager import NeMoModelCheckpoint
from nemo_aligner.utils.utils import custom_save_ckpt_func, extract_value_from_ckpt

"""Utilities for example scripts"""


def retrieve_custom_trainer_state_dict(ptl_trainer):
    """our custom trainer doesn't have access to the state dict because
        that logic is in NeMo. Instead we parse the state of our trainer
        from the loaded checkpoint path
    """
    consumed_samples = 0

    # pull values from checkpoint
    trainer_restore_path = ptl_trainer.ckpt_path
    trainer_state_dict = None

    if trainer_restore_path is not None:
        assert trainer_restore_path == ptl_trainer._checkpoint_connector._select_ckpt_path(
            ptl_trainer.state.fn, None, True, True
        )
        # extract the state dict from path because we are using PTL's mechanism
        consumed_samples = extract_value_from_ckpt(key="consumed_samples", ckpt_path=trainer_restore_path)
        step = extract_value_from_ckpt(key="step", ckpt_path=trainer_restore_path)
        epoch = extract_value_from_ckpt(key="epoch", ckpt_path=trainer_restore_path)
        ppo_optimization_step = extract_value_from_ckpt(key="ppo_optimization_step", ckpt_path=trainer_restore_path)
        trainer_state_dict = {
            "step": step,
            "consumed_samples": consumed_samples,
            "epoch": epoch,
            "ppo_optimization_step": ppo_optimization_step,
        }

    return trainer_state_dict


def init_distributed(ptl_trainer, ptl_model, use_te=False):
    """Initialize the distributed process group with help from the PTL Trrainer
    """
    # set up the trainer and initialize things
    ptl_trainer.state.fn = TrainerFn.FITTING
    ptl_trainer.strategy.connect(ptl_model)

    # Init DDP
    def dummy():
        return

    ptl_trainer.strategy.launcher.launch(dummy, trainer=ptl_trainer)
    ptl_trainer.strategy.setup_environment()

    if use_te:
        ptl_model.setup_transformer_engine_tp_groups()


def _fake_fn(*args, **kwargs):
    return


def disable_data_callbacks(ptl_model, train_dataloader, train_ds):
    """Disable data callbacks in NeMo Models so that setup() is called without setting up any data/dataloader
    """

    # we need to manually set train_ds/dl because this is used to compute steps for learning rate scheduler
    ptl_model._train_ds = train_ds
    ptl_model._train_dl = train_dataloader

    for attr in ["build_train_valid_test_datasets", "setup_training_data", "setup_validation_data", "setup_test_data"]:
        setattr(ptl_model, attr, _fake_fn)


def init_using_ptl(ptl_trainer, ptl_model, train_dataloader, train_ds):
    """initializes the model using PTL
    """
    disable_data_callbacks(ptl_model, train_dataloader, train_ds)

    call._call_setup_hook(ptl_trainer)
    ptl_trainer.strategy.setup(ptl_trainer)
    call._call_callback_hooks(ptl_trainer, "on_fit_start")
    call._call_lightning_module_hook(ptl_trainer, "on_fit_start")
    ptl_trainer._checkpoint_connector._restore_modules_and_callbacks(ptl_trainer.ckpt_path)
    ptl_trainer._checkpoint_connector.restore_training_state()
    ptl_trainer._checkpoint_connector.resume_end()
    scheduler = ptl_model._scheduler["scheduler"]

    # restore the previous state of the learning rate
    if scheduler.last_epoch > 0:
        # NOTE: we are doing this because load_state_dict on a LRScheduler
        # does not do anything that restores the learning rate on the optimizer
        # stepping here will restore it properly
        scheduler.step(scheduler.last_epoch)


def add_custom_checkpoint_callback(ptl_trainer, ptl_model):
    """get a function we can conveniently call within the trainer that saves the checkpoint  
    """
    for callback in ptl_trainer.callbacks:
        if isinstance(callback, NeMoModelCheckpoint):
            NeMoModelCheckpoint.custom_save_ckpt_func = custom_save_ckpt_func
            callback.custom_save = partial(callback.custom_save_ckpt_func, ptl_trainer, ptl_model)
            return callback

    class FakeCheckpointCallback:
        def custom_save(self, *args, **kwargs):
            ...

    return FakeCheckpointCallback()


def extract_optimizer_scheduler_from_ptl_model(ptl_model):
    return ptl_model.optimizers().optimizer, ptl_model.lr_schedulers()


@dataclass
class CustomLoggerWrapper:
    """a custom logger that wraps over a list of PTL loggers to make it easier
        for the trainer to call logging
    """

    loggers: list

    def apply_fn(self, name_of_fn, *args, **kwargs):
        for logger in self.loggers:
            if hasattr(logger, name_of_fn):
                getattr(logger, name_of_fn)(*args, **kwargs)

    def log_hyperparams(self, params):
        self.apply_fn("log_hyperparams", params)

    def log_table(self, *args, **kwargs):
        self.apply_fn("log_table", *args, **kwargs)

    def log_metrics(self, metrics, step=None, prefix=""):
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        self.apply_fn("log_metrics", metrics, step)

    def finalize(self):
        self.apply_fn("finalize", "success")


def resolve_and_create_trainer(cfg, pop_trainer_key):
    """resolve the cfg, remove the key before constructing the PTL trainer
        and then restore it after
    """
    OmegaConf.resolve(cfg)
    with temp_pop_from_config(cfg.trainer, pop_trainer_key):
        return MegatronTrainerBuilder(cfg).create_trainer()


@contextmanager
def temp_pop_from_config(cfg, name_to_temporarily_remove):
    """This is a context manager that removes a config
        and then adds it when the contextmanager exits, it's useful for patching configs over existing configs
    """
    with open_dict(cfg):
        temp = cfg.pop(name_to_temporarily_remove)
        try:
            yield
        finally:
            setattr(cfg, name_to_temporarily_remove, temp)
