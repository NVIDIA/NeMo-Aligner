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

import random
from collections import defaultdict
from enum import IntEnum, auto
from functools import partial
from math import ceil
from typing import Union

import pandas as pd
import torch
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.utils import logging
from nemo_aligner.utils.deep_search.mcts.run import run_mcts
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.train_utils import clip_gradients, clip_optimier_gradients
from nemo_aligner.utils.trainer_utils import check_progress
from nemo_aligner.utils.utils import clear_memory


class TrainMode(IntEnum):
    VALUE_ONLY = auto()
    POLICY_ONLY = auto()

    def cuda(self):
        return torch.cuda.LongTensor([self])


def safe_divide(a, b):
    return a / b if b != 0 else 0


def compute_limit_batches(number_of_batches: int, limit_batches: Union[int, float, None]):
    if limit_batches is None:
        limit_batches = 1.0

    if isinstance(limit_batches, float):
        limit_batches = int(number_of_batches * limit_batches)
    elif isinstance(limit_batches, int):
        limit_batches = min(number_of_batches, limit_batches)
    else:
        raise TypeError(f"Invalid data type of {type(limit_batches)} cannot compute limit batches")

    return limit_batches


class DeepSearchTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_policy_dataloader,
        train_value_dataloader,
        val_policy_dataloader,
        val_value_dataloader,
        val_dataloader_builder_func,
        train_dataloader_builder_func,
        feedback,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_policy_dataloader = train_policy_dataloader
        self.train_value_dataloader = train_value_dataloader
        self.val_dataloader_builder_func = val_dataloader_builder_func
        self.train_dataloader_builder_func = train_dataloader_builder_func

        self.val_policy_dataloader = val_policy_dataloader
        self.val_value_dataloader = val_value_dataloader

        self.feedback = feedback
        self.logger = logger
        self.ckpt_callback = ckpt_callback
        self.run_timer = run_timer

        self.step = 0
        self.epoch = 0
        self.policy_optimization_step = 0
        self.value_optimization_step = 0
        self.consumed_samples_values = 0
        self.consumed_samples = 0

        # TODO(geshen): steps probably wrong
        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )
        self.num_to_log_to_table = 5
        self.val_df = pd.DataFrame(columns=["step", "response", "reward", "ground_truth_answer"])
        self.train_df = pd.DataFrame(columns=["step", "response", "reward", "ground_truth_answer"])

    @torch.no_grad()
    def run_inference(self, dataloader):
        self.model.prepare_for_inference()

        total = 0
        num_correct = 0
        logged = 0

        tables = []

        limit_batches = compute_limit_batches(len(dataloader), self.cfg.limit_val_batches)

        loop_iter = zip(range(limit_batches), dataloader)
        inference_pbar = tqdm(loop_iter, total=min(len(dataloader), limit_batches), leave=True, desc="Inference")

        for (_, batch) in inference_pbar:
            output = self.model.generate(batch["question"])

            for response, answer in zip(output["sentences"], batch["answer"], strict=True):
                score = self.feedback.score(response, answer)

                if score > 0:
                    num_correct += 1

                if logged < self.num_to_log_to_table:
                    table = {}
                    table["reward"] = score
                    table["response"] = response
                    table["ground_truth_answer"] = answer

                    tables.append(table)
                    logged += 1

                total += 1

        self.model.finish_inference()

        metric_output = torch.as_tensor([num_correct, total], dtype=torch.long, device=torch.cuda.current_device())
        torch.distributed.all_reduce(metric_output, group=parallel_state.get_data_parallel_group())
        num_correct, total = metric_output.tolist()

        return {
            "global_total": total,
            "global_correct": num_correct,
            "global_accuracy": num_correct / total if total > 0 else 0,
            "table": tables,
        }

    def val_single_step(self, batch, train_mode):
        batch["train_mode"] = train_mode
        loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=True)
        metrics.update({"loss": loss_mean})

        return metrics

    def train_single_step(self, batch, train_mode):
        batch["train_mode"] = train_mode

        self.model.prepare_for_training_step()
        loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=False)
        self.model.finish_training_step()

        metrics.update({"loss": loss_mean})

        return metrics

    @torch.no_grad()
    def run_loss_val(self):
        self.model.prepare_for_inference()

        value_losses, policy_losses = [], []

        num_policy_batches = min(self.cfg.num_policy_batches, len(self.val_policy_dataloader))
        policy_batches = [output[1] for output in zip(range(num_policy_batches), self.val_policy_dataloader)]

        for batch in policy_batches:
            batch["amount_of_batches"] = len(policy_batches)
            policy_metrics = self.val_single_step(batch, TrainMode.POLICY_ONLY)
            policy_losses.append(policy_metrics["loss"])

        num_value_batches = min(self.cfg.num_value_batches, len(self.val_value_dataloader))
        value_batches = [output[1] for output in zip(range(num_value_batches), self.val_value_dataloader)]

        for batch in value_batches:
            batch["amount_of_batches"] = len(value_batches)
            value_metrics = self.val_single_step(batch, TrainMode.VALUE_ONLY)
            value_losses.append(value_metrics["loss"])

        metrics = {}

        value_loss = sum(value_losses)
        policy_loss = sum(policy_losses)

        metrics.update({"value_loss": value_loss})
        metrics.update({"policy_loss": policy_loss})
        metrics.update({"total_loss": value_loss + policy_loss})

        self.logger.log_metrics(
            metrics, step=self.step, prefix="search_eval/",
        )

        self.model.finish_inference()
        return metrics

    def run_training(self, policy_dataloader_iter, value_dataloader_iter):
        self.model.prepare_for_training()
        self.optimizer.zero_grad()
        dp_size = parallel_state.get_data_parallel_world_size()
        # TODO: add consumed samples logic
        # TODO: add optimizer step bump
        run_value = self.value_steps.pop()
        run_policy = self.policy_steps.pop()
        value_losses, policy_losses = [], []

        if run_policy:
            policy_batches = [output[1] for output in zip(range(self.cfg.num_policy_batches), policy_dataloader_iter)]

            for batch in policy_batches:
                # at least if we do this the lr are not synced between the 2 stages of training
                batch["amount_of_batches"] = len(policy_batches)
                policy_metrics = self.train_single_step(batch, TrainMode.POLICY_ONLY)
                policy_losses.append(policy_metrics["loss"])

                self.consumed_samples += batch["tokens"].size(0) * dp_size

            self.policy_optimization_step += 1

        if run_value:
            value_batches = [output[1] for output in zip(range(self.cfg.num_value_batches), value_dataloader_iter)]

            for batch in value_batches:
                # at least if we do this the lr are not synced between the 2 stages of training
                batch["amount_of_batches"] = len(value_batches)
                value_metrics = self.train_single_step(batch, TrainMode.VALUE_ONLY)
                value_losses.append(value_metrics["loss"])

                self.consumed_samples_values += self.model.cfg.critic_global_batch_size

            self.value_optimization_step += 1

        metrics = {}

        grad_norm = clip_optimier_gradients(self.model, self.optimizer, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        lr = self.optimizer.param_groups[0]["lr"]
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        metrics.update({"policy_optimization_step": self.policy_optimization_step})
        metrics.update({"value_optimization_step": self.value_optimization_step})
        metrics.update({"consumed_samples_values": self.consumed_samples_values})
        metrics.update({"consumed_samples": self.consumed_samples})
        metrics.update({"value_loss": sum(value_losses)})
        metrics.update({"policy_loss": sum(policy_losses)})
        metrics.update({"lr": lr})

        self.logger.log_metrics(
            metrics, step=self.step, prefix="train_optim/",
        )

        self.optimizer.step()
        self.scheduler.step()
        self.model.finish_training()
        return metrics

    def run_validation(self):
        self.timer.start("validation_time")

        dataloader = self.val_dataloader_builder_func()
        val_metrics = self.run_inference(dataloader)

        self.timer.stop("validation_time")
        val_metrics["validation_time"] = self.timer.get("validation_time")

        val_tables = val_metrics.pop("table")

        for table in val_tables:
            self.val_df.loc[len(self.val_df)] = [
                self.step,
                table["response"],
                table["reward"],
                table["ground_truth_answer"],
            ]

        self.logger.log_table("table/val", dataframe=self.val_df, step=self.step)
        self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")

        return val_metrics

    def run_train_evaluation(self):
        self.timer.start("train_eval")

        dataloader = self.train_dataloader_builder_func()
        train_metrics = self.run_inference(dataloader)

        self.timer.stop("train_eval")
        train_metrics["train_eval_timing"] = self.timer.get("train_eval")

        train_tables = train_metrics.pop("table")

        for table in train_tables:
            self.train_df.loc[len(self.train_df)] = [
                self.step,
                table["response"],
                table["reward"],
                table["ground_truth_answer"],
            ]

        self.logger.log_table("table/train_eval", dataframe=self.train_df, step=self.step)
        self.logger.log_metrics(train_metrics, step=self.step, prefix="train_eval/")

        return train_metrics

    def fit(self):
        self.run_timer.start_time()

        epoch_iter = range(self.epoch, self.cfg.max_epochs)

        if self.step == 0:
            self.run_validation()
            self.run_train_evaluation()

        for e in epoch_iter:
            # TODO(geshen): make sure to shuffle every epoch

            # max step is set accordingly because the dataloader
            # gets consumed each step
            loop_iter = range(self.max_steps)

            policy_dataloader_iter = iter(self.train_policy_dataloader)
            value_dataloader_iter = iter(self.train_value_dataloader)

            global_pbar = tqdm(loop_iter, total=self.max_steps, leave=True, desc=f"DeepSearch Epoch {e}",)

            inner_step = 0
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                self.timer.start("train_time")
                train_metrics = self.run_training(policy_dataloader_iter, value_dataloader_iter)
                self.timer.stop("train_time")
                timing_metrics["train_time"] = self.timer.get("train_time")

                self.step += 1
                inner_step += 1

                if inner_step == self.max_steps:
                    self.epoch += 1

                run_time_exceeded = self.run_timer.is_finished()

                run_val, save_model, is_train_end = check_progress(
                    inner_step,
                    self.max_steps,
                    self.cfg.val_check_interval,
                    self.cfg.save_interval,
                    1.0,
                    epoch=self.epoch,
                    max_epochs=self.cfg.max_epochs,
                    run_time_exceeded=run_time_exceeded,
                )

                if run_val:
                    val_metrics = self.run_validation()
                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                    train_eval_metrics = self.run_train_evaluation()
                    step_metrics.update({f"train_eval_{k}": v for k, v in train_eval_metrics.items()})

                    loss_eval_metrics = self.run_loss_val()
                    step_metrics.update({f"search_eval_{k}": v for k, v in loss_eval_metrics.items()})

                step_metrics.update(timing_metrics)
                step_metrics["epoch"] = self.epoch
                self.logger.log_metrics(timing_metrics | {"epoch": self.epoch}, step=self.step, prefix="timers/")

                global_pbar.set_postfix(step_metrics)

                if save_model:
                    step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                    self.save(step_metrics, is_train_end=is_train_end)

                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

            # when we resume we will load back the dataloader, which can have less max steps than 1 full epoch
            # so then in the next epoch the max steps is wrong, so we need to reset this
            self.set_max_steps()

    def set_max_steps(self):
        max_steps = self.cfg.get("max_steps", -1)

        # each step is one run of policy/value
        max_steps_policy = ceil(safe_divide(len(self.train_policy_dataloader), self.cfg.num_policy_batches))
        max_steps_value = ceil(safe_divide(len(self.train_value_dataloader), self.cfg.num_value_batches))

        dataloader_max_steps = max(max_steps_policy, max_steps_value)

        if max_steps == -1:
            # the dataloader already knows how much longer
            # because consumed samples is resumed
            max_steps = dataloader_max_steps
        else:
            raise NotImplementedError("manual max step doesn't work right now")

        self.max_steps = min(max_steps, dataloader_max_steps)

        # TODO(geshen): this slightly changes how things go when we restore
        # so it's not perfectly the same whenevr we get preempted
        value_steps = [True] * max_steps_value + [False] * (max_steps - max_steps_value)
        policy_steps = [True] * max_steps_policy + [False] * (max_steps - max_steps_policy)

        g = random.Random(self.epoch)
        g.shuffle(value_steps)
        g.shuffle(policy_steps)

        self.value_steps = value_steps
        self.policy_steps = policy_steps

    def state_dict(self):
        # TODO(geshen): add epoch logic
        return {
            "step": self.step,
            "epoch": self.epoch,
            "consumed_samples": self.consumed_samples,
            "consumed_samples_values": self.consumed_samples_values,
            "policy_optimization_step": self.policy_optimization_step,
            "value_optimization_step": self.value_optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.consumed_samples_values = state_dict["consumed_samples_values"]
        self.epoch = state_dict["epoch"]
        self.policy_optimization_step = state_dict["policy_optimization_step"]
        self.value_optimization_step = state_dict["value_optimization_step"]

        current_state_dict = self.state_dict()
        state_vals = [current_state_dict[k] for k in sorted(current_state_dict)]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(state_vals, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert state_vals == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def save(self, extra_candidates=None, is_train_end=False):
        """PTL based save"""
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)
