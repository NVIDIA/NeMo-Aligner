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

from functools import partial
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
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress
from nemo_aligner.utils.utils import clear_memory


def mcts_collate_fn(eos_id, batch):

    hist_action_probs = torch.stack(tuple(torch.as_tensor(x["hist_action_probs"]) for x in batch))
    hist_actions = torch.stack(tuple(torch.as_tensor(x["actions"]) for x in batch))
    hist_outcome = torch.stack(tuple(torch.as_tensor(x["hist_outcome"]) for x in batch))
    data_id = torch.stack(tuple(torch.as_tensor(x["data_id"]) for x in batch))

    tokens, context_lengths = map(torch.as_tensor, pad_batch([x["tokens"] for x in batch], eos_id, max_len=0))

    return {
        "tokens": tokens,
        "context_lengths": context_lengths,
        "actions": hist_actions,
        "action_probs": hist_action_probs,
        "reward": hist_outcome,
        "data_id": data_id,
    }


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
        train_dataloader,
        val_dataloader,
        feedback,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.feedback = feedback
        self.logger = logger
        self.ckpt_callback = ckpt_callback
        self.run_timer = run_timer

        self.step = 0
        self.epoch = 0
        self.optimization_step = 0
        self.consumed_samples = 0

        self._train_dataloader_len = len(train_dataloader)
        self.set_max_steps()

        self.limit_val_batches = compute_limit_batches(len(val_dataloader), self.cfg.limit_val_batches)
        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        total = 0
        num_correct = 0
        logged = False

        for _, batch in zip(range(self.limit_val_batches), self.val_dataloader):
            output = self.model.generate(batch["question"])

            for response, answer in zip(output["sentences"], batch["answer"]):
                score = self.feedback.score(response, answer)
                num_correct += score

                if torch.distributed.get_rank() == 0 and not logged:
                    logged = True
                    print("### GENERATED RESPONSE", response)
                    print("### ACTUAL ANSWER", answer)
                    print("### SCORE", score)

            total += len(batch["question"])

        self.model.finish_inference()

        return {"accuracy": num_correct / total if total > 0 else 0}

    def run_training(self, dataloader_iter, num_batches_to_use):
        self.model.prepare_for_training()

        for _, batch in zip(range(num_batches_to_use), dataloader_iter):
            self.optimizer.zero_grad()

            self.model.prepare_for_training_step()
            loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=False)
            self.model.finish_training_step()

            grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
            grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.step()
            self.scheduler.step()

            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm

            metrics.update({"lr": lr, "loss": loss_mean, "optim_step": self.optimization_step})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.optimization_step += 1

        self.model.finish_training()

        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()
        return loss_mean, metrics

    @torch.no_grad()
    def run_search(self, dataloader_iter):
        self.model.prepare_for_inference()

        output = []

        num_total = 0
        num_correct = 0

        # train dataloader loads gbs
        for _, batch in zip(range(1), dataloader_iter):
            output = run_mcts(batch, self.model)
            self.consumed_samples += len(batch["question"]) * parallel_state.get_data_parallel_world_size()

            for o in output:
                if o["hist_outcome"] > 0:
                    num_correct += 1

            num_total += len(output)

        # find how many passed
        metric_output = torch.as_tensor([num_correct, num_total], dtype=torch.long, device=torch.cuda.current_device())
        torch.distributed.all_reduce(metric_output, group=parallel_state.get_data_parallel_group())
        num_correct, num_total = metric_output.tolist()

        num_to_load_per_dp = divide(self.model.cfg.global_batch_size, parallel_state.get_data_parallel_world_size())

        dataloader = torch.utils.data.DataLoader(
            output,
            batch_size=num_to_load_per_dp,
            shuffle=False,  # TODO(geshen): turn this on
            num_workers=0,
            drop_last=True,
            collate_fn=partial(mcts_collate_fn, self.model.tokenizer.eos_id),
        )

        # mcts will return different lengths depending on the prompt
        # so we need to make sure to use the smallest DP rank's dataloader to not hang training
        output = torch.as_tensor([len(dataloader)], dtype=torch.long, device=torch.cuda.current_device())
        torch.distributed.all_reduce(
            output, op=torch.distributed.ReduceOp.MIN, group=parallel_state.get_data_parallel_group()
        )
        num_batches_to_use = output.item()

        self.model.finish_inference()
        return dataloader, num_batches_to_use, {"accuracy": num_correct / num_total if num_total > 0 else 0}

    def fit(self):
        self.run_timer.start_time()

        if self.cfg.max_epochs is not None and self.cfg.max_epochs > 1:
            # because we need to figure out a nice way to reset the shuffling on our dataset
            # otherwise epoch > 1 will loop over the dataset in the same order
            raise ValueError("epoch > 1 is not supported")

        epoch_iter = range(self.epoch, self.cfg.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        for _ in epoch_iter:
            # TODO(geshen): make sure to shuffle every epoch
            loop_iter = range(self.step, self.max_steps)

            # TODO(geshen): to change for when we support > 1 epoch
            if len(loop_iter) <= 0:
                return  # training ended

            dataloader_iter = iter(self.train_dataloader)

            global_pbar = tqdm(
                loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="DeepSearch Global Step"
            )

            dataloader_iter = iter(self.train_dataloader)

            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                self.timer.start("search_time")
                data_iter, num_batches_to_use, search_metrics = self.run_search(dataloader_iter)
                self.timer.stop("search_time")
                timing_metrics["search_time"] = self.timer.get("search_time")
                step_metrics.update({f"search_{k}": v for k, v in search_metrics.items()})

                clear_memory()
                self.timer.start("train_time")
                loss_mean, train_metrics = self.run_training(data_iter, num_batches_to_use)
                self.timer.stop("train_time")
                timing_metrics["train_time"] = self.timer.get("train_time")

                run_time_exceeded = self.run_timer.is_finished()
                run_val, save_model, is_train_end = check_progress(
                    self.step,
                    self.max_steps,
                    self.cfg.val_check_interval,
                    self.cfg.save_interval,
                    1.0,  # TODO:(geshen): allow for limit val batches
                    run_time_exceeded=run_time_exceeded,
                )

                self.step += 1

                if run_val:
                    self.timer.start("validation_time")
                    val_metrics = self.run_validation()
                    self.timer.stop("validation_time")
                    timing_metrics["validation_time"] = self.timer.get("validation_time")

                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                step_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})

                global_pbar.set_postfix(step_metrics)

                if save_model:
                    step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                    self.save(step_metrics, is_train_end=is_train_end)

                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

            self.epoch += 1

    def set_max_steps(self):
        max_steps = self.cfg.get("max_steps", -1)

        if max_steps == -1:
            # the dataloader already knows how much longer
            # because consumed samples is resumed
            max_steps = self._train_dataloader_len
        else:
            # user specified the max step, figure out how much longer
            # we need to run for
            max_steps = max_steps - self.step

        self.max_steps = min(max_steps, self._train_dataloader_len) + self.step

    def state_dict(self):
        # TODO(geshen): add epoch logic
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.optimization_step = state_dict["optimization_step"]

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
