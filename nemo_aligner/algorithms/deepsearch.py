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

import pandas as pd
import torch
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.utils.deep_search.mcts.feedback_functions import GSK8KFeedbackDataset
from nemo_aligner.utils.distributed import SyncTimer, masked_global_mean_var, normalize_tensor
from nemo_aligner.utils.server_utils import FutureResult
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress
from nemo_aligner.utils.utils import clear_memory, cpu_dict


def compute_num_rollout_microbatches(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )


steerlm_template="""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
{prompt}
Please show the calculation steps and lastly the final answer in format {{{{answer number}}}}
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2
"""

class DeepSearchTrainer:
    """Trainer to coordinate PPO training
    """

    def __init__(
        self, cfg: DictConfig, model, optimizer, scheduler, train_dataloader, logger, ckpt_callback,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.ckpt_callback = ckpt_callback

        self.step = 0
        self.epoch = 0

        self._train_dataloader_len = len(train_dataloader)
        self.set_max_steps()

    def run_regular_generation(self, val_ds):
        """val_ds is a huggingface ds
        """
        total_num = 8

        feedback = GSK8KFeedbackDataset(val_ds)
        self.model.prepare_for_inference()

        inputs = [steerlm_template.format(prompt=x) for x in val_ds['question'][:total_num]]
        output = self.model.generate(inputs)

        score = 0

        for i, item in enumerate(output['sentences']):
            score += feedback.score(item, i)

        accuracy = score / total_num

        self.model.finish_inference()

        return {"accuracy": accuracy, "text": output["sentences"]}

    def run_training(self, dataloader_iter):
# self.model.prepare_for_training()

        for batch in dataloader_iter:
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

            metrics.update({"lr": lr, "loss": loss_mean, "optim_step": self.ppo_optimization_step})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.ppo_optimization_step += 1

# self.model.finish_training()

        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()
        return loss_mean, metrics

    def run_search(self, dataloader_iter, num_rollout_micro_batches):
        # dataloader working but need to be able to run search properly
        # for data in dataloader_iter:
        # pretend this is the buffer
        return torch.load("/rlhf/batch.pt"), {}

    def fit(self):
        epoch_iter = range(self.epoch, self.cfg.max_epochs)

        for _ in epoch_iter:
            # TODO(geshen): make sure to shuffle every epoch
            loop_iter = range(self.step, self.max_steps)
            dataloader_iter = iter(self.train_dataloader)

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            num_rollout_micro_batches = compute_num_rollout_microbatches(self.train_dataloader)

            for _ in global_pbar:
                step_metrics = {}

                rollout_data, metrics = self.run_search(dataloader_iter, num_rollout_micro_batches)
                # clear_memory()

                loss_mean, train_metrics = self.run_training(rollout_data)

                self.step += 1

                if self.step % self.cfg.val_check_interval == 0:
                    val_metrics = self.run_regular_generation(self.train_dataloader.dataset)
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                step_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})

                global_pbar.set_postfix(step_metrics)

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
