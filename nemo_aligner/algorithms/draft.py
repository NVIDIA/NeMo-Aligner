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
import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
# from nemo_aligner.utils.utils import get_iterator_k_split_list
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.train_utils import clip_gradients

class DraftTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        ckpt_callback,
    ):  
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = 0
        self.epoch = 0
        self.consumed_samples = 0

        self.ckpt_callback = ckpt_callback

        # used to compute the max step
        self._train_dataloader_len = len(train_dataloader)
        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )


    def run_training(self, batch):
        
        # data_iter = get_iterator_k_split_list(dataloader_iter, get_num_microbatches())
        self.model.prepare_for_training()

        self.optimizer.zero_grad()

        # must be provided by the RM
        # the RM is responsible for broadcasting the loss_mean as well
        # the loss mean and metrics must be on rank 0 and on the CPU

        # NOTE: assume backward is called on the loss already

        self.model.prepare_for_training_step()
        loss_mean, metrics = self.model.get_loss_and_metrics(data_iter=batch, forward_only=False)
        self.model.finish_training_step()

        grad_norm = clip_gradients(self.model.megatron_model, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        lr = self.optimizer.param_groups[0]["lr"]

        metrics.update({"lr": lr, "grad_norm": grad_norm, 
                        "rewards": -loss_mean, "kl_penalty": metrics["kl_penalty"]})

        self.optimizer.step()
        self.scheduler.step()

        self.model.finish_training()
        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()

        return loss_mean, metrics


    def fit(self):

        epoch_iter = range(self.epoch, self.cfg.max_epochs)
        
        if len(epoch_iter) <= 0:
            # epoch done
            return

        for _ in epoch_iter:

            global_pbar = tqdm(self.train_dataloader, initial=self.step, total=self.max_steps, leave=True, desc="DRaFT Step")

            for dataloader_iter in global_pbar:

                step_metrics = {}
                timing_metrics = {}

                self.timer.start("train_step_time")
                loss, metrics = self.run_training(dataloader_iter)
                self.timer.stop("train_step_time")

                timing_metrics["train_step_time"] = self.timer.get("train_step_time")

                self.step += 1

                self.model.logger.log_metrics(timing_metrics, step=self.step, prefix="timers/")
                

                metrics = metrics | {"epoch":self.epoch, "training_steps":self.step}

                # TODO(geshen): maybe use the dataloader instead
                self.consumed_samples += self.cfg.global_batch_size
 
                global_pbar.set_postfix(metrics)
                self.model.logger.log_metrics(
                    metrics, step=self.step, prefix="train_metrics/",
                )
                monitor_candidates = {
                    "reduced_train_loss": torch.tensor(loss, dtype=torch.float32),
                }

                step_metrics.update(timing_metrics)
                step_metrics.update({f"train_{k}": v for k, v in metrics.items()})
                global_pbar.set_postfix(step_metrics)

                step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}

                if self.step % self.cfg.save_interval == 0:
                    step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                    self.save(step_metrics, is_train_end=False)
     
            self.epoch += 1
        
        self.save(
        monitor_candidates, is_train_end=True
        )

        self.model.logger.finalize()

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.epoch = state_dict["epoch"]

        loaded_values = [self.step, self.consumed_samples, self.epoch]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def save(self, extra_candidates=None, is_train_end=False):
        self.model.prepare_for_training()
        # load back in the adam states if needed
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        self.model.finish_training()
    
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

