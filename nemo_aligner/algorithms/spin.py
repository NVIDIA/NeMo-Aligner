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

from collections import defaultdict
from statistics import mean

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.ppo_utils import create_mask
from nemo_aligner.utils.text_generation_utils import TrackLengthGPTModelTextGenerationStrategy
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import (
    batch_pad_to_fixed_len,
    clear_memory,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)

"""
GPTSFTChatDataset output is dict with keys: ['input_ids', 'mask', 'context_ids', 'answer_ids', 'metadata']

input_ids: torch.LongTensor - the entire prompt + response, including the system preamble which is specified by "system" in the jsonl
mask: torch.BoolTensor with False for the preamble+prompt, and True for the response
context_ids: torch.LongTensor - the entire preamble + prompt
answer_ids: torch.LongTensor - the entire response only
metadata: dict - with keys "system" for the preamble, and "mask" which is "User" or "Assistant"
"""


def spin_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    input_ids = [item["input_ids"] for item in batch]
    masks = [item["mask"] for item in batch]
    context_ids = [item["context_ids"] for item in batch]
    answer_ids = [item["answer_ids"] for item in batch]
    context_lengths = torch.LongTensor([len(x) for x in context_ids])
    combined_lengths = torch.LongTensor([len(x) for x in input_ids])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=eos_id)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    context_ids = torch.nn.utils.rnn.pad_sequence(context_ids, batch_first=True, padding_value=eos_id)
    answer_ids = torch.nn.utils.rnn.pad_sequence(answer_ids, batch_first=True, padding_value=eos_id)

    output = {
        "prompts_and_answers": input_ids,
        "masks": masks,
        "prompts_only": context_ids,
        "answers_only": answer_ids,
        "prompt_lengths": context_lengths,
        "combined_lengths": combined_lengths,
    }

    return output


class SPINTrainer:
    """Trainer to coordinate SPIN SFT training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.step = 0
        self.consumed_samples = 0

        self.ckpt_callback = ckpt_callback

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(self.train_dataloader.batch_sampler)
        if (limit_train_batches := self.cfg.get("limit_train_batches")) is not None and limit_train_batches >= 0:
            self.num_steps_per_epoch = min(self.num_steps_per_epoch, limit_train_batches)

        self.limit_val_batches = compute_limit_batches(len(val_dataloader), self.cfg.limit_val_batches)
        self.val_check_interval = (
            int(self.cfg.val_check_interval * self._train_dataloader_len)
            if isinstance(self.cfg.val_check_interval, float)
            else self.cfg.val_check_interval
        )
        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        self.length_params = OmegaConf.to_container(self.model.cfg.spin.length_params, resolve=True)
        self.sampling_params = OmegaConf.to_container(self.model.cfg.spin.sampling_params, resolve=True)
        self.max_gen_seq_len = self.length_params["max_length"]

    def validation_step(self, global_batch):
        # these things should go into a GPTModel wrapper
        self.model.prepare_for_validation_step()

        loss_mean, metrics = self.model.get_loss_and_metrics_vanilla_sft(batch=global_batch, forward_only=True)

        self.model.finish_validation_step()
        return loss_mean, metrics

    @torch.no_grad()
    def run_validation(self):
        loss_means = []
        val_metrics = defaultdict(list)

        val_pbar = tqdm(
            zip(range(self.limit_val_batches), self.val_dataloader),
            total=self.limit_val_batches,
            leave=True,
            desc="Validation steps",
        )

        for _, batch in val_pbar:
            # self.model.prepare_for_validation()

            self.timer.start("validation_step_time")
            loss_mean, metrics = self.validation_step(batch)
            self.timer.stop("validation_step_time")
            validation_step_time = self.timer.get("validation_step_time")

            metrics["validation_step_time"] = validation_step_time

            loss_means.append(loss_mean)
            for k, v in metrics.items():
                val_metrics[k].append(v)
            log_val_metrics = {f"val_{k}": v for k, v in metrics.items()}
            val_pbar.set_postfix(log_val_metrics)

            # self.model.finish_validation()

        val_metrics = {k: mean(v) for k, v in val_metrics.items()}
        return mean(loss_means), val_metrics

    def train_single_step(self, global_batch):
        self.optimizer.zero_grad()

        self.model.prepare_for_training_step()

        # NOTE: assume backward is called on the loss already
        loss_mean, metrics = self.model.get_loss_and_metrics(batch=global_batch, forward_only=False)

        self.model.finish_training_step()

        grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()
        self.scheduler.step()

        trainer_metrics = {}
        if grad_norm is not None:
            trainer_metrics["grad_norm"] = grad_norm
        trainer_metrics.update({"lr": lr, "loss": loss_mean})

        return loss_mean, {**metrics, **trainer_metrics}

    @torch.no_grad()
    def get_generations(self, batch):
        self.model.prepare_for_inference()

        prompt_lengths = batch["prompt_lengths"]
        batch_max_length = prompt_lengths.max().item()
        max_possible_length = min(self.model.cfg.encoder_seq_length, batch_max_length + self.max_gen_seq_len)

        prompt_tokens = batch_pad_to_fixed_len(
            batch["prompts_only"], max_possible_length, pad_token=self.model.tokenizer.eos_id
        )
        prompt_tokens = prompt_tokens.cuda(non_blocking=True)
        prompt_lengths = prompt_lengths.cuda(non_blocking=True)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self.model, context_lengths=prompt_lengths, max_length=self.max_gen_seq_len
        )
        generations = self.model.generate(
            inputs=(prompt_tokens, prompt_lengths),
            length_params=self.length_params,
            sampling_params=self.sampling_params,
            strategy=strategy,
        )

        # this is a 1D LongTensor with the length of the responses where response is prompt+response
        response_lengths = strategy.get_lengths().cpu()
        max_response_length = response_lengths.max().item()
        response_tokens = torch.LongTensor(generations["token_ids"]).cpu()

        # Sanity check to validate response length.
        if max_response_length != response_tokens.size(1):
            # This may actually happen because NeMo does not always stop generation after `max_length` in batch mode
            # => `response_tokens` may contain up to `max_length + max_context_length` tokens.
            # TODO once NeMo fixes this issue we should be able to always raise an exception when the check above fails,
            # and remove the `if` below.
            if (
                max_response_length >= response_tokens.size(1)
                or response_tokens.size(1) != prompt_lengths.max().item() + self.max_gen_seq_len
            ):
                raise AssertionError(
                    f"max response length ({max_response_length}) does not match the size of "
                    f"`response_tokens` ({response_tokens.size(1)})"
                )

        self.model.finish_inference()

        return response_tokens, response_lengths

    def fit(self):
        if (not isinstance(self.train_dataloader.batch_sampler, MegatronPretrainingRandomBatchSampler)) and (
            self.cfg.max_epochs is not None and self.cfg.max_epochs > 1
        ):
            # if you use MegatronPretrainingBatchSampler as the batch_sampler passed to your train dataloader (in builders.py)
            # then each epoch will repeat all your samples in the same order as the previous epoch, there is no shuffling
            # to fix this, you should use MegatronPretrainingRandomBatchSampler instead, which alleviates this issue and allows
            # random shuffling for each epoch.
            raise ValueError(
                "max_epochs > 1 is not supported unless using `MegatronPretrainingRandomBatchSampler` as the batch_sampler for your train dataloader"
            )

        self.run_timer.start_time()

        iterations_iter = range(self.iteration, self.cfg.max_iterations)
        if len(iterations_iter) <= 0:
            # iteration done
            return

        for _ in iterations_iter:
            epoch_iter = range(self.epoch, self.cfg.max_epochs)
            if len(epoch_iter) <= 0:
                # epoch done
                return

            # call this in case the model is using a KL scheduler based on iteration number
            self.model.set_KL_penalty_by_iteration(self.iteration)

            for _ in epoch_iter:
                num_steps_in_epoch = min(
                    self.max_steps - self.step, self.num_steps_per_epoch - self.step % self.num_steps_per_epoch
                )
                loop_iter = range(num_steps_in_epoch)

                if not loop_iter:
                    return  # training ended

                global_pbar = tqdm(
                    self.augment_dataloader(self.train_dataloader),
                    initial=self.step,
                    total=self.max_steps,
                    leave=True,
                    desc="Training steps",
                )

                for _, global_batch in zip(loop_iter, global_pbar):
                    self.model.prepare_for_training()

                    self.timer.start("train_step_time")
                    loss, metrics = self.train_single_step(global_batch)
                    self.timer.stop("train_step_time")
                    train_step_time = self.timer.get("train_step_time")
                    # to help avoid fragmentation
                    clear_memory()

                    # TODO(geshen): maybe use the dataloader instead
                    # bump up the consumed samples but not the step
                    self.consumed_samples += self.model.cfg.global_batch_size
                    metrics["consumed_samples"] = self.consumed_samples
                    metrics["step_time"] = train_step_time
                    metrics["epoch"] = self.epoch
                    metrics["iteration"] = self.iteration
                    self.logger.log_metrics(
                        metrics, step=self.step, prefix="train/",
                    )
                    metrics = {f"train_{k}": v for k, v in metrics.items()}

                    self.step += 1

                    run_time_exceeded = self.run_timer.is_finished()
                    run_val, save_model, is_train_end = check_progress(
                        self.step,
                        self.max_steps,
                        self.cfg.val_check_interval,
                        self.cfg.save_interval,
                        self.limit_val_batches,
                        run_time_exceeded=run_time_exceeded,
                    )

                    if run_val:
                        val_loss, val_metrics = self.run_validation()
                        # validation is done on the UPDATED weights
                        # so we use the incremented self.step
                        self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")
                        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                        metrics.update(val_metrics)

                    global_pbar.set_postfix(metrics)

                    if save_model:
                        # PTL save wants tensors only
                        metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}
                        self.save(metrics, is_train_end=is_train_end)

                    if run_time_exceeded:
                        logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                        return

                    metrics.clear()
                    self.model.finish_training()

            # update the reference policy weights
            self.model.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(
                self.model, megatron_amp_O2=self.model.cfg.get("megatron_amp_O2", False)
            )

        self.logger.finalize()

    def save(self, extra_candidates=None, is_train_end=False):
        # load back in the adam states if needed
        self.model.prepare_for_training()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        # we don't want to save the ref policy at the very end, although this prohibits continuation training from the .nemo file
        if is_train_end:
            self.model.ref_policy_state_dict = None

        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs * self.cfg.max_iterations

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]

        loaded_values = [self.step, self.consumed_samples]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def augment_dataloader(self, dataloader):
        """Augment dataloader with generations and ref policy log probs"""
        iter_dataloader = iter(dataloader)
        done = False
        while not done:
            try:
                batch = next(iter_dataloader)

                # generations use the reference model weights, as per the paper
                with cpu_weight_swap(
                    self.model, self.model.ref_policy_state_dict, megatron_amp_O2=self.model.megatron_amp_O2
                ):
                    # Generation happens on GPU but the returned tensors are on CPU.
                    gen_tokens, gen_lengths = self.get_generations(batch)
                act_tokens = batch["prompts_and_answers"]
                act_lengths = batch["combined_lengths"]
                max_batch_len = max(act_tokens.shape[1], gen_tokens.shape[1])

                act_tokens_pad = batch_pad_to_fixed_len(
                    act_tokens, max_batch_len, pad_token=self.model.tokenizer.eos_id
                )
                gen_tokens_pad = batch_pad_to_fixed_len(
                    gen_tokens, max_batch_len, pad_token=self.model.tokenizer.eos_id
                )

                act_mask = create_mask(act_tokens_pad, batch["prompt_lengths"], act_lengths)
                gen_mask = create_mask(gen_tokens_pad, batch["prompt_lengths"], gen_lengths)

                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    act_tokens_pad,
                    self.model.tokenizer.eos_id,
                    self.model.cfg.data.reset_position_ids,
                    self.model.cfg.data.reset_attention_mask,
                    self.model.cfg.data.eod_mask_loss,
                )
                assert attention_mask.ndim == 4, "attention_mask is incorrect shape"
                if attention_mask.shape[0] == 1:
                    # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
                    # attention_mask = attention_mask.expand(len(act_tokens_pad), *((-1,) * (len(attention_mask.shape) - 1)))
                    attention_mask = attention_mask.repeat(
                        len(act_tokens_pad), *((1,) * (len(attention_mask.shape) - 1))
                    )

                new_batch = {}
                new_batch["actual"] = act_tokens_pad
                new_batch["generated"] = gen_tokens_pad
                new_batch["attention_mask"] = attention_mask
                new_batch["position_ids"] = position_ids
                new_batch["actual_mask"] = act_mask
                new_batch["generated_mask"] = gen_mask

                logprobs = self.model.get_ref_policy_logprobs(new_batch).cpu()
                act_logps, gen_logps = torch.split(logprobs, len(logprobs) // 2, dim=0)

                new_batch["ref_policy_log_probs_actual"] = act_logps
                new_batch["ref_policy_log_probs_generated"] = gen_logps

                yield new_batch

                del logprobs, act_logps, gen_logps, new_batch
            except StopIteration:
                done = True

    @property
    def epoch(self):
        return (self.step // self.num_steps_per_epoch) % self.cfg.max_epochs

    @property
    def iteration(self):
        return (self.step // self.num_steps_per_epoch) // self.cfg.max_epochs
