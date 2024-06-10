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

import threading
import time
from typing import Dict

import numpy as np
import torch
from megatron.core.utils import divide
from pytriton.decorators import batch, sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig
from tqdm import tqdm

from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.servers.constants import ServerSignal
from nemo_aligner.servers.server_callables import process_inference_request
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import SyncTimer, broadcast_2d_tensor, rebalance_nd_tensor
from nemo_aligner.utils.server_utils import lock_method, pad_input
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.utils import apply_func_to_dict

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


class CriticServerTrainer:
    r"""Class that implements the critic training via PyTriton requests.
        There are 3 things the server does
            1. training
            2. inference for the critic(and maybe the reward model)
            3. saving the critic

        It starts a PyTriton server on rank 0, and rank 0 will tell other
        ranks what to do
    """

    def __init__(self, cfg, model, optimizer, scheduler, logger, ckpt_callback):
        self.lock = threading.Lock()
        self.logger = logger
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_callback = ckpt_callback
        self.gbs = cfg.gbs
        self.forward_mbs = cfg.forward_mbs
        self.step = 0
        self.pad_sequence_length_to_multiple = cfg.pad_sequence_length_to_multiple

        self.timestamp = time.time()

        # server parameters
        self.combine_rm_and_critic_server = cfg.combine_rm_and_critic_server
        self.infer_fn = model.infer_rm_critic if self.combine_rm_and_critic_server else model.infer
        self.port = cfg.port
        self.max_inference_batch_size = cfg.inference_micro_batch_size * parallel_state.get_data_parallel_world_size()

        # PyTriton args
        self.infer_inputs = (
            Tensor(name="sentences", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="tokens", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="sequence_lengths", shape=(-1,), dtype=np.int64, optional=True),
            Tensor(name="add_EOS", shape=(1,), dtype=np.bool_, optional=True),
        )
        self.infer_outputs = [
            Tensor(name="values", shape=(-1,), dtype=np.float32),
        ]
        if self.combine_rm_and_critic_server:
            self.infer_outputs.append(Tensor(name="rewards", shape=(-1,), dtype=np.float32))

        self.train_inputs = (
            Tensor(name="tokens", shape=(-1, -1,), dtype=np.int64, optional=False),
            Tensor(name="returns", shape=(-1, -1,), dtype=np.float32, optional=False),
            Tensor(name="prev_values", shape=(-1, -1,), dtype=np.float32, optional=False),
            Tensor(name="mask", shape=(-1, -1,), dtype=np.float32, optional=False),
        )
        self.train_outputs = (Tensor(name="loss_mean", shape=(1,), dtype=np.float32),)

        self.save_inputs = (Tensor(name="dummy_var", shape=(1,), dtype=np.int64),)
        self.save_outputs = (Tensor(name="status", shape=(1,), dtype=np.int64),)

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

    @batch
    @lock_method("self.lock")
    def server_infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        # tell other ranks to start inference
        choice = ServerSignal.FORWARD.cuda()
        torch.distributed.broadcast(choice, 0)

        print(
            "### RUNNING INFERENCE ON BATCH SIZE on step: {} batch size {}".format(
                self.step, inputs["tokens"].shape[0]
            )
        )
        start_time = time.time()
        inputs, extra, prepad_sequence_length = process_inference_request(
            inputs,
            pad_to=self.forward_mbs * parallel_state.get_data_parallel_world_size(),
            pad_sequence_length_to_multiple=self.pad_sequence_length_to_multiple,
        )
        rewards, values = self.run_inference(inputs=inputs, extra=extra)

        if prepad_sequence_length > values.shape[1]:
            values = np.pad(
                values, ((0, 0), (0, prepad_sequence_length - values.shape[1])), mode="constant", constant_values=0
            )
        else:
            values = values[:, :prepad_sequence_length]

        end_time = time.time()
        print("#### INFER TOOK", end_time - start_time)

        output = {
            "values": values,
        }
        if self.combine_rm_and_critic_server:
            output["rewards"] = rewards[:, None]

        return {k: v[: v.shape[0] - extra] for k, v in output.items()}

    @sample
    @lock_method("self.lock")
    def server_save(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        # tell other ranks to start inference
        choice = ServerSignal.SAVE.cuda()
        torch.distributed.broadcast(choice, 0)
        self.save()
        return {"status": np.array((0,), dtype=np.int32)}

    @sample
    @lock_method("self.lock")
    def server_train(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        tokens = inputs.pop("tokens", None)
        returns = inputs.pop("returns", None)
        prev_values = inputs.pop("prev_values", None)
        mask = inputs.pop("mask", None)

        # we should pad to GBS
        tokens, extra_tokens = pad_input(tokens, self.gbs)
        returns, extra_returns = pad_input(returns, self.gbs)
        prev_values, extra_values = pad_input(prev_values, self.gbs)

        # have to set the pad value to 0, so the masked mean in the loss will
        # have no effect for the padded batch
        mask, extra_mask = pad_input(mask, self.gbs, pad_value=0)

        assert extra_tokens == extra_returns == extra_values == extra_mask

        batch = {
            "tokens": tokens,
            "returns": returns,
            "prev_values": prev_values,
            "mask": mask,
        }

        batch = apply_func_to_dict(torch.tensor, batch)

        choice = ServerSignal.TRAIN.cuda()
        torch.distributed.broadcast(choice, 0)

        loss_mean = self.run_training(**batch)
        return {"loss_mean": np.array((loss_mean,))}

    def run_server(self):
        if torch.distributed.get_rank() == 0:
            triton_config = TritonConfig(
                allow_http=True,
                allow_grpc=False,
                allow_metrics=False,
                http_address=ENDPOINT_BIND_ADDRESS,
                http_port=self.port,
            )
            # try to find a common multiple of forward mbs and dp so we don't need to pad
            dp_size = parallel_state.get_data_parallel_world_size()
            preferred_batch_size = [dp_size * self.forward_mbs * (i + 1) for i in range(1000)]

            # 1 second latency max
            dynamic_batcher = DynamicBatcher(
                max_queue_delay_microseconds=self.cfg.max_queue_delay_microseconds,
                preferred_batch_size=preferred_batch_size,
            )

            # we cut the batch into pieces so we don't need to have a max batch size
            infer_model_config = ModelConfig(batching=True, max_batch_size=9999999, batcher=dynamic_batcher)
            # the model will split the train batch by itself
            train_model_config = ModelConfig(batching=False, max_batch_size=0, batcher=None)
            save_model_config = ModelConfig(batching=False, max_batch_size=0, batcher=None)

            with Triton(config=triton_config) as triton:
                triton.bind(
                    model_name="critic_infer",
                    infer_func=self.server_infer,
                    inputs=self.infer_inputs,
                    outputs=self.infer_outputs,
                    config=infer_model_config,
                )
                triton.bind(
                    model_name="critic_train",
                    infer_func=self.server_train,
                    inputs=self.train_inputs,
                    outputs=self.train_outputs,
                    config=train_model_config,
                )
                triton.bind(
                    model_name="critic_save",
                    infer_func=self.server_save,
                    inputs=self.save_inputs,
                    outputs=self.save_outputs,
                    config=save_model_config,
                )
                triton.serve()

        else:
            self.run_subscriber_loop()

    def run_subscriber_loop(self):
        while True:
            command = ServerSignal.INVALID.cuda()
            torch.distributed.broadcast(command, 0)
            op = command.item()

            if op == ServerSignal.FORWARD:
                self.run_inference()
            elif op == ServerSignal.TRAIN:
                self.run_training()
            elif op == ServerSignal.SAVE:
                self.save()
            else:
                raise RuntimeError(f"Invalid operation: {op}")

    @torch.no_grad()
    def run_inference(self, inputs=None, extra=None):
        """only rank 0 has valid data
        """
        print(f"----start infer at {time.time()}")
        self.model.prepare_for_inference()
        tokens, lengths = None, None
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        is_rank_0 = torch.distributed.get_rank() == 0

        if is_rank_0:
            tokens = torch.as_tensor(inputs["inputs"], dtype=torch.long, device=torch.cuda.current_device())
            lengths = torch.as_tensor(inputs["sequence_length"], dtype=torch.long, device=torch.cuda.current_device())

        tokens = broadcast_2d_tensor(tokens, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank]
        lengths = broadcast_2d_tensor(lengths, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank].squeeze(-1)

        outputs = self.infer_fn(inputs=(tokens, lengths))

        if self.combine_rm_and_critic_server:
            rewards, values = outputs
            rewards = (
                rebalance_nd_tensor(rewards, group=parallel_state.get_data_parallel_group())
                .squeeze(dim=(1,))
                .cpu()
                .numpy()
            )

        else:
            values = outputs
            rewards = None

        values = rebalance_nd_tensor(values, group=parallel_state.get_data_parallel_group()).cpu().numpy()

        self.model.finish_inference()
        torch.distributed.barrier()
        return rewards, values

    def run_training(self, tokens=None, returns=None, prev_values=None, mask=None):
        print(f"-----starting training {time.time()}--------")

        """assume that the batch is already padded
        """
        # broadcast to every rank and then split out the tensor after
        batch = {
            "tokens": tokens,
            "returns": returns,
            "prev_values": prev_values,
            "mask": mask,
        }

        batch["tokens"] = broadcast_2d_tensor(batch["tokens"], src=0, group=None, dtype=torch.int64)
        batch["returns"] = broadcast_2d_tensor(batch["returns"], src=0, group=None, dtype=torch.float32)
        batch["prev_values"] = broadcast_2d_tensor(batch["prev_values"], src=0, group=None, dtype=torch.float32)
        batch["mask"] = broadcast_2d_tensor(batch["mask"], src=0, group=None, dtype=torch.float32)
        input_size = batch["tokens"].size(0)

        self.model.prepare_for_training()

        num_gbs = divide(input_size, self.gbs)

        # split the input into global batches
        gbs_iterator = get_iterator_k_split(batch, num_gbs)

        global_pbar = tqdm(gbs_iterator, total=num_gbs, leave=True, desc="Training steps")

        for gbs in global_pbar:
            # get the batch we need to process for DP
            dp_batch = list(get_iterator_k_split(gbs, parallel_state.get_data_parallel_world_size()))[
                parallel_state.get_data_parallel_rank()
            ]

            self.model.prepare_for_training_step()
            self.optimizer.zero_grad()

            self.timer.start("train_step_time")
            loss_mean, metrics = self.model.get_loss_and_metrics(dp_batch, forward_only=False)
            self.timer.stop("train_step_time")
            train_step_time = self.timer.get("train_step_time")
            metrics["step_time"] = train_step_time

            self.model.finish_training_step()

            grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
            grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.step()
            self.scheduler.step()

            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm

            metrics.update({"lr": lr, "loss": loss_mean})

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train/",
            )
            global_pbar.set_postfix(metrics)

            self.step += 1

        self.model.finish_training()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(f"-----finishing training {time.time()}--------")

        return loss_mean

    def save(self, extra_candidates=None, is_train_end=False, save_top_only=False):
        """PTL based save"""
        # when using adam offloading, need to load back the adam states
        # so need to call prepare for training
        self.model.prepare_for_training()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)

        logging.info(f"saving checkpoint at step {self.step}")
        self.ckpt_callback.custom_save(
            monitor_candidates=monitor_candidates, is_train_end=is_train_end, save_top_only=save_top_only
        )

        # make sure everyone is done saving
        torch.distributed.barrier()
        self.model.finish_training()

    def state_dict(self):
        return {
            "step": self.step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]

        loaded_values = [self.step]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)
        assert loaded_values == to_broadcast.tolist()
