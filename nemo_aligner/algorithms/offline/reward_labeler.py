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
from megatron.core import mpu

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import broadcast_2d_tensor

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RewardLabeler:
    def __init__(self, trainer, model, inference_cfg):
        # check whether the DDP is initialized
        if mpu.is_unitialized():

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        model.freeze()

        # Have to turn off activations_checkpoint_method for inference
        try:
            model.model.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass

        self.cfg = inference_cfg
        self.trainer = trainer
        self.model = model
        self.fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)

        # Reward standardization mean and std
        self.enable_standardization = inference_cfg.reward_standardization.enable
        self.rew_mean = inference_cfg.reward_standardization.mean
        self.rew_std = inference_cfg.reward_standardization.std

    def predict(self, inputs):
        assert len(inputs) == 2
        context_tokens_tensor, context_length_tensor = inputs
        if self.fp8_enabled:
            assert context_tokens_tensor.size(0) % 8 == 0

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            context_tokens_tensor,
            self.model.tokenizer.eos_id,
            self.model.cfg.get("reset_position_ids", False),
            self.model.cfg.get("reset_attention_mask", False),
            self.model.cfg.get("eod_mask_loss", False),
        )

        rewards = self.model.model(context_tokens_tensor, context_length_tensor, position_ids, attention_mask)

        if mpu.is_pipeline_last_stage():
            # Standardize values to subtract a bias.
            if self.enable_standardization and self.rew_mean is not None and self.rew_std is not None:
                rewards = (rewards - self.rew_mean) / self.rew_std

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            rewards = broadcast_2d_tensor(
                rewards, mpu.get_pipeline_model_parallel_last_rank(), mpu.get_pipeline_model_parallel_group(),
            )

        return {"reward": rewards}
