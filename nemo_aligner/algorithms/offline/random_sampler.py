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

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.utils import logging

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RandomSampler:
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

        self.trainer = trainer
        self.model = model
        self.fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)

        self.length_params: LengthParam = {
            "max_length": inference_cfg.tokens_to_generate,
            "min_length": inference_cfg.min_tokens_to_generate,
        }

        self.sampling_params: SamplingParam = {
            "use_greedy": inference_cfg.greedy,
            "temperature": inference_cfg.temperature,
            "top_k": inference_cfg.top_k,
            "top_p": inference_cfg.top_p,
            "repetition_penalty": inference_cfg.repetition_penalty,
            "add_BOS": inference_cfg.add_BOS,
            "all_probs": inference_cfg.all_probs,
            "compute_logprob": inference_cfg.compute_logprob,
            "end_strings": inference_cfg.end_strings,
        }

    def generate(self, inputs):
        if self.fp8_enabled:
            assert inputs[0].size(0) % 8 == 0

        # First method of running text generation, call model.generate method
        response = self.model.generate(
            inputs=inputs, length_params=self.length_params, sampling_params=self.sampling_params
        )

        return {"sentences": response["sentences"], "token_ids": response["token_ids"]}
