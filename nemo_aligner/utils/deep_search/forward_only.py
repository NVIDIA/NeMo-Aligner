# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.core import InferenceParams, parallel_state


def get_forward_output_only_func(text_gen_strategy, session_info):
    inference_params = text_gen_strategy.get_inference_params(session_info)

    def fwd_output_only_func(dataloader_iter, model):
        batch = next(dataloader_iter)
        extra_arg = {}
        (tokens, attention_mask, position_ids,) = batch
        tokens = tokens.cuda()
        position_ids = position_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
            # attention_mask = attention_mask[0:1]
        extra_arg["inference_params"] = inference_params
        output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

        def id_func(output_tensor):
            return output_tensor, {"logits": output_tensor}

        return output_tensor, id_func

    return fwd_output_only_func


def get_forward_output_only_func_hybrid(text_gen_strategy, session_info):
    inference_params = text_gen_strategy.get_inference_params(session_info)

    def fwd_output_only_func(dataloader_iter, model):
        batch = next(dataloader_iter)
        extra_arg = {}
        (tokens, attention_mask, position_ids,) = batch
        tokens = tokens.cuda()
        position_ids = position_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
            # attention_mask = attention_mask[0:1]
        extra_arg["inference_params"] = inference_params
        output_tensor, value = model(tokens, position_ids, attention_mask, **extra_arg)
        # TODO. hard coded signmoid for now. only works with model trained with cross entropy loss
        value = value.sigmoid()

        def id_func(output_tensor):
            return output_tensor, {"logits": output_tensor, "value": value}

        return output_tensor, id_func

    return fwd_output_only_func
