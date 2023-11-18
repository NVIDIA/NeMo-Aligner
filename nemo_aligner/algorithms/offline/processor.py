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
from tqdm import tqdm


def reward_normalization(cfg, objs):
    rewards = [float(obj[cfg.reward_key]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj[cfg.reward_key] = rewards[i].item()
    return objs


# Rejection Sampling
# See https://arxiv.org/abs/2307.09288
def rejection_sampling_processor(cfg, objs):
    out = {}
    for obj in tqdm(objs, desc=f"Best of N process...."):
        input = obj[cfg.input_key]
        output = obj[cfg.output_key]
        reward = float(obj[cfg.reward_key])

        if input not in out:
            out[input] = {cfg.output_key: output, cfg.reward_key: reward}
        elif reward > out[input][cfg.reward_key]:
            out[input][cfg.reward_key] = reward
            out[input][cfg.output_key] = output

    return [
        {cfg.input_key: k, cfg.output_key: v[cfg.output_key], cfg.reward_key: v[cfg.reward_key]}
        for k, v in out.items()
    ]


# Decision Transformer Alignment
# See https://arxiv.org/abs/2308.12050
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def decesion_transformer_processor(cfg, objs):
    reward_prompt = cfg.get("reward_template", DEFAULT_REWARD_PROMPT)
    assert f"{{{cfg.input_key}}}" in reward_prompt
    assert f"{{{cfg.reward_key}}}" in reward_prompt

    for obj in tqdm(objs, desc="Decision Transformer process..."):
        input = obj[cfg.input_key]
        reward = "{:.1f}".format(float(obj[cfg.reward_key]))
        input = reward_prompt.replace(f"{{{cfg.reward_key}}}", reward).replace(f"{{{cfg.input_key}}}", input)
        obj[cfg.input_key] = input

    return objs


# Reinforced Self-Training
# See https://arxiv.org/abs/2308.08998
def rest_processor(cfg, objs):
    threshold = cfg.get("threshold", 0)
    print(f"ReST (threshold: {threshold}) process...")

    out = [obj for obj in objs if float(obj[cfg.reward_key]) > threshold]
    return out


PROCESSORS = {
    "rs": rejection_sampling_processor,
    "dt": decesion_transformer_processor,
    "rest": rest_processor,
}


def get_processor(name):
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")
