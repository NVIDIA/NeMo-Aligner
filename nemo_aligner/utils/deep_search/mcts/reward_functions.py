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


from typing import List

import numpy as np
from pytriton.client import FuturesModelClient

__all__ = ["get_reward"]


def _str_list2numpy(str_list: List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def get_reward(
    sentences: List[str], add_EOS=False, host="localhost", port=5555, model_name="reward_model", round_to_int=True
):
    sentences = _str_list2numpy(sentences)

    futures = []

    with FuturesModelClient(f"{host}:{port}", model_name) as client:
        for sen in np.split(sentences, sentences.shape[0]):
            add_EOS_arr = np.ones_like(sen, dtype=bool) * add_EOS
            future = client.infer_batch(sentences=sen, add_EOS=add_EOS_arr)
            futures.append(future)

    all_result_dicts = [f.result() for f in futures]

    all_rewards, all_exceeded = [], []

    for output_dict in all_result_dicts:
        reward_out = output_dict["rewards"].flatten().tolist()

        all_rewards.append(reward_out)
        all_exceeded += output_dict["exceeded"].tolist()

    # return all_rewards, all_exceeded
    outputs = []
    for reward in all_rewards:
        if round_to_int:
            reward_each = [min(4.0, max(0.0, float(r))) for r in reward]
            reward_each = [round(r) for r in reward_each]
        else:
            reward_each = reward
        outputs.append(reward_each)
    return outputs


if __name__ == "__main__":
    prompt = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User
There were 672 balloons that were either green, blue, yellow, or red. They were divided into equal groups and then Anya took half of the yellow ones home. How many balloons did Anya take home?
Please show the calculation steps and lastly make sure to put the answer (and only answer) inside \boxed{}.
<extra_id_1>Assistant
Sure!

Let's use variables to represent the number of balloons of each color:

g = number of green balloons
b = number of blue balloons
y = number of yellow balloons
r = number of red balloons

The total number of balloons is:

g + b + y + r = 672

Since the balloons were divided into equal groups, we can divide the total number of balloons by the number of colors to find the number of balloons in each group:

(g + b + y + r) / 4 = number of balloons in each group
672 / 4 = 168

So there were 168 balloons in each group.

We can use this information to find the initial number of yellow balloons:

y = 168

Anya took half of the yellow balloons home, so the number of yellow balloons she took home is:

y / 2 = 168 / 2 = 84

Therefore, Anya took 84 yellow balloons home.

\\boxed{{84}}
<extra_id_2>
"""
    get_reward([prompt], False, host="localhost", port=1424, model_name="reward_model")
