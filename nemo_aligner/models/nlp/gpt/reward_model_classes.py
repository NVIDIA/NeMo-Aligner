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

import enum

from nemo_aligner.models.nlp.gpt.megatron_gpt_regression_reward_model import MegatronGPTRegressionRewardModel
from nemo_aligner.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel

__all__ = ["RewardModelType", "REWARD_MODEL_CLASS_DICT"]


class RewardModelType(enum.Enum):
    BINARY_RANKING = "binary_ranking"
    REGRESSION = "regression"


REWARD_MODEL_CLASS_DICT = {
    RewardModelType.BINARY_RANKING: MegatronGPTRewardModel,
    RewardModelType.REGRESSION: MegatronGPTRegressionRewardModel,
}
