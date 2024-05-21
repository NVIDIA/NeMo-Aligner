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
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.nlp.gpt.megatron_gpt_regression_reward_model import MegatronGPTRegressionRewardModel
from nemo_aligner.utils.train_utils import set_sync_funcs


class MegatronGPTCategoricalRewardModel(MegatronGPTRegressionRewardModel):
    """
    Megatron GPT Regression Reward Model Training. 
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    def loss_func(self, output_tensor, label_tensor):
        """
        output_tensor: [B, num_attributes * num_category]
        label_tensor: [B, num_attributes]
        """
        mask_val = int(self.cfg.get("loss_mask_val", -100))
        mask = label_tensor != mask_val
        num_valid_attributes = mask.float().sum()
        assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"

        # Reshape output_tensor to [B * num_attributes, num_category]
        output_tensor = output_tensor.view(output_tensor.size(0) * self.cfg.num_attributes, self.cfg.num_category)
        # Flatten label_tensor to [B * num_attributes]
        label_tensor = label_tensor.view(-1).int()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=mask_val)
        # Calculate the loss
        loss = criterion(output_tensor, label_tensor)

        return loss
