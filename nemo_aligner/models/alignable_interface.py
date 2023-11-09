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

from abc import ABC, abstractmethod

"""This file defines interfaces that the model has to implement and the
    trainer needs to call. There is no strict requirement to implement everything
    and we keep it as minimial as possible
"""


class SupervisedInterface(ABC):
    """for models that can be fine tuned using the supervised training loop
        such as reward models, or SFT models
    """

    @abstractmethod
    def get_loss_and_metrics(self, batch, forward_only):
        """Take a micro_batch_size * num microbatches input and return loss as well as metrics
        if forward_only is False, then it's expected the user calls
            loss.backward and populate the gradients

        NOTE: the metrics must be on the CPU and be replicated across all ranks
        """

    # TODO(geshen): consider adding prepare_for_{train,validation} to go along with
    # the prepare_for_{train,validation}_step
    @abstractmethod
    def prepare_for_training_step(self):
        """things to call to prepare for training
        """

    @abstractmethod
    def finish_training_step(self):
        """things to call to finish training for example grad reductions
        """

    def prepare_for_validation_step(self):
        """things to call to prepare for validation
        """
        raise NotImplementedError

    def finish_validation_step(self):
        """things to call to prepare for validation
        """
        raise NotImplementedError


class Inferrable(ABC):
    """For models that we want to infer on. On a language model
        this should run generate, on a reward model/critic this should
        give the numerical values
    """

    @abstractmethod
    def prepare_for_inference(self):
        """to prepare things for inference
        """

    @abstractmethod
    def finish_inference(self):
        """to restore things after doing inference
        """

    @abstractmethod
    def infer(self, *args, **kwargs):
        """to run inference on the RM to get the rewards out
        """


class CriticModelInterface(SupervisedInterface, Inferrable):
    def prepare_for_training(self):
        """prepare for training, only called once before we start training
        """

    def finish_training(self):
        """prepare for training, only called once before we start training. NOTE: this is different than
            finish_training_step because this is called before and after training stage
        """

    def infer_rm_critic(self, *args, **kwargs):
        """this is called when the critic and reward model are co-located. To return for both RM and critic.
            have to return the rewards and critic in that order
        """


class AlignableGenerativeInterface(SupervisedInterface, Inferrable):
    @abstractmethod
    def prepare_for_training(self):
        """Takes care of any optimizer onloading or configuration"""

    @abstractmethod
    def finish_training(self):
        """Takes care of any optimizer offloading or configuration"""

    def get_init_policy_logprobs(self, rollout_batches: list):
        """get the logprobs for initial policy, trainers should only call this
            if the trainer knows this is needed
        """
