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


from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

## tracks mapping of strings to callbacks. 
## if a user would like to use custom callbacks, they should be sure
## to add the callbacks to this dict
## dict should map strings to callbacks
## e.g. CALLBACKS_MAPPING['my_callback_name'] = CallbackClass()
global CALLBACKS_MAPPING = {}


def add_custom_meta_to_loggers(trainer, meta_dict):
    for logger in trainer.loggers:
        if isinstance(logger, MLFlowLogger):
            client = logger.experiment  # logger.experiment is a MLFlowClient object
            run_id = logger.run_id
            for k, v in meta_dict.items():
                client.log_param(run_id, k, v)

        if isinstance(logger, WandbLogger):
            run = logger.experiment
            for k, v in meta_dict.items():
                run.config[k] = v