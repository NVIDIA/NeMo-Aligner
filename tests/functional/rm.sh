#!/bin/bash

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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
set -eoux pipefail

export NCCL_ALGO=Tree
export NVTE_APPLY_QK_LAYER_SCALING=1

KL=${KL:-0.1}
LR=${LR:-9e-7}
GBS=${GBS:-4}
PRETRAINED_CHECKPOINT_NEMO_FILE=${PRETRAINED_CHECKPOINT_NEMO_FILE}


TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/test-rm.jsonl
VALID_DATA_PATH=$SCRIPT_DIR/test_data/test-rm.jsonl

NAME="rm_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=rm_test

# START HETEROGENEUS JOB 3
CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="training_rm"


mkdir -p $RESULTS_DIR

rm_training() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
torchrun --nproc_per_node=2 ${GPFS}/examples/nlp/gpt/train_reward_model.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONF_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    model.global_batch_size=4 \
    ++model.data.data_impl=jsonl \
    ++model.data.train_ds.shuffle=False \
    ++model.data.validation_ds.shuffle=False \
    ++model.data.shuffle_documents=False \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    trainer.rm.val_check_interval=1 \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.explicit_log_dir=/results \
    exp_manager.checkpoint_callback_params.save_top_k=2 \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=False \
    pretrained_checkpoint.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
    trainer.rm.save_interval=1 \
    trainer.rm.max_steps=1 \
    model.optim.lr=${LR} \
    model.optim.sched.min_lr=${LR} \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.warmup_steps=10 \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    ++model.activations_checkpoint_granularity="full" \
    ++model.activations_checkpoint_method="uniform" \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.sequence_parallel=False \
    model.reward_model_type="regression" \
    ++model.dist_ckpt_load_strictness=log_all \
    model.regression.num_attributes=9
}

log_file=$(mktemp /tmp/rm-log-XXXXXX)
rm_training | tee $log_file
echo "[Finished] $0"