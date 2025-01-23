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

KL=${KL:-0.03}
LR=${LR:-9e-7}
RUN_ONLY=${RUN_ONLY:-}
GBS=${GBS:-2}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
RESHARD=${RESHARD:-True}
RM_NEMO_FILE=${RM_NEMO_FILE}
ACTOR_NEMO_FILE=${ACTOR_NEMO_FILE}


MIN_LR=$(awk -v var="$LR" 'BEGIN {print var - 1e-11}')

TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/synthetic-123.jsonl
VALID_DATA_PATH=$SCRIPT_DIR/test_data/synthetic-123.jsonl

NAME="reinforce_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=reinforce_test

REWARD_LOG_DIR="${RESULTS_DIR}/reward_results"
REWARD_PORT=5555

mkdir -p $REWARD_LOG_DIR

REWARD_NAME="${NAME}_reward"

reward() {
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
python -u ${GPFS}/examples/nlp/gpt/serve_reward_model.py \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    inference.port=${REWARD_PORT} \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    ++model.dist_ckpt_load_strictness=log_all \
    rm_model_file=${RM_NEMO_FILE}
}
reward_log_file=$(mktemp /tmp/reward-reinforce-log-XXXXXX)
if [[ $RUN_ONLY =~ actor* ]]; then
    echo SKIPPING REWARD
elif [[ $RUN_ONLY == reward ]]; then
    reward 2>&1 | stdbuf -o0 sed 's/^/[REWARD_SERVER]: /' | tee $reward_log_file
    exit $?
else
    reward 2>&1 | stdbuf -o0 sed 's/^/[REWARD_SERVER]: /' | tee $reward_log_file &
fi

if [[ -z "${FAST:-}" ]]; then
    sleep 15
fi
#########################################################################################

ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
mkdir -p $ACTOR_LOG_DIR

ACTOR_NAME="${NAME}_actor"
host_reward=localhost

actor() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
mpirun -np 2 --allow-run-as-root python -u ${GPFS}/examples/nlp/gpt/train_gpt_reinforce_actor.py \
    "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    pretrained_checkpoint.restore_from_path=${ACTOR_NEMO_FILE} \
    exp_manager.explicit_log_dir=${ACTOR_LOG_DIR} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.create_checkpoint_callback=True \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    trainer.reinforce.trt_llm.enable=True \
    ++model.offload_adam_states=False \
    trainer.reinforce.trt_llm.reshard=${RESHARD} \
    trainer.reinforce.val_check_interval=2 \
    ++trainer.reinforce.save_interval=2 \
    ++model.micro_batch_size=1 \
    ++model.global_batch_size=${GBS} \
    ++model.tensor_model_parallel_size=${TP_SIZE} \
    ++model.pipeline_model_parallel_size=${PP_SIZE} \
    ++model.reinforce.entropy_bonus=0.0 \
    ++model.reinforce.ratio_eps=0.2 \
    ++model.encoder_seq_length=64 \
    ++exp_manager.checkpoint_callback_params.save_top_k=10 \
    ++model.reinforce.num_rollout_samples=${GBS} \
    ++model.reinforce.rollout_micro_batch_size=1 \
    ++model.reinforce.length_params.max_length=32      \
    ++model.reinforce.forward_micro_batch_size=1 \
    trainer.reinforce.initial_policy_kl_penalty="${KL}" \
    trainer.reinforce.rollout_batch_seq_length=32        \
    ++trainer.reinforce.flask_server.enable=True \
    ++model.optim.lr=${LR} \
    ++model.optim.sched.min_lr=${MIN_LR} \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.optim.bucket_cap_mb=200 \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++model.enable_nge=True \
    remote_rm.reward_model.ip=${host_reward} \
    remote_rm.reward_model.port=${REWARD_PORT} \
    \
    +model.overwrite_base_config.optim=True \
    '~model.optim' \
    '++model.optim={name:sgd}' \
    model.reinforce.sampling_params.use_greedy=True \
    trainer.reinforce.save_interval=0 \
    trainer.reinforce.max_steps=3 \
    trainer.reinforce.trt_llm.model_type=llama \
    ++exp_manager=null \
    \
    ++model.dist_ckpt_load_strictness=log_all \
    $@
}

actor_log_file=$(mktemp /tmp/actor-reinforce-log-XXXXXX)
if [[ -z "$RUN_ONLY" || "$RUN_ONLY" == actor_trt || "$RUN_ONLY" == trt ]]; then
  actor 2>&1 | stdbuf -o0 sed 's/^/[ACTOR_TRT]: /'
elif [[ "$RUN_ONLY" == actor_nemo || "$RUN_ONLY" == nemo ]]; then
  actor trainer.reinforce.trt_llm.enable=False 2>&1 | stdbuf -o0 sed 's/^/[ACTOR_NEMO]: /'
else
  echo "Only accepts RUN_ONLY=actor_nemo or actor_trt"
  exit 1
fi | tee $actor_log_file || true

REWARD_ID=$(grep -oP "kill -SIGINT \K\d+" $reward_log_file)
if [[ $REWARD_ID =~ ^[0-9]+$ ]]; then
    echo "Valid integer: $REWARD_ID"
    kill -SIGINT $REWARD_ID
else
    echo "Invalid REWARD_ID=$REWARD_ID detected!"
    exit 1
fi

if ! fgrep 'Cleaning up communicator' $actor_log_file &>/dev/null; then
  echo "[ERROR] Did not find 'Cleaning up communicator' in the actor logs ($actor_log_file) indicating the actor reached the end"
  exit 1
fi

echo "Waiting for backgrounded processes to finish..."
wait
set +x
echo "[Finished] $0"
