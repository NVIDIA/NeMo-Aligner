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

NAME="ppo_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=ppo_test

CRITIC_CONFIG_PATH="$GPFS/examples/nlp/gpt/conf/"
CRITIC_CONFIG_NAME="gpt_ppo_critic"

CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR

CRITIC_NAME="${NAME}_critic"

critic() {
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
python -u ${GPFS}/examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=${CRITIC_CONFIG_PATH} \
    --config-name=${CRITIC_CONFIG_NAME} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    exp_manager.explicit_log_dir=${CRITIC_LOG_DIR} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${CRITIC_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    trainer.ppo.inference_micro_batch_size=1 \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    trainer.ppo.port=${CRITIC_PORT} \
    ++model.reward_model_type=regression \
    ++model.regression.num_attributes=1 \
    ++model.forward_mbs=1 \
    ++model.micro_batch_size=1 \
    ++model.global_batch_size=1 \
    ++model.tensor_model_parallel_size=1 \
    ++model.optim.bucket_cap_mb=200 \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++trainer.ppo.pad_sequence_length_to_multiple=32 \
    model.reward_standardization.enable=True \
    model.reward_standardization.mean=5.3735 \
    model.reward_standardization.std=1.2723 \
    pretrained_checkpoint.restore_from_path=${RM_NEMO_FILE} \
    ++model.mcore_gpt=True \
    exp_manager.create_checkpoint_callback=False \
    \
    exp_manager.create_wandb_logger=False \
    model.encoder_seq_length=$((1024+512)) # generation + input len

}
critic_log_file=$(mktemp /tmp/critic-ppo-log-XXXXXX)
if [[ $RUN_ONLY =~ actor* ]]; then
    echo SKIPPING CRITIC
elif [[ $RUN_ONLY == critic ]]; then
    critic 2>&1 | stdbuf -o0 sed 's/^/[CRITIC_SERVER]: /' | tee $critic_log_file
    exit $?
else
    critic 2>&1 | stdbuf -o0 sed 's/^/[CRITIC_SERVER]: /' | tee $critic_log_file &
fi

if [[ -z "${FAST:-}" ]]; then
    sleep 15
fi
#########################################################################################

CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="gpt_ppo_actor"

ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
mkdir -p $ACTOR_LOG_DIR

ACTOR_NAME="${NAME}_actor"
host_critic=localhost

actor() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
mpirun -np 2 --allow-run-as-root python -u ${GPFS}/examples/nlp/gpt/train_gpt_ppo_actor.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONF_NAME} \
    "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    pretrained_checkpoint.restore_from_path=${ACTOR_NEMO_FILE} \
    exp_manager.explicit_log_dir=${ACTOR_LOG_DIR} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.create_checkpoint_callback=True \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    trainer.ppo.trt_llm.enable=True \
    ++model.offload_adam_states=False \
    trainer.ppo.trt_llm.reshard=${RESHARD} \
    trainer.ppo.val_check_interval=2 \
    ++trainer.ppo.save_interval=2 \
    ++model.micro_batch_size=1 \
    ++model.global_batch_size=${GBS} \
    ++model.tensor_model_parallel_size=${TP_SIZE} \
    ++model.pipeline_model_parallel_size=${PP_SIZE} \
    ++model.ppo.entropy_bonus=0.0 \
    ++model.ppo.ratio_eps=0.2 \
    ++model.encoder_seq_length=64 \
    ++exp_manager.checkpoint_callback_params.save_top_k=10 \
    ++model.ppo.num_rollout_samples=${GBS} \
    ++model.ppo.rollout_micro_batch_size=1 \
    ++model.ppo.length_params.max_length=32      \
    ++model.ppo.forward_micro_batch_size=1 \
    trainer.ppo.initial_policy_kl_penalty="${KL}" \
    trainer.ppo.rollout_batch_seq_length=32        \
    ++trainer.ppo.flask_server.enable=True \
    ++model.optim.lr=${LR} \
    ++model.optim.sched.min_lr=${MIN_LR} \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.optim.bucket_cap_mb=200 \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++model.enable_nge=True \
    remote_critic_rm.critic.ip=${host_critic} \
    remote_critic_rm.critic.port=${CRITIC_PORT} \
    \
    +model.overwrite_base_config.optim=True \
    '~model.optim' \
    '++model.optim={name:sgd}' \
    model.ppo.sampling_params.use_greedy=True \
    trainer.ppo.save_interval=0 \
    trainer.ppo.max_steps=3 \
    trainer.ppo.trt_llm.model_type=llama \
    ++exp_manager=null \
    remote_critic_rm.pad_to_length=$((512+256)) $@ # (match critic) generation + prompt = model.ppo.length_params.max_length + model.ppo.trt_llm.max_input_len (512) = self.trtllm_generate.max_generation_length + self.trtllm_generate.max_input_len
}

actor_log_file=$(mktemp /tmp/actor-ppo-log-XXXXXX)
if [[ -z "$RUN_ONLY" || "$RUN_ONLY" == actor_trt || "$RUN_ONLY" == trt ]]; then
  actor 2>&1 | stdbuf -o0 sed 's/^/[ACTOR_TRT]: /'
elif [[ "$RUN_ONLY" == actor_nemo || "$RUN_ONLY" == nemo ]]; then
  actor trainer.ppo.trt_llm.enable=False 2>&1 | stdbuf -o0 sed 's/^/[ACTOR_NEMO]: /'
else
  echo "Only accepts RUN_ONLY=actor_nemo or actor_trt"
  exit 1
fi | tee $actor_log_file || true

CRITIC_ID=$(grep -oP "kill -SIGINT \K\d+" $critic_log_file)
if [[ $CRITIC_ID =~ ^[0-9]+$ ]]; then
    echo "Valid integer: $CRITIC_ID"
    kill -SIGINT $CRITIC_ID
else
    echo "Invalid CRITIC_ID=$CRITIC_ID detected!"
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
