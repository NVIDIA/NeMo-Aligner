#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
set -eoux pipefail

export NCCL_ALGO=Tree
export NVTE_APPLY_QK_LAYER_SCALING=1

KL=${KL:-0.1}
GBS=${GBS:-4}
PRETRAINED_CHECKPOINT_NEMO_FILE=${PRETRAINED_CHECKPOINT_NEMO_FILE}


TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/dummy_dpo_packed_90.npy
VALID_DATA_PATH=$SCRIPT_DIR/test_data/dummy_dpo_packed_90.npy

NAME="dpo_pack_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# START HETEROGENEUS JOB 3
CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="gpt_dpo"

mkdir -p $RESULTS_DIR

dpo() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
torchrun --nproc-per-node 2 ${GPFS}/examples/nlp/gpt/train_gpt_dpo.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONF_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    ++model.data.data_impl=packed_jsonl \
    ++model.data.seq_length=128 \
    ++model.global_batch_size=${GBS} \
    ++model.micro_batch_size=1 \
    ++model.mcore_gpt=true \
    ++model.megatron_amp_O2=true \
    ++model.dpo.ref_policy_kl_penalty=${KL} \
    ++model.dpo.log_prob_forward_micro_batch_size=1 \
    ++model.dpo.average_log_probs=false \
    ++model.dpo.sft_loss_weight=0.1 \
    ++model.dpo.preference_loss_weight=1.0 \
    pretrained_checkpoint.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    exp_manager.create_checkpoint_callback=False \
    model.data.num_workers=2 \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    trainer.dpo.max_steps=10 \
    trainer.dpo.val_check_interval=10 \
    trainer.dpo.limit_val_batches=8 \
    trainer.dpo.save_interval=0 \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.dist_ckpt_load_strictness=log_all \
    ++model.transformer_engine=True
}

log_file=$(mktemp /tmp/dpo-pack-log-XXXXXX)
dpo | tee $log_file