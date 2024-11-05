#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
set -eoux pipefail

export NCCL_ALGO=Tree
export NVTE_APPLY_QK_LAYER_SCALING=1

PRETRAINED_CHECKPOINT_NEMO_FILE=${PRETRAINED_CHECKPOINT_NEMO_FILE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}

TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/dummy-sft.jsonl
VALID_DATA_PATH=$SCRIPT_DIR/test_data/dummy-sft.jsonl

NAME="sft_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=sft_test

CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="gpt_sft"

CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
TENSOBOARD_DIR="${RESULTS_DIR}/tensorboard"

mkdir -p $RESULTS_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

sft() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
torchrun --nproc-per-node 2 --allow-run-as-root python -u ${GPFS}/examples/nlp/gpt/train_gpt_sft.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONF_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    ++model.mcore_gpt=True \
    ++model.megatron_amp_O2=True \
    model.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
    exp_manager.create_checkpoint_callback=False \
    model.data.num_workers=0 \
    model.data.chat=True \
    model.data.chat_prompt_tokens.system_turn_start=\'\<extra_id_0\>\' \
    model.data.chat_prompt_tokens.turn_start=\'\<extra_id_1\>\' \
    model.data.chat_prompt_tokens.label_start=\'\<extra_id_2\>\' \
    model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
    model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.train_ds.file_path=${TRAIN_DATA_PATH} \
    model.data.train_ds.index_mapping_dir=${SCRIPT_DIR}/test_data \
    model.data.train_ds.add_eos=False \
    model.data.train_ds.hf_dataset=True \
    model.data.validation_ds.max_seq_length=${MAX_SEQ_LENGTH} \
    model.data.validation_ds.file_path=${VALID_DATA_PATH} \
    model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.data.validation_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.validation_ds.index_mapping_dir=${SCRIPT_DIR}/test_data \
    model.data.validation_ds.add_eos=False \
    model.data.validation_ds.hf_dataset=True \
    model.answer_only_loss=True \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    trainer.sft.max_steps=5 \
    trainer.sft.val_check_interval=1 \
    trainer.sft.limit_val_batches=8 \
    trainer.sft.save_interval=0 \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.dist_ckpt_load_strictness=log_all
}

log_file=$(mktemp /tmp/sft-log-XXXXXX)
sft | tee $log_file