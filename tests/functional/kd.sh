#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
set -eoux pipefail

export NCCL_ALGO=Tree
export NVTE_APPLY_QK_LAYER_SCALING=1

PRETRAINED_CHECKPOINT_NEMO_FILE=${PRETRAINED_CHECKPOINT_NEMO_FILE}

#MIN_LR=$(awk -v var="$LR" 'BEGIN {print var - 1e-11}')

TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/synthetic-123-kd-CHUNK_ID.jsonl
VALID_DATA_PATH=$SCRIPT_DIR/test_data/synthetic-123-kd-CHUNK_ID.jsonl

NAME="llama3_kd_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
LOGITS_DATA_DIR="/tmp/${NAME}_data"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=kd_test

# START HETEROGENEUS JOB 3
CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="gpt_kd"

CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
TENSOBOARD_DIR="${RESULTS_DIR}/tensorboard"

mkdir -p $RESULTS_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

## note: for convenience, the teacher and the student are the same model in this example
## this should never be done in practice

kd() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
torchrun --nproc-per-node 2 python -u ${GPFS}/examples/nlp/synthetic_data_gen/compute_topk_logits.py \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    trainer.precision=bf16 \
    pretrained_checkpoint.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
    model.megatron_amp_O2=True \
    data.sample=True \
    data.num_workers=0 \
    data.data.max_seq_length=128 \
    data.data.file_path=$SCRIPT_DIR/test_data/synthetic-kd.jsonl \
    data.data.add_eos=False \
    data.data.hf_dataset=True \
    top_k=4 \
    model.global_batch_size=16 \
    model.micro_batch_size=2 \
    start_from_idx=0 \
    end_at_idx=49 \
    output_path=${LOGITS_DATA_DIR}/train_with_logits_0.jsonl

TRAIN_DATA_PATH=${LOGITS_DATA_DIR}/train_with_logits_CHUNK_ID.jsonl
VALID_DATA_PATH=${LOGITS_DATA_DIR}/train_with_logits_CHUNK_ID.jsonl

mpirun -np 2 --allow-run-as-root python -u ${GPFS}/examples/nlp/gpt/train_gpt_knowledge_distillation.py \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    trainer.knowledge_distillation.max_steps=3 \
    pretrained_checkpoint.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
    ++model.data.data_impl=chunked_jsonl \
    ++model.data.n_chunks=1 \
    ++"model.data.n_examples_per_chunk={train: 50, validation: 50, test: 50}" \
    ++model.data.seq_length=128 \
    ++model.global_batch_size=4 \
    ++model.micro_batch_size=1 \
    ++model.mcore_gpt=true \
    ++model.megatron_amp_O2=true \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    exp_manager.create_checkpoint_callback=False \
    model.data.num_workers=2 \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.dist_ckpt_load_strictness=log_all \
    model.knowledge_distillation.target_logits_scale=1.0 \
    model.knowledge_distillation.logits_scale=1.0 \
    model.knowledge_distillation.sft_loss_weight=0.1 \
    model.knowledge_distillation.kd_loss_weight=1 \
    model.knowledge_distillation.kd_loss=bwd_kl \

}

log_file=$(mktemp /tmp/kd-log-XXXXXX)
kd | tee $log_file