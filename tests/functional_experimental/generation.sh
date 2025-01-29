#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
set -eoux pipefail

export NCCL_ALGO=Tree
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_APPLY_QK_LAYER_SCALING=1

GBS=${GBS:-4}
PRETRAINED_CHECKPOINT_NEMO_FILE=${PRETRAINED_CHECKPOINT_NEMO_FILE}


#MIN_LR=$(awk -v var="$LR" 'BEGIN {print var - 1e-11}')

TRAIN_DATA_PATH=$SCRIPT_DIR/test_data/sft_512_sample_llama3_format.jsonl
VALID_DATA_PATH=$SCRIPT_DIR/test_data/sft_512_sample_llama3_format.jsonl

NAME="llama3_generation_test"

# PARAMETERS
RESULTS_DIR="/tmp/${NAME}"
mkdir -p $RESULTS_DIR

GPFS=$(git rev-parse --show-toplevel)

# W&B Logging
PROJECT=llama3_generation_test

# START HETEROGENEUS JOB 3
CONF_DIR="${GPFS}/examples/nlp/gpt/conf/"
CONF_NAME="gpt_generation"

CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
TENSOBOARD_DIR="${RESULTS_DIR}/tensorboard"

mkdir -p $RESULTS_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

generation() {
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${GPFS}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
mpirun -np 2 --allow-run-as-root python -u ${GPFS}/examples/nlp/gpt/run_generation.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONF_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    pretrained_checkpoint.restore_from_path=\"${PRETRAINED_CHECKPOINT_NEMO_FILE}\" \
    "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
    ++exp_manager.explicit_log_dir=${RESULTS_DIR} \
    ++exp_manager.create_wandb_logger=False \
    ++exp_manager.create_checkpoint_callback=False \
    ++model.mcore_gpt=true \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    ++model.megatron_amp_O2=true \
    ++model.sequence_parallel=false \
    ++model.encoder_seq_length=4096 \
    ++model.max_position_embeddings=4096 \
    ++trainer.generation.max_epochs=1 \
    ++trainer.generation.max_steps=-1 \
    ++trainer.generation.limit_train_batches=8 \
    ++model.global_batch_size=${GBS} \
    ++model.micro_batch_size=1 \
    ++model.generation.num_responses_to_gen=1 \
    ++model.generation.length_params.max_length=2048 \
    ++model.generation.sampling_params.use_greedy=true \
    ++model.generation.sampling_params.temperature=0.2 \
    ++model.generation.sampling_params.top_p=0.8 \
    ++model.data.chat=true \
    ++model.data.sample=false \
    ++model.data.num_workers=0 \
    ++model.data.train_ds.max_seq_length=1900 \
    ++model.data.train_ds.add_eos=false \
    ++model.data.train_ds.hf_dataset=true \
    ++model.generation.sampling_params.end_strings=[\"\<\|eot_id\|\>\"] \
    ++model.data.chat_prompt_tokens.system_turn_start=\"\<\|begin_of_text\|\>\" \
    ++model.data.chat_prompt_tokens.turn_start=\"\" \
    ++model.data.chat_prompt_tokens.end_of_turn=\"\<\|eot_id\|\>\" \
    ++model.data.chat_prompt_tokens.end_of_name=\"$'\x0A\x0A'\" \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.generation.length_params.max_length=2048 \
    ++trainer.generation.trt_llm.enable=false \
    ++trainer.generation.trt_llm.model_type=llama \
    ++model.dist_ckpt_load_strictness=log_all
}

log_file=$(mktemp /tmp/generation-log-XXXXXX)
generation | tee $log_file