#!/bin/bash

RLHF_DIR=<<<PATH/TO/nemo-rlhf>>>
export PYTHONPATH=${RLHF_DIR}:$PYTHONPATH

GPU_NUMBER=<<<GPU_NUMBER>>>
MODEL_PATH=<<<POLICY_MODEL_PATH>>>
MICRO_BATCH_SIZE=<<<MICRO_BATCH_SIZE>>>
MAX_SEQUENCE_LENGTH=<<<MAX_SEQUENCE_LENGTH>>>

BEST_OF_N=<<<GENERATE_N_SAMPLES_EACH_PROMPT>>>

INPUT_PATH=<<<INPUT_JSON_FILE_PATH>>>
OUTPUT_PATH=<<<OUTPUT_JSON_FILE_PATH>>>

python ${RLHF_DIR}/examples/nlp/gpt/offline/launch_random_sampler.py \
        gpt_model_file=${MODEL_PATH} \
        inference.greedy=False \
        inference.add_BOS=False \
        inference.tokens_to_generate=$((MAX_SEQUENCE_LENGTH / 2)) \
        trainer.devices=<<<GPU_NUMBER>>> \
        trainer.num_nodes=1  \
        trainer.precision=bf16-mixed \
        megatron_amp_O2=True \
        tensor_model_parallel_size=-1 \
	pipeline_model_parallel_size=-1 \
        output_file=$OUTPUT_PATH \
        data.hf_dataset=True \
	data.micro_batch_size=$MICRO_BATCH_SIZE \
        data.max_seq_length=$((MAX_SEQUENCE_LENGTH / 2)) \
        data.best_of_n=$BEST_OF_N \
        data.concat_sampling_probabilities=[1] \
        data.file_names=[$INPUT_PATH]