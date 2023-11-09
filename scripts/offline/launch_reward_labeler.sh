#!/bin/bash

RLHF_DIR=<<<PATH/TO/nemo-rlhf>>>
export PYTHONPATH=${RLHF_DIR}:$PYTHONPATH

GPU_NUMBER=<<<GPU_NUMBER>>>
MODEL_PATH=<<<REWARD_MODEL_PATH>>>
MICRO_BATCH_SIZE=<<<MICRO_BATCH_SIZE>>>
MAX_SEQUENCE_LENGTH=<<<MAX_SEQUENCE_LENGTH>>>

INPUT_PATH=<<<INPUT_JSONL_FILE_PATH>>>
OUTPUT_PATH=<<<OUTPUT_JSONL_FILE_PATH>>>

REWARD_NORM_ENABLE=True # set to True or False
REWARD_NORM_MEAN=0.0 # set to float or null
REWARD_NORM_STD=1.0 # set to float or null

# post processor, set to best_of_n (Rejection Sampling), dt (Decision Transformer), filter (ReST) or null
POST_PROCESSOR=null 

python ${RLHF_DIR}/examples/nlp/gpt/offline/launch_reward_labeler.py \
        gpt_model_file=${MODEL_PATH} \
        trainer.devices=$GPU_NUMBER \
        trainer.num_nodes=1  \
        trainer.precision=bf16-mixed \
        tensor_model_parallel_size=-1 \
	pipeline_model_parallel_size=-1 \
        output_file=$OUTPUT_PATH \
        data.hf_dataset=True \
	data.micro_batch_size=$MICRO_BATCH_SIZE \
        data.max_seq_length=$MAX_SEQUENCE_LENGTH \
        processor=$POST_PROCESSOR \
        reward_standardization.enable=$REWARD_NORM_ENABLE \
        reward_standardization.mean=$REWARD_NORM_MEAN \
        reward_standardization.std=$REWARD_NORM_STD \
        data.concat_sampling_probabilities=[1] \
        data.file_names=[$INPUT_PATH]