#!/bin/sh
TRAIN_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/gsm8k/train_full.jsonl
VALID_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/gsm8k/test.jsonl
TEST_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/gsm8k/test.jsonl
CHECKPOINT=/opt/NeMo/checkpoints/qwen2-1-5b-it.nemo
#CHECKPOINT=/opt/NeMo/checkpoints/llama3-1-8B-instruct.nemo
#CHECKPOINT=/opt/NeMo/checkpoints/llama3-2-1B-instruct.nemo
#CHECKPOINT=/opt/NeMo-Aligner/checkpoints/2b_mcore_actor.nemo

python train_gpt_star.py \
    pretrained_checkpoint.restore_from_path=$CHECKPOINT \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    model.star.rollout_micro_batch_size=4 \
    model.star.num_rollout_samples=32 \
    model.star.num_rollouts_per_prompt=4 \
    model.star.top_n_rollouts=4 \
    model.micro_batch_size=2 \
    model.global_batch_size=32 \
    +model.tensor_model_parallel_size=1
