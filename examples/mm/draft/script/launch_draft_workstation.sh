#!/bin/bash
PROJECT="NeMo-draft"
WANDB="8256bec8f68d1a0ee4a3208685a8db0474d3806b"
 
export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

ACTOR_BATCH_SIZE=1
ACTOR_NUM_DEVICES=1
ACTOR_MICRO_BS=1
ACTOR_GLOBAL_BATCH_SIZE=$((ACTOR_BATCH_SIZE*ACTOR_NUM_DEVICES))

ACTOR_CONFIG_PATH="/opt/nemo-aligner/examples/mm/draft/conf"
ACTOR_CONFIG_NAME="draft_sd_policy"
ACTOR_CKPT="/opt/nemo-aligner/checkpoints/model_weights.ckpt"
VAE_CKPT="/opt/nemo-aligner/checkpoints/vae.bin"
ACTOR_WANDB_NAME="draft-ws"
DIR_SAVE_CKPT_PATH="/opt/nemo-aligner/saved_ckpts"


ACTOR_DEVICE="0"
echo "Running DRaFT on ${ACTOR_DEVICE}"
git config --global --add safe.directory /opt/nemo-aligner \
&& wandb login ${WANDB} \
&&  MASTER_PORT=15003 CUDA_VISIBLE_DEVICES="${ACTOR_DEVICE}" python -u /opt/nemo-aligner/examples/mm/draft/train_sd_draft.py \
    --config-path=${ACTOR_CONFIG_PATH} \
    --config-name=${ACTOR_CONFIG_NAME} \
    model.optim.lr=0.0005 \
    model.optim.weight_decay=0.005 \
    model.optim.sched.warmup_steps=0 \
    model.infer.inference_steps=25 \
    model.infer.eta=1.0 \
    model.kl_coeff=10.0 \
    model.truncation_steps=1 \
    model.max_epochs=1 \
    model.max_steps=10 \
    model.save_interval=1000000 \
    model.unet_config.from_pretrained=${ACTOR_CKPT} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.micro_batch_size=${ACTOR_MICRO_BS} \
    model.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    trainer.val_check_interval=20 \
    trainer.gradient_clip_val=10.0 \
    trainer.devices=${ACTOR_NUM_DEVICES} \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_WANDB_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} #&> /opt/nemo-aligner/examples/mm/logs/draft_log.txt &
    