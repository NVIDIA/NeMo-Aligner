#!/bin/bash
PROJECT="NeMo-draft++"
WANDB="8256bec8f68d1a0ee4a3208685a8db0474d3806b"
 
export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

ACTOR_NUM_DEVICES=2
ACTOR_MICRO_BS=1
GRAD_ACCUMULATION=4
ACTOR_GLOBAL_BATCH_SIZE=$((ACTOR_MICRO_BS*ACTOR_NUM_DEVICES*GRAD_ACCUMULATION))
KL_COEF=0.1
LR=0.0005

ACTOR_CONFIG_PATH="/opt/nemo-aligner/examples/mm/draft/conf"
ACTOR_CONFIG_NAME="draft_sd_policy"
ACTOR_CKPT="/opt/nemo-aligner/checkpoints/model_weights.ckpt"
VAE_CKPT="/opt/nemo-aligner/checkpoints/vae.bin"
ACTOR_WANDB_NAME=draft-ws-LR_${LR}-KL_${KL_COEF}-BS_${ACTOR_GLOBAL_BATCH_SIZE}
DIR_SAVE_CKPT_PATH="/opt/nemo-aligner/draft_saved_ckpts"

mkdir -p ${DIR_SAVE_CKPT_PATH}

ACTOR_DEVICE="0,1"
echo "Running DRaFT on ${ACTOR_DEVICE}"
git config --global --add safe.directory /opt/nemo-aligner \
&& wandb login ${WANDB} \
&&  MASTER_PORT=15003 CUDA_VISIBLE_DEVICES="${ACTOR_DEVICE}" torchrun /opt/nemo-aligner/examples/mm/draft/train_sd_draft.py \
    --config-path=${ACTOR_CONFIG_PATH} \
    --config-name=${ACTOR_CONFIG_NAME} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=0.005 \
    model.optim.sched.warmup_steps=0 \
    model.infer.inference_steps=25 \
    model.infer.eta=0.0 \
    model.kl_coeff=${KL_COEF} \
    model.truncation_steps=1 \
    model.max_epochs=40 \
    model.max_steps=4000 \
    model.save_interval=500 \
    model.unet_config.from_pretrained=${ACTOR_CKPT} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.micro_batch_size=${ACTOR_MICRO_BS} \
    model.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    trainer.val_check_interval=20 \
    trainer.gradient_clip_val=10.0 \
    trainer.devices=${ACTOR_NUM_DEVICES} \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_WANDB_NAME} \
    exp_manager.resume_if_exists=False \
    exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} #&> /opt/nemo-aligner/examples/mm/logs/draft_log.txt &
