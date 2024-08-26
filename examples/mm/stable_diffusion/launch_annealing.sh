#!/bin/bash
export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

LR=${LR:=0.00025}
INF_STEPS=${INF_STEPS:=25}
KL_COEF=${KL_COEF:=0.1}
ETA=${ETA:=0.0}
DATASET=${DATASET:="pickapic50k.tar"}
MICRO_BS=${MICRO_BS:=2}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
PEFT=${PEFT:="sdlora"}
NUM_DEVICES=8
GLOBAL_BATCH_SIZE=$((MICRO_BS*NUM_DEVICES*GRAD_ACCUMULATION))

WANDB_NAME=SD_DRaFT+${JOBNAME}_lr_${LR}_data_${DATASET}_kl_${KL_COEF}_bs_${GLOBAL_BATCH_SIZE}_infstep_${INF_STEPS}_eta_${ETA}_peft_${PEFT}
WEBDATASET_PATH=/path/to/${DATASET}

CONFIG_PATH="/opt/nemo-aligner/examples/mm/stable_diffusion/conf"
CONFIG_NAME="draftp_sd"
UNET_CKPT="/path/to/unet.ckpt"
VAE_CKPT="/path/to/vae.ckpt"
RM_CKPT="/path/to/rewardmodel.nemo"
DIR_SAVE_CKPT_PATH=/opt/nemo-aligner/sd_draft_runs/draftp_saved_ckpts_${JOBNAME}

# change this as an end-user
PROMPT=${PROMPT:-"Bananas growing on an apple tree"}

mkdir -p ${DIR_SAVE_CKPT_PATH}

EVAL_SCRIPT=${EVAL_SCRIPT:-"anneal_sd.py"}
set -x
DEVICE="0,1,2,3,4,5,6,7"
echo "Running DRaFT on ${DEVICE}"
export HYDRA_FULL_ERROR=1 \
&& MASTER_PORT=15003 CUDA_VISIBLE_DEVICES="${DEVICE}" torchrun --nproc_per_node=8 /opt/nemo-aligner/examples/mm/stable_diffusion/${EVAL_SCRIPT} \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=0.005 \
    model.optim.sched.warmup_steps=0 \
    model.infer.inference_steps=${INF_STEPS} \
    model.infer.eta=0.0 \
    model.kl_coeff=${KL_COEF} \
    model.truncation_steps=1 \
    trainer.draftp_sd.max_epochs=1 \
    trainer.draftp_sd.max_steps=4000 \
    trainer.draftp_sd.save_interval=100 \
    model.unet_config.from_pretrained=${UNET_CKPT} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.micro_batch_size=${MICRO_BS} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.peft.peft_scheme=${PEFT} \
    model.data.webdataset.local_root_path=$WEBDATASET_PATH \
    rm.model.restore_from_path=${RM_CKPT} \
    +prompt="${PROMPT}" \
    trainer.draftp_sd.val_check_interval=20 \
    trainer.draftp_sd.gradient_clip_val=10.0 \
    trainer.devices=${NUM_DEVICES} \
    rm.trainer.devices=${NUM_DEVICES} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${WANDB_NAME} \
    exp_manager.resume_if_exists=True \
    exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} $ADDITIONAL_KWARGS
