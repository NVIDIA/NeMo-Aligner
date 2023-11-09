#!/bin/bash

#SBATCH -p <<<PARTITION>>>         
#SBATCH -A <<<ACCOUNT>>>
#SBATCH -J <<<JOB_NAME>>>
#SBATCH -N 4                       # 64x8x4
#SBATCH -t 0-4:00:00               # wall time
#SBATCH --ntasks-per-node=8        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=out.log
#SBATCH --dependency singleton

# ================================= SETTINGS ======================================

# Please modify the variable in the <<<>>>  according to your project

# Project settings
RLHF_DIR=<<<PATH/TO/nemo-rlhf>>>
PROJECT_NAME=<<<WANDB_PROJECT_NAME>>>
WANDB=<<<WANDB_API_KEY>>>

# Training settings
LEARNING_RATE=<<<LEARNING_RATE>>> # Constant learning rate
WEIGHT_DECAY=<<<WEIGHT_DECAY>>> # L2 weight decay, such as 0.01
PROCESSOR=dt # Can be set to best_of_n (Rejection Sampling), filter (ReST) or dt (Decision Transformer）
MAX_NUM_SAMPLES=<<<MAX_NUM_SAMPLES>>>

RM_MICRO_BATCH_SIZE=<<<RM_MICRO_BATCH_SIZE>>> # Set it to a larger value integer 16 to improve efficiency
GLOBAL_TRAIN_BATCH_SIZE=<<<GLOBAL_TRAIN_BATCH_SIZE>>>
MICRO_TRAIN_BATCH_SIZE=<<<MICRO_TRAIN_BATCH_SIZE>>>

SEQUENCE_MAX_LENGTH=<<<SEQUENCE_MAX_LENGTH>>> # Maximum sequence length of the model

REWARD_NORM_ENABLE=True # set to True or False
REWARD_NORM_MEAN=null # set to float or null (null means calculating the mean of all input samples)
REWARD_NORM_STD=null # set to float or null (null means calculating the std of all input samples)

# Models path
POLICY_MODEL_PATH=<<<INIT_POLICY_PATH>>>
REWARD_MODEL_PATH=<<<REWARD_MODEL_PATH>>>

# -1 means read the TP/PP size from model config
TENSOR_MODEL_PARALLEL_SIZE=-1
PIPELINE_MODEL_PARALLEL_SIZE=-1

# JSON datasets path
RM_DATA_PATH=<<<RM_DATA_PATH>>>
SFT_DATA_PATH=<<<SFT_DATA_PATH>>>
LABELED_SAMPLES_PATH=<<<LABELED_DATA_PATH>>>

# Logs path
OUTPUT_PATH=<<<LOG_OUTPUT_DIR>>>
mkdir -p $OUTPUT_PATH
MODEL_OUTPUT_PATH=$OUTPUT_PATH/checkpoints/megatron_gpt_sft.nemo
LOGS_PATH=$OUTPUT_PATH/$SLURM_JOB_ID.log

# Max run time 
MAX_TIME_PER_RUN=null # days:hours:mins:secs
REMAIN_TIME=null
START_TIME=$(date +%s)

# Docker settings
CONTAINER_IMAGE=<<<CONTAINER_IMAGE>>>
MOUNT="$RLHF_DIR:$RLHF_DIR,<<<CONTAINER_MOUNT>>>"

# ================================= UTILS ===================================

# Please don't modify the utils

# Check status
checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "Check $1 FAILED"
        exit 1
    fi
}

# Get remain maxtime
getRemainTime() {
    declare -g REMAIN_TIME
    if [[ "${MAX_TIME_PER_RUN}" == "null" ]] || [[ -z "${MAX_TIME_PER_RUN}" ]]; then
        REMAIN_TIME="null"
        return
    fi

    IFS=':' read -ra time_arr <<<"$MAX_TIME_PER_RUN"
    days=${time_arr[0]}
    hours=${time_arr[1]}
    mins=${time_arr[2]}
    secs=${time_arr[3]}
    input_seconds=$((days * 24 * 3600 + hours * 3600 + mins * 60 + secs))

    now=$(date +%s)
    elapsed=$((now - START_TIME))
    remaining=$((input_seconds - elapsed))

    if ((remaining <= 0)); then
        echo "Max time reached."
        exit 0
    fi

    REMAIN_TIME=$(printf '%d:%d:%d:%d' $((remaining / 86400)) $((remaining % 86400 / 3600)) $((remaining % 3600 / 60)) $((remaining % 60)))
    echo "Remain Time: $REMAIN_TIME."
}

# ============================== TRAINING SCRIPTS =============================

# For DT Alignment, see https://arxiv.org/abs/2308.12050.
# We found that the DT Alignment without generation achieved similar performance to the DT with generation version.
# And DT without generation is significantly faster than DT with generation (In half the time).

# Labeling and filtering the samples
while [ ! -e $LABELED_SAMPLES_PATH ]; do
    getRemainTime &>>$LOGS_PATH
    read -r -d '' filter_commands <<EOF
cd ${RLHF_DIR} \
&& export PYTHONPATH="${RLHF_DIR}:\$PYTHONPATH" \
&& python examples/nlp/gpt/offline/launch_reward_labeler.py \
        gpt_model_file=${REWARD_MODEL_PATH} \
        trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
        trainer.devices=${SLURM_NTASKS_PER_NODE} \
        trainer.precision=bf16-mixed \
        megatron_amp_O2=True \
        tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
        pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
        data.micro_batch_size=$RM_MICRO_BATCH_SIZE \
        data.max_seq_length=$SEQUENCE_MAX_LENGTH \
        data.concat_sampling_probabilities=[0.5,0.5] \
        data.file_names=[$SFT_DATA_PATH,$RM_DATA_PATH] \
        data.num_samples=$MAX_NUM_SAMPLES \
        data.hf_dataset=True \
        reward_standardization.enable=$REWARD_NORM_ENABLE \
        reward_standardization.mean=$REWARD_NORM_MEAN \
        reward_standardization.std=$REWARD_NORM_STD \
        processor=$PROCESSOR \
        output_file=$LABELED_SAMPLES_PATH \
        checkpoint_interval=8 \
        max_time_per_run=$REMAIN_TIME
EOF

    echo $filter_commands &>>$LOGS_PATH
    srun --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNT" bash -c "$filter_commands" &>>$LOGS_PATH
    checkSuccess "RewardLabeler" &>>$LOGS_PATH
done

# SFT training
getRemainTime &>>$LOGS_PATH
read -r -d '' training_commands <<EOF
cd ${RLHF_DIR} \
&& export PYTHONPATH="${RLHF_DIR}:\$PYTHONPATH" \
&& export WANDB_API_KEY=${WANDB} \
&& export CUDA_DEVICE_MAX_CONNECTIONS=1 \
&& python examples/nlp/gpt/train_gpt_sft.py \
        --config-path=${RLHF_DIR}/examples/nlp/gpt/conf \
        --config-name=sft_gpt \
        trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
        trainer.devices=${SLURM_NTASKS_PER_NODE} \
        trainer.precision=bf16-mixed \
        ++trainer.sft.max_steps=-1 \
        ++trainer.sft.skip_validation=True \
        ++trainer.sft.save_interval=100 \
        model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
        model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
        model.activations_checkpoint_granularity=selective \
        model.activations_checkpoint_method=uniform \
        model.megatron_amp_O2=True \
        model.restore_from_path=$POLICY_MODEL_PATH \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=$LEARNING_RATE \
        model.optim.weight_decay=$WEIGHT_DECAY \
        model.optim.betas=[0.9,0.95] \
        model.answer_only_loss=True \
        model.data.train_ds.micro_batch_size=$MICRO_TRAIN_BATCH_SIZE \
        model.data.train_ds.global_batch_size=$GLOBAL_TRAIN_BATCH_SIZE \
        model.data.train_ds.max_seq_length=$SEQUENCE_MAX_LENGTH \
        model.data.train_ds.file_path=$LABELED_SAMPLES_PATH \
        model.data.train_ds.concat_sampling_probabilities=[1] \
        model.data.train_ds.hf_dataset=True \
        exp_manager.create_wandb_logger=True \
        exp_manager.wandb_logger_kwargs.name=$PROJECT_NAME \
        exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME \
        exp_manager.explicit_log_dir=$OUTPUT_PATH \
        ++exp_manager.max_time_per_run=$REMAIN_TIME \
        exp_manager.checkpoint_callback_params.monitor=global_step \
        exp_manager.checkpoint_callback_params.mode=max \
        ++exp_manager.checkpoint_callback_params.save_top_k=2 \
        ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True
EOF

echo $training_commands &>>$LOGS_PATH
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNT" bash -c "$training_commands" &>>$LOGS_PATH
checkSuccess "SFTTrainer" &>>$LOGS_PATH

echo "Training done." &>>$LOGS_PATH
