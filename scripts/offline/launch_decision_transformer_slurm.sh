#!/bin/bash

#SBATCH -p <<<PARTITION>>>         
#SBATCH -A <<<ACCOUNT>>>
#SBATCH -J <<<JOB_NAME>>>
#SBATCH -N <<<NUM_NODES>>>         # 64x8x4
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

RLHF_DIR=<<<PATH/TO/nemo-rlhf>>>

# Training settings
LEARNING_RATE=<<<LEARNING_RATE>>> # Constant learning rate
WEIGHT_DECAY=<<<WEIGHT_DECAY>>> # L2 weight decay, such as 0.01
MAX_NUM_SAMPLES=<<<MAX_NUM_SAMPLES>>> # Max training samples

RM_MICRO_BATCH_SIZE=<<<RM_MICRO_BATCH_SIZE>>> # Set it to a large integer to improve inference efficiency
GLOBAL_TRAIN_BATCH_SIZE=<<<GLOBAL_TRAIN_BATCH_SIZE>>>
MICRO_TRAIN_BATCH_SIZE=<<<MICRO_TRAIN_BATCH_SIZE>>>
SEQUENCE_MAX_LENGTH=<<<SEQUENCE_MAX_LENGTH>>> # Maximum sequence length of the model

MAX_TIME_PER_RUN=null # days:hours:mins:secs

# Models settings
POLICY_MODEL_PATH=<<<INIT_POLICY_PATH>>>
REWARD_MODEL_PATH=<<<REWARD_MODEL_PATH>>>

# Datasets settings
RM_DATA_PATH=<<<RM_JSONL_FILE_PATH>>>
SFT_DATA_PATH=<<<SFT_JSONL_FILE_PATH>>>
LABELED_SAMPLES_PATH=<<<LABELED_JSONL_FILE_PATH>>>

# Logs settings
OUTPUT_PATH=<<<LOG_OUTPUT_DIR>>>
mkdir -p $OUTPUT_PATH
LOGS_PATH=$OUTPUT_PATH/$SLURM_JOB_ID.log

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
REMAIN_TIME=null
START_TIME=$(date +%s)

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
# And DT without generation is significantly faster than DT with generation.

# Labeling and process the samples
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
        data.micro_batch_size=$RM_MICRO_BATCH_SIZE \
        data.max_seq_length=$SEQUENCE_MAX_LENGTH \
        data.concat_sampling_probabilities=[0.5,0.5] \
        data.file_names=[$SFT_DATA_PATH,$RM_DATA_PATH] \
        data.num_samples=$MAX_NUM_SAMPLES \
        data.hf_dataset=True \
        reward_standardization.enable=True \
        reward_standardization.mean=null \
        reward_standardization.std=null \
        processor=dt \
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
        trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
        trainer.devices=${SLURM_NTASKS_PER_NODE} \
        trainer.precision=bf16-mixed \
        ++trainer.sft.max_steps=-1 \
        ++trainer.sft.skip_validation=True \
        ++trainer.sft.save_interval=100 \
        model.tensor_model_parallel_size=-1 \
        model.pipeline_model_parallel_size=-1 \
        model.activations_checkpoint_granularity=selective \
        model.activations_checkpoint_method=uniform \
        model.megatron_amp_O2=True \
        model.restore_from_path=$POLICY_MODEL_PATH \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=$LEARNING_RATE \
        model.optim.weight_decay=$WEIGHT_DECAY \
        model.optim.betas=[0.9,0.95] \
        ~model.optim.sched \
        model.answer_only_loss=True \
        model.data.train_ds.micro_batch_size=$MICRO_TRAIN_BATCH_SIZE \
        model.data.train_ds.global_batch_size=$GLOBAL_TRAIN_BATCH_SIZE \
        model.data.train_ds.max_seq_length=$SEQUENCE_MAX_LENGTH \
        model.data.train_ds.file_path=$LABELED_SAMPLES_PATH \
        model.data.train_ds.concat_sampling_probabilities=[1] \
        model.data.train_ds.hf_dataset=True \
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
