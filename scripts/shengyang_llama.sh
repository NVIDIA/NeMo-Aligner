#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -N 8
#SBATCH -t 4:00:00
#SBATCH -J rm-1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit

HF_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_sft/rlhf/checkpoints/community/llama3/hf_home"
RLHF_SHARED_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo
# first run: lr=1e-6
# b: lr=1e-7
# c: lr=3e-7
# d: lr=5e-7
LR=3.0001e-6
MIN_LR=3e-6
NAME="rm_llama3_8b_competent-agama_lr_version_rand_3e-6"
COMMIT_ID=070faee8
CONTAINER="${RLHF_SHARED_DIR}/containers/nemo-aligner:v2-022924-nemo-1.23.0.sqsh"

WANDB_API_KEY="87853532c501f3e0eb65339d361373388081efc6"
WANDB_PROJECT="pref_optimization"

# PARAMETERS
CONFIG_NAME="training_rm"
DATASET_DIR="/lustre/fsw/portfolios/llmservice/users/shengyangs/results/projects/nemotron-synthetic-data/preference_optimization/data/preference/"
TRAIN_DATA_PATH="${DATASET_DIR}/multi_models/rm.rm9200.reject-rand-chosen-min-0.0.train.jsonl"
VALID_DATA_PATH="${DATASET_DIR}/multi_models/rm.rm9200.reject-rand-chosen-min-0.0.val.jsonl"
PRETRAINED_CHECKPOINT_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo"


# Directory management
RESULTS_DIR="/lustre/fsw/portfolios/llmservice/users/shengyangs/results/${NAME}"
mkdir -p ${RESULTS_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    #git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/NeMo-Aligner.git
    git clone git@github.com:NVIDIA/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout ${COMMIT_ID} || exit 1
popd
popd

# Config directory
CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"

OUTFILE="${RESULTS_DIR}/slurm-%j.out"
ERRFILE="${RESULTS_DIR}/error-%j.out"

MOUNTS="--container-mounts=${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${PRETRAINED_CHECKPOINT_NEMO_FILE}:${PRETRAINED_CHECKPOINT_NEMO_FILE},${DATASET_DIR}:${DATASET_DIR},/lustre:/lustre"


read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& export HF_HOME=$HF_HOME \
&& echo "---------------" \
&& wandb login ${WANDB_API_KEY} \
&& echo "Starting training" \
&& cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u ${NEMO_RLHF_DIR}/examples/nlp/gpt/train_reward_model.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.rm.max_epochs=1 \
    trainer.rm.val_check_interval=20 \
    trainer.rm.limit_val_batches=1 \
    model.global_batch_size=512 \
    model.micro_batch_size=1 \
    pretrained_checkpoint.restore_from_path=\"${PRETRAINED_CHECKPOINT_NEMO_FILE}\" \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    exp_manager.checkpoint_callback_params.save_top_k=0 \
    exp_manager.explicit_log_dir=\"${RESULTS_DIR}\" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    exp_manager.wandb_logger_kwargs.name=\"${NAME}\" \
    ++model.tensor_model_parallel_size=4 \
    model.optim.lr=${LR} \
    model.optim.sched.min_lr=${MIN_LR} \
    model.data.data_impl=jsonl \
&& echo "Done"
EOF

srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
set +x