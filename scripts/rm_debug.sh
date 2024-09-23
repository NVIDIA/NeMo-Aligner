#!/bin/bash
#SBATCH -N 1 --mem=0 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rs_debug16-rlfh:42-9_critic+reward -t 4:00:00 --exclusive --gpus-per-node=8 --partition=interactive

RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"

NAME="minitron-debug"
#COMMIT_ID=f5766ae
COMMIT_ID=85556aa
#CONTAINER="${RLHF_SHARED_DIR}/containers/train:pipe.16440368-x86.sqsh"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/geshen/small_model_alignment/dl+joc+nemo-ci+main+train+pipe.17556452-x86.sqsh"
echo "Starting job at $(date '+%Y-%m-%d %H:%M:%S')"

RESULTS_DIR="${DATA_DIR}/exp/rlhf/${NAME}"
mkdir -p ${RESULTS_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    #git clone git@github.com:NVIDIA/NeMo-Aligner.git
    git clone https://github.com/abukharin3/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout -B minitron ${COMMIT_ID} || exit 1
git submodule update --init --recursive
popd
popd

NUM_ROLLOUTS=8
NORMALIZE="True"
ACTOR_LR="1e-6"
ACTOR_GBS=8
CRITIC_GBS=64

NORMALIZE_REWARD=True
REWARD_MEAN=0s
REWARD_STD=1

# PARAMETERS
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/zhilinw/models/llama31_70b_instruct_regression_helpsteer_v11_0_to_4_helpfulness_only_to_bt_weighted_shuffled_all_weights_1_epochs_constant_lr_1e-6_step_80"
#ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo"
ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/geshen/share/8b_dpo-urban_3.002e-7-kl-1e-3-dpo-loss-rpo_fwd_kl-sft-weight-1e-5_megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0/megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/llama3_code_train.jsonl"
VALID_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/llama3_code_val.jsonl"
HF_HOME="/lustre/fsw/portfolios/llmservice/users/zhilinw/hf_home"
MOUNTS="--container-mounts=$HF_HOME:/hf_home,/lustre/fsw/portfolios/llmservice/users/abukharin/data/:/lustre/fsw/portfolios/llmservice/users/abukharin/data/,${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"

# W&B Logging
WANDB_PROJECT="nemo-aligner-alex-stable"

# START HETEROGENEUS JOB 0 =======================================================
CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CRITIC_CONFIG_NAME="inference_rm"
CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j.log"
CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j.err"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR

CRITIC_NAME="${NAME}_critic"


#########################################################################################

# START HETEROGENEUS JOB 1

CONF_DIR="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CONFIG_NAME="gpt_reinforce_actor"

ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

PPO_ERRFILE="${ACTOR_LOG_DIR}/actor_error_%j.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/actor_output_%j.log"

mkdir -p $ACTOR_LOG_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

ACTOR_NAME="${NAME}_actor"

host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"


read -r -d '' cmd_critic_inference <<EOF
export WANDB_API_KEY=${WANDB_API_KEY}  \
&& cd ${NEMO_RLHF_DIR} \
&& export HYDRA_FULL_ERROR=1 \
&& export HF_TOKEN="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" \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=${CRITIC_CONFIG_PATH} \
    --config-name=${CRITIC_CONFIG_NAME} \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    ++model.regression.num_attributes=9 \
    ++model.regression.merge_attributes=True \
    ++model.regression.attribute_weights="[0, 0, 0, 0, 1, 0, 0, 0, 0]" \
    rm_model_file=${RM_NEMO_FILE} \
    inference.port=${CRITIC_PORT}
EOF

srun -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" & pids[0]=$!

# srun -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}" & pids[1]=$!

sleep 14400

echo "Job terminated successfully"
