#!/bin/bash
#SBATCH -N 1 --mem=0 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rs_debug16-rlfh:42-9_critic+reward -t 4:00:00 --exclusive --gpus-per-node=8 --partition=interactive

RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"

NAME="alex-reinforce-debug-code"
#COMMIT_ID=f5766ae
COMMIT_ID=41a2e43
CONTAINER="${RLHF_SHARED_DIR}/containers/train:pipe.16440368-x86.sqsh"

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
git checkout -B multitask ${COMMIT_ID} || exit 1
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
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/shengyangs/results/rm_llama3_8b_competent-agama_lr_rand_3e-6/checkpoints/megatron_gpt.nemo"
ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/llama3_apps_train.jsonl"
VALID_DATA_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/data/llama3_apps_val.jsonl"

MOUNTS="--container-mounts=/lustre/fsw/portfolios/llmservice/users/abukharin/data/:/lustre/fsw/portfolios/llmservice/users/abukharin/data/,${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"

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


read -r -d '' cmd_ppo <<EOF
pip install immutabledict
pip install nltk
pip install langdetect
echo "import nltk;
nltk.download('all')" > download_nltk_data.py
python download_nltk_data.py
cd ${NEMO_RLHF_DIR}
pip install -e evalplus

export WANDB_API_KEY=${WANDB_API_KEY} \
&& export HF_HOME="/lustre/fsw/portfolios/llmservice/users/abukharin/test/hf_home" \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/train_gpt_reinforce_code.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONFIG_NAME} \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.explicit_log_dir=\"${ACTOR_LOG_DIR}\" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=\"${ACTOR_NAME}\" \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    ++exp_manager.max_time_per_run=\"00:03:30:00\" \
    trainer.reinforce.max_epochs=1 \
    trainer.reinforce.max_steps=313 \
    trainer.reinforce.max_steps=313 \
    trainer.reinforce.initial_policy_kl_penalty=0.01 \
    trainer.reinforce.val_check_interval=10 \
    trainer.num_nodes=1 \
    +trainer.reinforce.beta=0.1 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    model.global_batch_size=${ACTOR_GBS} \
    model.micro_batch_size=1 \
    model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
    model.optim.sched.warmup_steps=0 \
    model.optim.sched.constant_steps=312 \
    model.optim.sched.min_lr=${ACTOR_LR} \
    model.optim.weight_decay=0.01 \
    model.reinforce.num_rollout_samples=${NUM_ROLLOUTS} \
    model.reinforce.rollout_micro_batch_size=4 \
    model.reinforce.forward_micro_batch_size=4 \
    model.reinforce.val_rollout_micro_batch_size=4 \
    model.data.data_impl=jsonl \
    model.reinforce.sampling_params.end_strings="[\"<|endoftext|>\", \"<|eot_id|>\"]" \
    remote_critic_rm.reward_model.ip=${host_critic} \
    remote_critic_rm.reward_model.port=${CRITIC_PORT} \
    model.reinforce.num_rollout_per_prompt=4 \
    trainer.reinforce.baseline=\"RLOO\"
EOF

srun -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}" & pids[1]=$!

sleep 14400

echo "Job terminated successfully"