#!/bin/bash
#SBATCH -N 16 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rloo:42-9_critic+reward -t 4:00:00 --dependency singleton --exclusive --partition=batch_block1,batch_block3,batch_block4 --gpus-per-node=8
#SBATCH hetjob
#SBATCH -N 2 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_rloo:42-9_actor+init_policy -t 4:00:00 --dependency singleton --exclusive --partition=batch_block1,batch_block3,batch_block4 --gpus-per-node=8

RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"

NAME="ablation-eloirm-rloo-hh-lr1e-6-bsz64-kl0.01"
COMMIT_ID=770873d
CONTAINER="${RLHF_SHARED_DIR}/containers/nemo-aligner:v2-022924-nemo-1.23.0.sqsh"

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
git checkout -B debug ${COMMIT_ID} || exit 1
popd
popd

NUM_ROLLOUTS=128
NORMALIZE="True"
ACTOR_LR="1e-7"
ACTOR_GBS=512
CRITIC_GBS=64

NORMALIZE_REWARD=True
REWARD_MEAN=-0.5032332877024343
REWARD_STD=3.377394892211421


# PARAMETERS
DATASET="anthropic_hh"
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/ealonso/exp/reward_model/01_02_resume_15b_helpsteer_1ep/checkpoints/megatron_gpt.nemo"
ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/15b_8T_ct3_vegan-skunk_lr3e-6/step2400.nemo"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="${DATASET_DIR}/${DATASET}_train_prompts.jsonl"
VALID_DATA_PATH="${DATASET_DIR}/${DATASET}_val_prompts_shuffled_512.jsonl"

MOUNTS="--container-mounts=${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"

# W&B Logging
WANDB_PROJECT="reinforce-ablation-2"

# START HETEROGENEUS JOB 0 =======================================================
CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CRITIC_CONFIG_NAME="inference_rm"
CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j.log"
CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j.err"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR

CRITIC_NAME="${NAME}_critic"

# NB: we set `attribute_weights` to only use the Helpfulness attribute

read -r -d '' cmd_critic_inference <<EOF
export WANDB_API_KEY=${WANDB_API_KEY}  \
&& cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=${CRITIC_CONFIG_PATH} \
    --config-name=${CRITIC_CONFIG_NAME} \
    trainer.num_nodes=2 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    model.reward_standardization.enable=${NORMALIZE_REWARD} \
    model.reward_standardization.mean=${REWARD_MEAN} \
    model.reward_standardization.std=${REWARD_STD} \
    rm_model_file=${RM_NEMO_FILE} \
    inference.port=${CRITIC_PORT}
EOF

srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" & pids[0]=$!

# END HETEROGENEUS JOB 0

sleep 30
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
export WANDB_API_KEY=${WANDB_API_KEY} \
&& cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& export CUDA_LAUNCH_BLOCKING=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
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
    trainer.ppo.initial_policy_kl_penalty=0.01 \
    trainer.ppo.max_epochs=1 \
    trainer.ppo.max_steps=313 \
    trainer.ppo.val_check_interval=4 \
    trainer.num_nodes=8 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    model.global_batch_size=${ACTOR_GBS} \
    model.micro_batch_size=1 \
    model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
    model.optim.sched.warmup_steps=0 \
    model.optim.sched.constant_steps=312 \
    model.optim.sched.min_lr=${ACTOR_LR} \
    model.optim.weight_decay=0.01 \
    model.ppo.num_rollout_samples=${NUM_ROLLOUTS} \
    model.ppo.rollout_micro_batch_size=16 \
    model.ppo.forward_micro_batch_size=16 \
    model.ppo.val_rollout_micro_batch_size=16 \
    model.data.data_impl=jsonl \
    remote_critic_rm.reward_model.ip=${host_critic} \
    remote_critic_rm.reward_model.port=${CRITIC_PORT} \
    model.ppo.num_rollout_per_prompt=4 \
    trainer.ppo.baseline=\"RLOO\"
EOF

srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}" & pids[1]=$!

# END HETEROGENEUS JOB 1


# The code below monitors the four SLURM jobs to ensure any failure forces them all to stop
# (otherwise some jobs may remain pending until they reach the cluster time limit).
all_done=false
while ! $all_done; do
    all_done=true
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null; then
            # Process is still running.
            all_done=false
        else
            # Process is no longer running => check its exit status.
            wait "$pid"
            exit_code=$?
            echo "Process $pid exited with code $exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
            # Wait a bit (to get a clean stack trace in case there is one being generated), then kill the
            # remaining processes if needed.
            sleep 60
            for other_pid in "${pids[@]}"; do
                if ps -p "$other_pid" > /dev/null; then
                    echo "Killing processs $other_pid"
                    kill -9 "$other_pid"
                fi
            done
            exit $exit_code
        fi
    done

    # Sleep for a while before checking again.
    sleep 60
done

echo "Job terminated successfully"