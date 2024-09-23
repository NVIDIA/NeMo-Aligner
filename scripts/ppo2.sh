#!/bin/bash
#SBATCH -N 4 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_ppo-rlfh:38-6b_critic+reward -t 4:00:00 --exclusive --dependency singleton --partition=batch_block1,batch_block3,batch_block4 --gpus-per-node=8
#SBATCH hetjob
#SBATCH -N 16 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_ppo-rlfh:38-6b_actor+init_policy -t 4:00:00 --exclusive --dependency singleton --partition=batch_block1,batch_block3,batch_block4 --gpus-per-node=8

# first run: instability issues (f855d790)
# b: enabling the higher stability log softmax (7d009157)
RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"
NAME="ppo-15bcont-vg-hh-rkl1e-2-lr1e-7-large"
COMMIT_ID=7d009157
CONTAINER="${RLHF_SHARED_DIR}/containers/nvidian+nemo+aligner.sqsh"

echo "Starting job at $(date '+%Y-%m-%d %H:%M:%S')"

RESULTS_DIR="${DATA_DIR}/exp/rlhf/${NAME}"
mkdir -p ${RESULTS_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    #git clone git@github.com:NVIDIA/NeMo-Aligner.git
    git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout ${COMMIT_ID} || exit 1
popd
popd

NUM_ROLLOUTS=512
NORMALIZE="True"
ACTOR_LR="1e-7"
ACTOR_GBS=512
CRITIC_GBS=64

NORMALIZE_REWARD=True
REWARD_MEAN=0.27
REWARD_STD=1

# PARAMETERS
DATASET="anthropic_hh"
#RM_NEMO_FILE="${DATA_DIR}/exp/rlhf/38-19_rm-15bct2-mx-hh-lr3e-6/checkpoints/megatron_gpt.nemo"
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/ealonso/exp/reward_model/01_02_resume_15b_helpsteer_1ep/checkpoints/megatron_gpt.nemo"
ACTOR_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/15b_8T_ct3_vegan-skunk_lr3e-6/step2400.nemo"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="${DATASET_DIR}/${DATASET}_train_prompts.jsonl"
VALID_DATA_PATH="${DATASET_DIR}/${DATASET}_val_prompts_shuffled_512.jsonl"

MOUNTS="--container-mounts=${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"

# W&B Logging
WANDB_PROJECT="reinforce-ablation"

# START HETEROGENEUS JOB 0 =======================================================
CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CRITIC_CONFIG_NAME="gpt_ppo_critic"
CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j.log"
CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j.err"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR

CRITIC_NAME="${NAME}_critic"

read -r -d '' cmd_critic_inference <<EOF
wandb login ${WANDB_API_KEY} \
&& cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=${CRITIC_CONFIG_PATH} \
    --config-name=${CRITIC_CONFIG_NAME} \
    exp_manager.explicit_log_dir=\"${CRITIC_LOG_DIR}\" \
    exp_manager.checkpoint_callback_params.save_top_k=0 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=\"${CRITIC_NAME}\" \
    exp_manager.wandb_logger_kwargs.project=\"${WANDB_PROJECT}\" \
    trainer.num_nodes=4 \
    trainer.devices=8 \
    trainer.ppo.max_steps=5_000_000 \
    trainer.ppo.inference_micro_batch_size=2 \
    ++model.tensor_model_parallel_size=4 \
    model.global_batch_size=${CRITIC_GBS} \
    model.micro_batch_size=1 \
    model.optim.sched.warmup_steps=0 \
    model.offload_adam_states=true \
    model.reward_standardization.enable=${NORMALIZE_REWARD} \
    model.reward_standardization.mean=${REWARD_MEAN} \
    model.reward_standardization.std=${REWARD_STD} \
    pretrained_checkpoint.restore_from_path=${RM_NEMO_FILE} \
    trainer.ppo.port=${CRITIC_PORT}
EOF

srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" & pids[0]=$!

# END HETEROGENEUS JOB 0

sleep 30
#########################################################################################

# START HETEROGENEUS JOB 1

CONF_DIR="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CONFIG_NAME="gpt_ppo_actor"

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
wandb login ${WANDB_API_KEY} \
&& cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTRITON_HOME=/pytriton_cache \
&& export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
&& python -u examples/nlp/gpt/train_gpt_ppo_actor.py \
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
    trainer.ppo.max_epochs=1 \
    trainer.ppo.max_steps=313 \
    trainer.ppo.val_check_interval=4 \
    trainer.num_nodes=16 \
    trainer.devices=8 \
    ++model.tensor_model_parallel_size=4 \
    model.global_batch_size=${ACTOR_GBS} \
    model.micro_batch_size=1 \
    model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
    model.optim.sched.warmup_steps=0 \
    model.optim.sched.constant_steps=312 \
    model.optim.sched.min_lr=${ACTOR_LR} \
    model.optim.weight_decay=0.01 \
    model.ppo.entropy_bonus=0.0 \
    model.ppo.num_rollout_samples=${NUM_ROLLOUTS} \
    trainer.ppo.normalize_advantages=${NORMALIZE} \
    model.ppo.ratio_eps=0.2 \
    trainer.ppo.initial_policy_kl_penalty=1e-2 \
    model.ppo.rollout_micro_batch_size=16 \
    model.ppo.forward_micro_batch_size=16 \
    model.ppo.val_rollout_micro_batch_size=16 \
    model.data.data_impl=jsonl \
    trainer.ppo.use_absolute_kl=false \
    remote_critic_rm.critic.ip=${host_critic} \
    remote_critic_rm.critic.port=${CRITIC_PORT}
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
