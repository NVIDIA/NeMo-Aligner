#!/bin/bash
#SBATCH -N 1 --gpus-per-node=8 --ntasks-per-node 8 -A <<ACCOUNT>> --job-name reinforce.rm -t 04:00:00 --exclusive --dependency singleton --partition=<<PARTITION>>
#SBATCH hetjob
#SBATCH -N 16 --gpus-per-node=8 --ntasks-per-node 8 -A <<ACCOUNT>> --job-name reinforce.actor -t 04:00:00 --exclusive --dependency singleton --partition=<<PARTITION>>

export NVTE_APPLY_QK_LAYER_SCALING=1
USE_FLASK=False

GBS=64
NUM_ROLLOUT_SAMPLES=64

KL=0.01
LR=5e-7
USE_TRT_LLM=True
USE_RESHARD=True
PP=4
MIN_LR=4e-7

NAME="round1_70b_kl_${KL}_LR_${LR}_${NUM_ROLLOUT_SAMPLES}_trt_llm_${USE_TRT_LLM}_use_reshard_${USE_RESHARD}_pp_${PP}_with_zarr_1_epochs_gbs64"

# PARAMETERS
RM_NEMO_FILE="/path_to/Llama-3.1-Nemotron-70B-Reward/"
ACTOR_NEMO_FILE="/path_to/llama3.1/70b_instruct"

TRAIN_DATA_PATH="/helpsteer2_in_llama3.1_template.jsonl"
VALID_DATA_PATH="/helpsteer2_in_llama3.1_template_val.jsonl"


RESULTS_DIR="results/${NAME}"
mkdir -p $RESULTS_DIR

NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    git clone https://github.com/abukharin3/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout -B reinforce-trtllm || exit 1
popd
popd

MOUNTS="--container-mounts=${RESULTS_DIR}:${RESULTS_DIR},${RM_NEMO_FILE}:${RM_NEMO_FILE},${ACTOR_NEMO_FILE}:${ACTOR_NEMO_FILE}"


CONTAINER="/containers/nemo:24.07.framework.sqsh"
PROJECT=llama3.1-TRT-70B
export WANDB_API_KEY="..."

CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CRITIC_CONFIG_NAME="inference_rm"

CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j_%t.log"
CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j_%t.err"
CRITIC_PORT=5567

mkdir -p $CRITIC_LOG_DIR
CRITIC_NAME="${NAME}_critic"



read -r -d '' cmd_critic_inference <<EOF
export WANDB_API_KEY=${WANDB_API_KEY}  \
&& cd ${NEMO_RLHF_DIR} \
&& export HYDRA_FULL_ERROR=1 \
&& export NCCL_DEBUG="WARN" \
&& export HF_TOKEN="..." \
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

srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" & pids[0]=$!

# START HETEROGENEUS JOB 3
CONF_DIR="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
CONFIG_NAME="gpt_reinforce_actor"

ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

PPO_ERRFILE="${ACTOR_LOG_DIR}/actor_error_%j_%t.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/actor_output_%j_%t.log"

mkdir -p $ACTOR_LOG_DIR
mkdir -p $TENSOBOARD_DIR
mkdir -p $CHECKPOINT_DIR

ACTOR_NAME="${NAME}_actor"

host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"

read -r -d '' cmd_ppo <<EOF
export WANDB_API_KEY=${WANDB_API_KEY} \
&& cd ${NEMO_RLHF_DIR} \
&& export NCCL_DEBUG="WARN" \
&& export HYDRA_FULL_ERROR=1 \
&& export HF_TOKEN="..." \
&& /usr/bin/python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
    --config-path=${CONF_DIR} \
    --config-name=${CONFIG_NAME} \
    "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    pretrained_checkpoint.restore_from_path=${ACTOR_NEMO_FILE} \
    exp_manager.explicit_log_dir=${ACTOR_LOG_DIR} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.create_checkpoint_callback=True \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES_HET_GROUP_1} \
    trainer.devices=8 \
    trainer.reinforce.trt_llm.enable=${USE_TRT_LLM} \
    trainer.reinforce.trt_llm.reshard=${USE_RESHARD} \
    trainer.reinforce.trt_llm.unload_engine_train=False \
    trainer.reinforce.trt_llm.model_type=llama \
    trainer.reinforce.val_check_interval=5 \
    ++trainer.reinforce.max_steps=318 \
    ++trainer.reinforce.save_interval=5 \
    ++exp_manager.checkpoint_callback_params.save_top_k=10 \
    ++model.activations_checkpoint_granularity=full \
    ++model.activations_checkpoint_method=uniform \
    ++model.activations_checkpoint_num_layers=1 \
    ++model.micro_batch_size=1 \
    ++model.global_batch_size=${GBS} \
    ++model.reinforce.rollout_micro_batch_size=4 \
    ++model.reinforce.forward_micro_batch_size=4 \
    ++model.tensor_model_parallel_size=8 \
    ++model.reinforce.num_rollout_samples=${NUM_ROLLOUT_SAMPLES} \
    ++model.pipeline_model_parallel_size=${PP} \
    ++model.reinforce.entropy_bonus=0.0 \
    ++model.reinforce.length_params.max_length=2048 \
    trainer.reinforce.initial_policy_kl_penalty="${KL}" \
    trainer.reinforce.rollout_batch_seq_length=4096 \
    ++model.optim.lr=${LR} \
    trainer.reinforce.batch_iterator.use_flask=${USE_FLASK} \
    ++model.optim.sched.min_lr=${MIN_LR} \
    ++model.optim.bucket_cap_mb=200 \
    ++model.dist_ckpt_format=zarr \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++model.enable_nge=True \
    remote_critic_rm.reward_model.ip=${host_critic} \
    remote_critic_rm.reward_model.port=${CRITIC_PORT}
EOF

srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --mpi=pmix --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}" & pids[1]=$!

# END HETEROGENEUS JOB 3

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