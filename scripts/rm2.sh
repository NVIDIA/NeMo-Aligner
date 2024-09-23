#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -N 4
#SBATCH -t 4:00:00
#SBATCH -J llmservice_modelalignment_ppo-reward_model:1-1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch_block1,batch_block3,batch_block4

# TODO: define them in bashrc
RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"

# first run: lr=1e-6
NAME="rm_baseline_lr1e-5_bsz512-helpsteer"
COMMIT_ID=f3f9dc4
#CONTAINER="${RLHF_SHARED_DIR}/containers/nemo-aligner:v2-022924-nemo-1.23.0.sqsh"
#CONTAINER="/lustre/fsw/portfolios/llmservice/users/yidong/data/models/images/nvidian+nemo+aligner.sqsh"
CONTAINER="gitlab-master.nvidia.com/dl/joc/nemo-ci/main/train:pipe.16440368-x86"

# PARAMETERS
CONFIG_NAME="training_rm"
DATASET_DIR="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl"
TRAIN_DATA_PATH="${DATASET_DIR}/scale_2.2_train_comparisons.jsonl"
VALID_DATA_PATH="${DATASET_DIR}/scale_2.2_val_comparisons.jsonl"
PRETRAINED_CHECKPOINT_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/15b_8T_ct3_daring-anteater_lr3e-6/step3800.nemo"

# W&B Logging
WANDB_PROJECT="reward_model"

# Directory management
RESULTS_DIR="${DATA_DIR}/exp/reward_model/${NAME}"
mkdir -p ${RESULTS_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
    #git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/NeMo-Aligner.git
    git clone https://github.com/abukharin3/NeMo-Aligner.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout -B reward_modeling ${COMMIT_ID} || exit 1
popd
popd

# Config directory
CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"

OUTFILE="${RESULTS_DIR}/slurm-%j.out"
ERRFILE="${RESULTS_DIR}/error-%j.out"

MOUNTS="--container-mounts=${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${RESULTS_DIR}:${RESULTS_DIR},${PRETRAINED_CHECKPOINT_NEMO_FILE}:${PRETRAINED_CHECKPOINT_NEMO_FILE},${DATASET_DIR}:${DATASET_DIR},${DATA_DIR}:${DATA_DIR},/lustre:/lustre"


read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB_API_KEY} \
&& echo "Starting training" \
&& cd ${NEMO_RLHF_DIR} \
&& export HYDRA_FULL_ERROR=1 \
&& python -u ${NEMO_RLHF_DIR}/examples/nlp/gpt/train_reward_model.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    trainer.num_nodes=4 \
    trainer.rm.max_epochs=5 \
    trainer.rm.save_interval=100 \
    trainer.rm.val_check_interval=20 \
    trainer.rm.limit_val_batches=16 \
    trainer.rm.iterative_data_smoothing=False \
    trainer.rm.beta=0.7 \
    model.global_batch_size=512 \
    model.micro_batch_size=8 \
    pretrained_checkpoint.restore_from_path=\"${PRETRAINED_CHECKPOINT_NEMO_FILE}\" \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    exp_manager.checkpoint_callback_params.save_top_k=0 \
    exp_manager.explicit_log_dir=\"${RESULTS_DIR}\" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    exp_manager.wandb_logger_kwargs.name=\"${NAME}\" \
    ++model.tensor_model_parallel_size=4 \
    model.optim.lr=1e-5 \
    model.optim.sched.min_lr=0.999e-6 \
    model.data.data_impl=jsonl \
&& echo "Done"
EOF

srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
set +x