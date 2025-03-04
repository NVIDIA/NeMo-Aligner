#!/bin/bash

#SBATCH -A [ACCOUNT]
#SBATCH -J [JOBNAME]
#SBATCH -t 4:00:00 
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton 

# This script assumes you have $HUGGINGFACE_TOKEN and $WANDB_API_KEY set in your bash environment prior to running this script
echo "JOBID $SLURM_JOB_ID"

export NAME=run_name
EXP_DIR=experiment_dir

INFTIME=parent_dir
export TRAIN_OUTDIR="/results/${NAME}"

export HF_HOME=huggingface_home_dir

NEMO_CONTAINER=path_to_nemo2412_container

MOUNTS="/lustre:/lustre/,\
${INFTIME}/${EXP_DIR}/code//NeMo:/opt/NeMo,\
${INFTIME}/${EXP_DIR}/code/NeMo-Aligner:/opt/NeMo-Aligner,\
${INFTIME}/NeMo-Skills:/opt/NeMo-Skills,\
${INFTIME}/checkpoints:/opt/checkpoints,\
${INFTIME}/${EXP_DIR}/results/${NAME}:${TRAIN_OUTDIR},\
${INFTIME}/prompts:/opt/prompts,\
${INFTIME}/datasets:/opt/datasets"

OUTDIR=${INFTIME}/${EXP_DIR}/outputs
mkdir -p $OUTDIR
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/policy
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/policy/error
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/policy/output
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/grader
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/grader/error
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/grader/output

mkdir -p ${INFTIME}/${EXP_DIR}/results/
mkdir -p ${INFTIME}/${EXP_DIR}/results/${NAME}

export HOST_GRADER="$(scontrol show hostnames=$SLURM_JOB_NODELIST | head -n1)"
export PORT_GRADER=5567
export RUNDIR="/opt/NeMo-Aligner/examples/nlp/gpt"
export CHECKPOINTS='/opt/checkpoints'
srun -l \
    --container-image ${NEMO_CONTAINER} \
    --container-mounts ${MOUNTS} \
    --no-container-mount-home \
    --export=ALL \
    --ntasks-per-node=8 \
    --gres=gpu:8 \
    --cpus-per-task=16 \
    -o ${OUTDIR}/%j/policy/output/%t.txt \
    -e ${OUTDIR}/%j/policy/error/%t.txt \
    --mpi=pmix bash -c \
    '
    cd ${RUNDIR}
    mkdir -p ${TRAIN_OUTDIR}/generations/${SLURM_JOB_ID}
    TRAIN_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/math/train.jsonl
    VALID_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/math/validation.jsonl
    TEST_DATA_PATH=/opt/NeMo-Skills/nemo_skills/dataset/math/test.jsonl
    PROMPT_FILE=/opt/prompts/cot.txt

    CHECKPOINT=${CHECKPOINTS}/path_to_8b_checkpoint.nemo

    echo HF_HOME $HF_HOME
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    export WANDB_API_KEY=$WANDB_API_KEY
    
    python train_gpt_grpo.py \
        pretrained_checkpoint.restore_from_path=$CHECKPOINT \
        "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
        model.data.prompt_file=${PROMPT_FILE} \
        model.data.apply_chat_template=True \
        trainer.num_nodes=${SLURM_NNODES} \
        trainer.grpo.environments.math.servers.math_grader.ip=${HOST_GRADER} \
        trainer.grpo.environments.math.servers.math_grader.port=${PORT_GRADER} \
        trainer.grpo.val_check_interval=10 \
        trainer.grpo.initial_policy_kl_penalty=0.001 \
        trainer.grpo.use_leave_one_out_baseline=True \
        trainer.grpo.normalize_rewards=True \
        trainer.grpo.generation_rollout_mbs=32 \
        trainer.grpo.prompt_micro_batch_size=1 \
        trainer.grpo.num_prompts_per_grpo_step=32 \
        trainer.grpo.samples_per_prompt=64 \
        trainer.grpo.val_prompt_micro_batch_size=16 \
        trainer.grpo.val_num_prompts_per_grpo_step=512 \
        trainer.grpo.val_samples_per_prompt=2 \
        trainer.grpo.generation_save_dir=${TRAIN_OUTDIR}/generations/${SLURM_JOB_ID}/ \
        trainer.grpo.trt_llm.enable=True \
        trainer.grpo.trt_llm.reshard=False \
        model.grpo.forward_micro_batch_size=4 \
        model.micro_batch_size=1 \
        model.global_batch_size=2048 \
        model.optim.lr=3e-7 \
        model.optim.sched.min_lr=2e-7 \
        model.data.shuffle_train_data=False \
        exp_manager.explicit_log_dir=${TRAIN_OUTDIR} \
        exp_manager.create_wandb_logger=True \
        exp_manager.wandb_logger_kwargs.project=llm_rl_grpo \
        exp_manager.wandb_logger_kwargs.name=${NAME} \
        exp_manager.checkpoint_callback_params.save_top_k=10 \
        exp_manager.create_checkpoint_callback=True \
        +model.tensor_model_parallel_size=4 \
        +model.context_parallel_size=1 \
        ++model.optim.bucket_cap_mb=200 \
        ++model.optim.overlap_grad_sync=False \
        ++model.optim.contiguous_grad_buffer=True \
        ++model.mcore_gpt=True \
        ++model.dist_ckpt_format=torch_dist \
        ++model.dist_ckpt_load_on_device=True \
        ++model.dist_ckpt_parallel_save=True \
        ++model.dist_ckpt_parallel_save_within_dp=False \
        ++model.dist_ckpt_parallel_load=False \
        ++model.dist_ckpt_torch_dist_multiproc=2 \
        ++model.dist_ckpt_assume_constant_structure=False \
        ++model.dist_ckpt_parallel_dist_opt=True \
    ' \
    &
pid_train=$!

EXEC_CONTAINER=path_to_execution_container
srun -l \
    --container-image ${EXEC_CONTAINER} \
    --container-mounts ${MOUNTS} \
    --no-container-mount-home \
    --export=ALL \
    --overlap \
    --ntasks-per-node=1 \
    --gpus=0 \
    -o ${OUTDIR}/%j/grader/output/%t.txt \
    -e ${OUTDIR}/%j/grader/error/%t.txt \
    --mpi=pmix bash -c \
    '
    cd /opt/NeMo-Skills
    
    while true; do python serve_flask_math_grader.py; done
    ' \
    &

echo "Waiting for train ${pid_train} to end. If fails, will cancel at ${SLURM_JOB_ID}"
wait $pid_train || { echo "train job failed"; scancel $SLURM_JOB_ID; exit 1; }
