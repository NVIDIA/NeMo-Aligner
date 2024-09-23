#!/bin/bash
#SBATCH -N 1 --ntasks-per-node 8 -A llmservice_modelalignment_ppo --job-name llmservice_modelalignment_ppo-rlhf:38-3 -t 4:00:00 --exclusive --gpus-per-node=8 --partition=interactive

# List of evaluations performed with this script:
#   rm38-1-hh-val (a9b785b7 / 1cfb27e)
#   rm38-2-hh-val (... / ...)
#   rm38-19-hh-val (2c905d1d / ...)
#   rm38-20-uf-val (... / ...)
#   rm38-21-uf-val (... / ...)
#   rm38-22-hh+uf-val (... / ...)
#   rm38-29-hh-val (... / ...)
#   rm38-29-uf-val (... / ...)
#   rm38-31-uf-val (... / ...)
#   rm38-31-hh-val (... / ...)
#   rm38-32-hh-val (... / ...)
#   rm38-32-uf-val (... / ...)
#   rm38-51d-nectar-val (... / ...)
#   rm38-58b-hh-val (eb15079e / ...)

RLHF_SHARED_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo"
DATA_DIR="/lustre/fsw/portfolios/llmservice/users/abukharin/test"
WANDB_API_KEY="d5c9af701b905bfeadb7a5c7a4c2101afcbf3cc1"

NAME="eval-rm-eloi"
COMMIT_ID=eb86c49
SCRIPTS_COMMIT_ID=1cfb27e
CONTAINER="${RLHF_SHARED_DIR}/containers/nemo-aligner:v2-022924-nemo-1.23.0.sqsh"

EXP_ID="rm38-58b-hh-val-eloirm"
RM_NEMO_FILE="/lustre/fsw/portfolios/llmservice/users/ealonso/exp/reward_model/01_02_resume_15b_helpsteer_1ep/checkpoints/megatron_gpt.nemo"
INPUT_JSONL="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl/anthropic_hh_val_comparisons.jsonl"
#INPUT_JSONL="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl/ultrafeedback_val_comparisons.jsonl"
#INPUT_JSONL="${RLHF_SHARED_DIR}/data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl/nectar_val_comparisons.jsonl"
#INPUT_JSONL="${RLHF_SHARED_DIR}/data/byte_prefix_end_with_backslash_n_x11_jsonl/anthropic_hh_val_comparisons.jsonl"
#INPUT_JSONL="${RLHF_SHARED_DIR}/data/byte_prefix_end_with_backslash_n_x11_jsonl/ultrafeedback_val_comparisons.jsonl"
#INPUT_JSONL="${RLHF_SHARED_DIR}/data/byte_prefix_end_with_backslash_n_x11_jsonl/anthropic_hh_and_ultrafeedback_val_comparisons.jsonl"
TP=4
BS=${5:-"32"}
ADD_EOS=${6:-"0"}

echo "Starting job at $(date '+%Y-%m-%d %H:%M:%S')"

RESULTS_DIR="${DATA_DIR}/exp/rlhf/${NAME}"
OUTPUT_DIR="${RESULTS_DIR}/${EXP_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
NEMO_RLHF_DIR=${RESULTS_DIR}/NeMo-Aligner
NEMO_RLHF_SCRIPTS_DIR=${RESULTS_DIR}/nemo-rlhf-scripts

pushd ${RESULTS_DIR}
if [ ! -d "${NEMO_RLHF_DIR}" ]; then
        git clone https://github.com/abukharin3/NeMo-Aligner.git
        #git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/NeMo-Aligner.git
fi
if [ ! -d "${NEMO_RLHF_SCRIPTS_DIR}" ]; then
        git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/nemo-rlhf-scripts.git
fi
pushd ${NEMO_RLHF_DIR}
git fetch origin
git checkout ${COMMIT_ID} || exit 1
popd
pushd ${NEMO_RLHF_SCRIPTS_DIR}
git fetch origin
git checkout ${SCRIPTS_COMMIT_ID} || exit 1
popd
popd


MOUNTS="--container-mounts=${NEMO_RLHF_DIR}:/rlhf,${NEMO_RLHF_SCRIPTS_DIR}:/rlhf_scripts,${INPUT_JSONL}:${INPUT_JSONL},${RM_NEMO_FILE}:${RM_NEMO_FILE},${RLHF_SHARED_DIR}:${RLHF_SHARED_DIR},${OUTPUT_DIR}:${OUTPUT_DIR},${DATA_DIR}:${DATA_DIR},${DATA_DIR}/c/pytriton:/pytriton_cache,/lustre:/lustre"


# START HETEROGENEUS JOB 0 =======================================================
RM_OUTFILE="${LOG_DIR}/rm_output_%j_%t.log"
RM_ERRFILE="${LOG_DIR}/rm_error_%j_%t.err"

read -r -d '' cmd_rm_inference <<EOF
export PYTRITON_HOME=/pytriton_cache \
&& bash /rlhf_scripts/reward_models/eval_scripts/switch_serve.sh $INPUT_JSONL $OUTPUT_DIR $RM_NEMO_FILE $TP $BS $ADD_EOS
EOF

srun -o $RM_OUTFILE -e $RM_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_rm_inference}"

echo "DONE!"
sleep 3600