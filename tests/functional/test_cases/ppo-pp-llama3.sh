#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eoux pipefail

RM_NEMO_FILE=${ALIGNER_CI_DIR}/checkpoints/llama3--nlayers4-hidden64-ffn224-dummy_rm-megatron_gpt.nemo \
ACTOR_NEMO_FILE=${ALIGNER_CI_DIR}/checkpoints/tiny-llama3-results-nlayers2-hidden128-ffn448-nhead4-qgroup2-megatron_gpt.nemo \
bash ../ppo.sh