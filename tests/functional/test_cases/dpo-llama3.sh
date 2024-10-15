#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eoux pipefail

PRETRAINED_CHECKPOINT_NEMO_FILE=${ALIGNER_CI_DIR}//tiny-llama3-results-nlayers2-hidden128-ffn448-nhead4-qgroup2-megatron_gpt.nemo \
bash /opt/NeMo-Aligner/tests/functional/dpo.sh