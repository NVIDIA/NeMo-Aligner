#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eoux pipefail

PRETRAINED_CHECKPOINT_NEMO_FILE=/home/terryk/saved_experiments/tiny-mixtral-nlayers2-hidden128-ffn448-nhead4-qgroup2.nemo \
bash ../dpo.sh \
  ++model.optim.name=mcore_distributed_optim \
  ++model.tensor_model_parallel_size=2 \
  ++model.expert_model_parallel_size=1 \
  ++model.sequence_parallel=True \
  2>&1 | tee $(basename $0 .sh).log
