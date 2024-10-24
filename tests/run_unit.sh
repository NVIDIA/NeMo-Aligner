#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

NUM_GPUS_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ $NUM_GPUS_AVAILABLE -lt 2 ]]; then
    echo "[ERROR]: Unit tests require at least 2 gpus"
    exit 1
fi

export PYTHONPATH=$(realpath ..):${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m pytest .. -rA -s -x $@
