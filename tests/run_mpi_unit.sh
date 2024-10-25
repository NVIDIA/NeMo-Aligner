#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

NUM_GPUS_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ $NUM_GPUS_AVAILABLE -lt 2 ]]; then
    echo "[ERROR]: Unit tests require at least 2 gpus"
    exit 1
fi

export PYTHONPATH=$(realpath ..):${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root pytest .. -rA -s -x -vv --mpi $@ || true

if [[ -f PYTEST_SUCCESS ]]; then
    echo SUCCESS
else
    echo FAILURE
    exit 1
fi
