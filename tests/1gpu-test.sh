#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

NUM_GPUS_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ $NUM_GPUS_AVAILABLE -lt 2 ]]; then
    echo "[ERROR]: Unit tests require at least 2 gpus"
    exit 1
fi

export PYTHONPATH=$(realpath ..):${PYTHONPATH:-}
export PYTHONPATH=/home/terryk/aligner-mcore-inf/megatron-lm:$PYTHONPATH
#export PYTHONPATH=/home/terryk/aligner-mcore-inf/NeMo:$PYTHONPATH
PP=1 CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --allow-run-as-root pytest .. -rA -s -x -vv --mpi -k test_trtllm_matches_mcore_inf $@ 2>&1 | tee log || true

if [[ -f PYTEST_SUCCESS ]]; then
    echo SUCCESS
else
    echo FAILURE
    exit 1
fi
