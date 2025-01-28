#!/usr/bin/env bash
set -ex

export MAX_JOBS=8
export TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
export PYTRITON_VERSION=0.5.10
export NEMO_TAG=ko3n1g/build/improve-installer # On: main
export MLM_TAG=                                # On: main
export ALIGNER_COMMIT=main
export APEX_TAG=main
export TRTLLM_VERSION=v0.13.0
export PROTOBUF_VERSION=4.24.4

cd /opt

(rm -rf NeMo || true) &&
    git clone https://github.com/NVIDIA/NeMo.git &&
    pushd NeMo &&
    git fetch &&
    git checkout ${NEMO_TAG} &&
    bash reinstall.sh &&
    popd

(rm -rf TensorRT-LLM || true) &&
    git clone https://github.com/NVIDIA/TensorRT-LLM.git &&
    pushd TensorRT-LLM &&
    git checkout ${TRTLLM_VERSION} &&
    source docker/common/install_tensorrt.sh &&
    python3 ./scripts/build_wheel.py --job_count $(nproc) --trt_root /usr/local/tensorrt --python_bindings --benchmarks &&
    pip install -e .

(rm -rf NeMo-Aligner || true) &&
    git clone https://github.com/NVIDIA/NeMo-Aligner.git &&
    pushd NeMo-Aligner &&
    git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge' &&
    git checkout -f $ALIGNER_COMMIT &&
    # case 1: ALIGNER_COMMIT is a local branch so we have to apply remote changes to it
    # case 2: ALIGNER_COMMIT is a commit, so git-pull is expected to fail
    git pull --rebase || true &&
    pip install --no-cache-dir -e . &&
    popd
