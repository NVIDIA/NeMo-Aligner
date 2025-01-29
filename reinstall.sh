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

#!/bin/bash

# List of all supported libraries (update this list when adding new libraries)
ALL_LIBRARIES=(
    "nemo"
    "trtllm"
    "aligner"
)

# --------------------------
# Library Functions (Implement your logic here)
# --------------------------

nemo() {
    local mode="$1"
    cd /opt

    (rm -rf NeMo || true) &&
        git clone https://github.com/NVIDIA/NeMo.git &&
        pushd NeMo &&
        git fetch &&
        git checkout ${NEMO_TAG} &&
        bash reinstall.sh &&
        popd
}

trtllm() {
    local mode="$1"
    cd /opt
    
    (rm -rf TensorRT-LLM || true) &&
        git clone https://github.com/NVIDIA/TensorRT-LLM.git &&
        pushd TensorRT-LLM &&
        git checkout ${TRTLLM_VERSION}

    if [[ "$mode" == "build" ]]; then
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash &&
            apt-get install git-lfs &&
            git lfs install &&
            git lfs pull &&
            apt-get clean

        . docker/common/install_tensorrt.sh &&
            python3 ./scripts/build_wheel.py --job_count $(nproc) --trt_root /usr/local/tensorrt --python_bindings --benchmarks
    else
        pip install /tmp/build/trtllm*.whl
    fi
}

aligner() {
    local mode="$1"

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
}

# --------------------------
# Argument Parsing & Validation
# --------------------------

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --library)
        LIBRARY_ARG="$2"
        shift 2
        ;;
    --mode)
        MODE="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Validate required arguments
if [[ -z "$LIBRARY_ARG" ]]; then
    echo "Error: --library argument is required"
    exit 1
fi

if [[ -z "$MODE" ]]; then
    echo "Error: --mode argument is required"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "build" && "$MODE" != "install" ]]; then
    echo "Error: Invalid mode. Must be 'build' or 'install'"
    exit 1
fi

# Process library argument
declare -a LIBRARIES
if [[ "$LIBRARY_ARG" == "all" ]]; then
    LIBRARIES=("${ALL_LIBRARIES[@]}")
else
    IFS=',' read -ra TEMP_ARRAY <<<"$LIBRARY_ARG"
    for lib in "${TEMP_ARRAY[@]}"; do
        trimmed_lib=$(echo "$lib" | xargs)
        if [[ -n "$trimmed_lib" ]]; then
            LIBRARIES+=("$trimmed_lib")
        fi
    done
fi

# Validate libraries array
if [[ ${#LIBRARIES[@]} -eq 0 ]]; then
    echo "Error: No valid libraries specified"
    exit 1
fi

# Validate each library is supported
for lib in "${LIBRARIES[@]}"; do
    if [[ ! " ${ALL_LIBRARIES[@]} " =~ " ${lib} " ]]; then
        echo "Error: Unsupported library '$lib'"
        exit 1
    fi
done

# --------------------------
# Execution Logic
# --------------------------

# Run operations for each library
for library in "${LIBRARIES[@]}"; do
    echo "Processing $library ($MODE)..."
    "$library" "$MODE"

    # Check if function succeeded
    if [[ $? -ne 0 ]]; then
        echo "Error: Operation failed for $library"
        exit 1
    fi
done

echo "All operations completed successfully"
exit 0
