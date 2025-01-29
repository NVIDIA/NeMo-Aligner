#!/usr/bin/env bash
set -ex
cd /opt

# List of all supported libraries (update this list when adding new libraries)
# This also defines the order in which they will be installed by --libraries "all"
ALL_LIBRARIES=(
    "nemo"
    "trtllm"
    "te"
    "apex"
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
        pip install --no-cache-dir /tmp/trtllm/tensorrt_llm*.whl 
        pip install --no-cache-dir pynvml==${PYNVML_VERSION}
    fi
}

te() {
    local mode="$1"
    cd /opt
    
    (rm -rf TransformerEngine || true) &&
        git clone https://github.com/NVIDIA/TransformerEngine.git &&
        pushd TransformerEngine &&
        git checkout ${TE_TAG}

    if [[ "$mode" == "build" ]]; then
        git submodule init && git submodule update && \
        NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip wheel . && \
        ls -al
    else
        pip install /tmp/te/transformerengine*.whl
    fi
}

apex() {
    local mode="$1"
    cd /opt
    
    (rm -rf Apex || true) &&
        git clone https://github.com/NVIDIA/Apex.git &&
        pushd Apex &&
        git checkout ${APEX_TAG}

    if [[ "$mode" == "build" ]]; then
        pip wheel -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./ && \
        ls -al
    else
        pip install /tmp/apex/apex*.whl
    fi
}


aligner() {
    local mode="$1"
    cd /opt

    (rm -rf NeMo-Aligner || true) &&
        git clone https://github.com/NVIDIA/NeMo-Aligner.git &&
        pushd NeMo-Aligner &&
        git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge' &&
        git checkout -f $ALIGNER_COMMIT &&
        # case 1: ALIGNER_COMMIT is a local branch so we have to apply remote changes to it
        # case 2: ALIGNER_COMMIT is a commit, so git-pull is expected to fail
        git pull --rebase || true &&
        pip install --no-cache-dir --upgrade-strategy only-if-needed nvidia-pytriton==$PYTRITON_VERSION &&
        pip install --no-cache-dir -U --no-deps protobuf==$PROTOBUF_VERSION &&
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
