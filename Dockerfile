# To build NeMo-Aligner from a base PyTorch container:
#
#   docker buildx build -t aligner:latest .
#
# To update NeMo-Aligner from a pre-built NeMo-Framework container:
#
#   docker buildx build --target=aligner-bump --build-arg=BASE_IMAGE=nvcr.io/nvidia/nemo:24.07 -t aligner:latest .
#

# Number of parallel threads for compute heavy build jobs
# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8
# Git refs for dependencies
ARG TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
ARG PYTRITON_VERSION=0.5.10
ARG NEMO_TAG=e033481e26e6ae32764d3e2b3f16afed00dc7218  # On: r2.0.0rc1
ARG MLM_TAG=a3fe0c75df82218901fa2c3a7c9e389aa5f53182  # On: core_r0.8.0
ARG ALIGNER_COMMIT=main
ARG TRTLLM_VERSION=v0.10.0
ARG PROTOBUF_VERSION=4.24.4

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.03-py3

# NeMo Aligner
FROM ${BASE_IMAGE} AS aligner-bump
ARG ALIGNER_COMMIT
WORKDIR /opt
RUN <<"EOF" bash -exu
    if [[ ! -d NeMo-Aligner ]]; then
        git clone https://github.com/NVIDIA/NeMo-Aligner.git
    fi
    cd NeMo-Aligner
    git fetch -a
    # -f since git status may not be clean
    git checkout -f $ALIGNER_COMMIT
    # case 1: ALIGNER_COMMIT is a local branch so we have to apply remote changes to it
    # case 2: ALIGNER_COMMIT is a commit, so git-pull is expected to fail
    git pull --rebase || true

    pip install --no-deps -e .
EOF

# TRTLLM
FROM ${BASE_IMAGE} AS trtllm-build
WORKDIR /opt
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install
COPY --from=aligner-bump /opt/NeMo-Aligner/setup/trtllm.patch /opt/NeMo-Aligner/setup/trtllm.patch
ARG TRTLLM_VERSION
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git checkout ${TRTLLM_VERSION} && \
    patch -p1 < ../NeMo-Aligner/setup/trtllm.patch && \
    . docker/common/install_tensorrt.sh && \
    python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt 

# TE
FROM ${BASE_IMAGE} AS te-build
ARG TE_TAG
WORKDIR /opt
RUN pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    if [ ! -z $TE_TAG ]; then \
        git fetch origin $TE_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    git submodule init && git submodule update && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip wheel .

FROM nemoci.azurecr.io/nemo_aligner_container_trtllm_build:${TE_TAG} as trt-llm-proxy

# Final image
FROM ${BASE_IMAGE} AS final
WORKDIR /opt
# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"
# install TransformerEngine
ARG MAX_JOBS
COPY --from=te-build /opt/TransformerEngine/ /opt/TransformerEngine/
RUN pip uninstall -y transformer-engine && \
    pip install /opt/TransformerEngine/*.whl

# install latest apex
ARG APEX_TAG
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    if [ ! -z $APEX_TAG ]; then \
        git fetch origin $APEX_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# place any util pkgs here
ARG PYTRITON_VERSION
RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==$PYTRITON_VERSION
ARG PROTOBUF_VERSION
RUN pip install -U --no-deps protobuf==$PROTOBUF_VERSION
RUN pip install --upgrade-strategy only-if-needed jsonlines

# NeMo
ARG NEMO_TAG
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git pull && \
    if [ ! -z $NEMO_TAG ]; then \
        git fetch origin $NEMO_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    pip install -e ".[nlp]" && \
    cd nemo/collections/nlp/data/language_modeling/megatron && make

# MLM
ARG MLM_TAG
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git pull && \
    if [ ! -z $MLM_TAG ]; then \
        git fetch origin $MLM_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -e .

# We have this a bit weird copy order to not break the cache
# (as aligner will certainly update stuff and so successive layers
COPY --from=trt-llm-proxy /opt/TensorRT-LLM /opt/TensorRT-LLM
# TRTLLM
RUN cd /opt/TensorRT-LLM && \
    pip install ./build/tensorrt_llm*.whl
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat/lib.real/

# WAR(0.4.0): The pin of NeMo requires a higher nvidia-modelopt version than
#             TRT-LLM allows. This installation must follow TRT-LLM and is
#             only necessary when NeMo 2.0.0rc1 is installed with TRT-LLM v10.
RUN pip install --upgrade-strategy only-if-needed nvidia-modelopt==0.13.0

COPY --from=aligner-bump /opt/NeMo-Aligner /opt/NeMo-Aligner

RUN cd /opt/NeMo-Aligner && \
    pip install --no-deps -e .
