ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.03-py3

FROM ${BASE_IMAGE}
ARG APEX_TAG=23c1f86520e22b505e8fdfcf6298273dff2d93d8
ARG TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
ARG MLM_TAG=core_r0.8.0
ARG NEMO_TAG=r2.0.0rc1
ARG ALIGNER_COMMIT=r0.4.0rc0

ARG PYTRITON_VERSION=0.5.8
ARG PROTOBUF_VERSION=4.24.4
ARG TRTLLM_VERSION=v0.10.0

# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8

# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"

WORKDIR /opt

# install TransformerEngine
RUN pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    if [ ! -z $TE_TAG ]; then \
        git fetch origin $TE_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    git submodule init && git submodule update && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# install latest apex
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    if [ ! -z $APEX_TAG ]; then \
        git fetch origin $APEX_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -e . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# place any util pkgs here
RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==$PYTRITON_VERSION
RUN pip install -U --no-deps protobuf==$PROTOBUF_VERSION
RUN pip install --upgrade-strategy only-if-needed jsonlines

# NeMo
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
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git pull && \
    if [ ! -z $MLM_TAG ]; then \
        git fetch origin $MLM_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -e .

# NeMo Aligner
RUN git clone https://github.com/NVIDIA/NeMo-Aligner.git && \
    cd NeMo-Aligner && \
    git pull && \
    if [ ! -z $ALIGNER_COMMIT ]; then \
        git fetch origin $ALIGNER_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install --no-deps -e .

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

# TRTLLM
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git checkout ${TRTLLM_VERSION} && \
    patch -p1 < ../NeMo-Aligner/trtllm.patch && \
    . docker/common/install_tensorrt.sh && \
    python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt 

RUN cd TensorRT-LLM && \
    pip install ./build/tensorrt_llm*.whl
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/compat/lib.real/
