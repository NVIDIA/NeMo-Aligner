# CUDA 12.3
FROM nvcr.io/nvidia/pytorch:24.01-py3

### config tags
ARG APEX_TAG=master
ARG TE_TAG=release_v1.4
ARG MLM_TAG=43792028f003ed25a3ee8c5a0d4cad82317d81b5
ARG NEMO_TAG=9d86acd5ebf3cec020f84dfe7e25c109506803b1 
ARG PYTRITON_VERSION=0.4.1
ARG PROTOBUF_VERSION=4.24.4
ARG ALIGNER_COMMIT=main

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
    pip install install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

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

WORKDIR /workspace
