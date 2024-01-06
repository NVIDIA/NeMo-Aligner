# CUDA 12.2
FROM nvcr.io/nvidia/pytorch:23.10-py3

### config tags
ARG APEX_TAG=master
ARG TE_TAG=release_v1.1
ARG MLM_TAG=core_r0.4.0
ARG NEMO_TAG=r1.22.0

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
#RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--permutation_search" --config-settings "--build-option=--bnp" --config-settings "--build-option=--xentropy" --config-settings "--build-option=--focal_loss" --config-settings "--build-option=--index_mul_2d" --config-settings "--build-option=--deprecated_fused_adam" --config-settings "--build-option=--fast_layer_norm" --config-settings "--build-option=--fmha" --config-settings "--build-option=--fast_multihead_attn" --config-settings "--build-option=--transducer" --config-settings "--build-option=--cudnn_gbn" --config-settings "--build-option=--fused_conv_bias_relu" .
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
	cd apex && \
    if [ ! -z $APEX_TAG ]; then \
        git fetch origin $APEX_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# place any util pkgs here
RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==0.4.1
RUN pip install -U --no-deps protobuf==4.24.4
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
    git cherry-pick --no-commit -X theirs fa8d416793d850f4ce56bea65e1fe28cc0d092c0 a7f0bc1903493888c31436efc2452ff721fa5a67 && \
    sed -i 's/shutil.rmtree(ckpt_to_dir(filepath))/shutil.rmtree(ckpt_to_dir(filepath), ignore_errors=True)/g' nemo/collections/nlp/parts/nlp_overrides.py && \
    rm -rf .git && pip install -e ".[nlp]" && \
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

WORKDIR /opt

# install the latest NeMo-Aligner
RUN pip install --no-deps git+https://github.com/NVIDIA/NeMo-Aligner.git@main

WORKDIR /workspace
