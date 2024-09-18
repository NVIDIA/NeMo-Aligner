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

FROM ${BASE_IMAGE} AS aligner-bump

ARG ALIGNER_COMMIT

WORKDIR /opt

# NeMo Aligner
RUN <<"EOF" bash -exu
if [[ ! -d NeMo-Aligner ]]; then
    git clone https://github.com/NVIDIA/NeMo-Aligner.git
    cd NeMo-Aligner
    git checkout $ALIGNER_COMMIT
    pip install --no-deps -e .
    cd -
fi
cd NeMo-Aligner
git fetch -a
git checkout -f ${ALIGNER_COMMIT}
git pull
EOF

FROM aligner-bump as final

# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"
# install TransformerEngine
ARG MAX_JOBS
ARG TE_TAG
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

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

# TRTLLM
ARG TRTLLM_VERSION
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git checkout ${TRTLLM_VERSION} && \
    patch -p1 < ../NeMo-Aligner/setup/trtllm.patch && \
    . docker/common/install_tensorrt.sh && \
    python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt 

RUN cd TensorRT-LLM && \
    pip install ./build/tensorrt_llm*.whl
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat/lib.real/

# WAR(0.4.0): The pin of NeMo requires a higher nvidia-modelopt version than
#             TRT-LLM allows. This installation must follow TRT-LLM and is
#             only necessary when NeMo 2.0.0rc1 is installed with TRT-LLM v10.
RUN pip install --upgrade-strategy only-if-needed nvidia-modelopt==0.13.0
