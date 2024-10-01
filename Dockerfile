# To build NeMo-Aligner from a base PyTorch container:
#
#   docker buildx build -t aligner:latest .
#
# To update NeMo-Aligner from a pre-built NeMo-Framework container:
#
#   docker buildx build --target=aligner-bump -t aligner:latest .
#

# Number of parallel threads for compute heavy build jobs
# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8
# Git refs for dependencies
ARG TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
ARG PYTRITON_VERSION=0.5.10
ARG NEMO_TAG=8332f43ee27b3b24406303fc0aa0056d9e924420  # On: r2.0.0
ARG MLM_TAG=3d9d28a0d09d273740d88ffc70520c17e53c36b8  # On: core_r0.9.0
ARG ALIGNER_COMMIT=main
ARG TRTLLM_VERSION=v0.12.0
ARG PROTOBUF_VERSION=4.24.4
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3

FROM ${BASE_IMAGE} AS aligner-bump
ARG ALIGNER_COMMIT
WORKDIR /opt
# NeMo Aligner
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

pip install --no-cache-dir --no-deps -e .
EOF

FROM ${BASE_IMAGE} as final
WORKDIR /opt
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
    . docker/common/install_tensorrt.sh && \
    python3 ./scripts/build_wheel.py --job_count $(nproc) --trt_root /usr/local/tensorrt  --python_bindings --benchmarks

RUN cd TensorRT-LLM && \
    pip install -e .
RUN cd TensorRT-LLM && patch -p1 < ../NeMo-Aligner/setup/trtllm.patch
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat/lib.real/

COPY --from=aligner-bump /opt/NeMo-Aligner /opt/NeMo-Aligner
RUN cd /opt/NeMo-Aligner && \
    pip install --no-deps -e .

RUN <<"EOF" bash -exu
cd NeMo
# Ensures we don't cherry-pick "future" origin/main commits
git fetch -a
# 10654: feat: Migrate GPTSession refit path in Nemo export to ModelRunner for Aligner NeMo#10654
# 10651: [fix] Ensures disabling exp_manager with exp_manager=null does not error NeMo#10651
# 10652: [feat] Update get_model_parallel_src_rank to support tp-pp-dp ordering NeMo#10652
# 10653: fix: MegatronGPTModel get_forward_output_only_func position_ids=None NeMo#10653
for pr in 10654 10651 10652 10653; do
  git fetch origin pull/${pr}/head:PR-${pr}
  # cherry-picks all commits between main and the top of the PR
  git cherry-pick --allow-empty $(git merge-base origin/main PR-${pr})..PR-${pr}
  # Tag cherry-picks to help
  git tag cherry-pick-PR-${pr}
done
EOF
