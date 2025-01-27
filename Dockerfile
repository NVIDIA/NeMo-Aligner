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
ARG NEMO_TAG=06eae2895c0fea09f8dd7c34feff0163e55c419a  # On: main
ARG MLM_TAG=844119f5c856a3037ec7c7f6d6ef7b3518ceee6b   # On: main
ARG ALIGNER_COMMIT=main
ARG TRTLLM_VERSION=v0.13.0
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
git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge'
git checkout -f $ALIGNER_COMMIT
# case 1: ALIGNER_COMMIT is a local branch so we have to apply remote changes to it
# case 2: ALIGNER_COMMIT is a commit, so git-pull is expected to fail
git pull --rebase || true

pip install --no-cache-dir --no-deps -e .
EOF

FROM ${BASE_IMAGE} AS final
LABEL "nemo.library"="nemo-aligner"
WORKDIR /opt
# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"
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

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install && \
    apt-get clean

# TRTLLM
ARG TRTLLM_VERSION
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git checkout ${TRTLLM_VERSION} && \
    . docker/common/install_tensorrt.sh && \
    python3 ./scripts/build_wheel.py --job_count $(nproc) --trt_root /usr/local/tensorrt  --python_bindings --benchmarks && \
    pip install -e .
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat/lib.real/

# TODO: This pinning of pynvml is only needed while on TRTLLM v13 since pynvml>=11.5.0 but pynvml==12.0.0 contains a
#   breaking change. The last known working verison is 11.5.3
RUN pip install pynvml==11.5.3

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

# TODO: Allow installing from the default branch, but introduce a build
#  arg if compatibility starts breaking
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/NeMo-Run.git
    
# TODO: While we are on Pytorch 24.07, we need to downgrade triton since 3.2.0 introduced a breaking change
#   This un-pinned requirement comes from mamba-ssm, and this pin can be removed once Pytorch base image is
#   updated.
RUN pip install triton==3.1.0

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

# TODO: This is redundant since NeMo installs this as of 24.12, but keep
#  it until 25.03 to give folks enough time to transition.
RUN pip install --no-cache-dir lightning

COPY --from=aligner-bump /opt/NeMo-Aligner /opt/NeMo-Aligner
RUN cd /opt/NeMo-Aligner && \
    pip install --no-deps -e .

RUN cd TensorRT-LLM && patch -p1 < ../NeMo-Aligner/setup/trtllm.patch

# NOTE: Comment this layer out if it is not needed
# NOTE: This section exists to allow cherry-picking PRs in cases where
#  we do not wish to simply update to the top-of-tree. Sometimes PRs
#  cannot be cherry-picked cleanly if rebased a few times to top-of-tree
#  so this logic also requires you to select a SHA (can be dangling) from
#  the PR.
RUN <<"EOF" bash -exu
cd NeMo
# Ensures we don't cherry-pick "future" origin/main commits
git fetch -a
# d27dd28b4186f6ecd9f46f1c5679a5eef9bad14e: fix: export weight name mapping if model is nemo model#11497
for pr_and_commit in \
  "11497 d27dd28b4186f6ecd9f46f1c5679a5eef9bad14e" \
; do
  pr=$(cut -f1 -d' ' <<<"$pr_and_commit")
  head_pr_commit=$(cut -f2 -d' ' <<<"$pr_and_commit")
  git fetch origin $head_pr_commit:PR-${pr}
  # cherry-picks all commits between main and the top of the PR
  git cherry-pick --allow-empty $(git merge-base origin/main PR-${pr})..PR-${pr}
  # Tag cherry-picks to help
  git tag cherry-pick-PR-${pr}
done
EOF
