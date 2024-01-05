# CUDA 12.2
FROM nvcr.io/nvidia/pytorch:23.10-py3

# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8
# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"

WORKDIR /opt

# install TransformerEngine
RUN pip uninstall -y transformer-engine
RUN pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.1

# install latest apex
RUN pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--permutation_search" --config-settings "--build-option=--bnp" --config-settings "--build-option=--xentropy" --config-settings "--build-option=--focal_loss" --config-settings "--build-option=--index_mul_2d" --config-settings "--build-option=--deprecated_fused_adam" --config-settings "--build-option=--fast_layer_norm" --config-settings "--build-option=--fmha" --config-settings "--build-option=--fast_multihead_attn" --config-settings "--build-option=--transducer" --config-settings "--build-option=--cudnn_gbn" --config-settings "--build-option=--fused_conv_bias_relu" .

# place any util pkgs here
RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==0.4.1
RUN pip install -U --no-deps protobuf==4.24.4
RUN pip install --upgrade-strategy only-if-needed jsonlines

# NeMo
RUN git clone -b r1.22.0 https://github.com/NVIDIA/NeMo.git
RUN cd NeMo && git cherry-pick -X theirs fa8d416793d850f4ce56bea65e1fe28cc0d092c0 a7f0bc1903493888c31436efc2452ff721fa5a67
WORKDIR /opt/NeMo
# waiting on https://github.com/NVIDIA/NeMo/pull/8130/files to get merged before deleting the following WAR
RUN sed -i 's/shutil.rmtree(ckpt_to_dir(filepath))/shutil.rmtree(ckpt_to_dir(filepath), ignore_errors=True)/g' nemo/collections/nlp/parts/nlp_overrides.py
RUN sed -i 's/\[all\]/\[nlp\]/g' reinstall.sh
RUN rm -rf .git && ./reinstall.sh

# MLM
RUN pip uninstall -y megatron-core
RUN git clone -b core_r0.4.0 https://github.com/NVIDIA/Megatron-LM.git
RUN cd Megatron-LM && pip install -e .

WORKDIR /opt

# install the latest NeMo-Aligner
RUN pip install --no-deps git+https://github.com/NVIDIA/NeMo-Aligner.git@main

WORKDIR /workspace
