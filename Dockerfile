# CUDA 12.2
FROM nvcr.io/nvidia/pytorch:23.10-py3

# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8
# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"

WORKDIR /opt

RUN git clone -b 23.08 https://github.com/NVIDIA/Megatron-LM.git
RUN cd Megatron-LM && git cherry-pick 28363ee2af1d7384a402a84a9e15a03271b59db7 && pip install -e .

RUN git clone -b r1.21.0 https://github.com/NVIDIA/NeMo.git
RUN cd NeMo && git cherry-pick -X theirs 05ecfe40a74ea495e3bcd1d7c38307399701aa04 25c31382078e8328cd2e6ea0cc20ecbb7d702a07 8303d0d482a15b281a474fc6ea083dab6e59d645 06b13f912f05e395546bf728adea65ccaa464080 88c872aa7e633ae37849f971be76f889b5ac6069 54ce830c9d9f5bf2a0996f5ad0a9c1cccf1a0d39
WORKDIR /opt/NeMo
# fixes a bug in Nemo when used with latest apex
RUN sed -i 's/_fast_layer_norm(x, self.weight + 1, self.bias, self.epsilon)/_fast_layer_norm(x, self.weight + 1, self.bias, self.epsilon, False)/g' nemo/collections/nlp/modules/common/megatron/layer_norm_1p.py
RUN sed -i 's/shutil.rmtree(ckpt_to_dir(filepath))/shutil.rmtree(ckpt_to_dir(filepath), ignore_errors=True)/g' nemo/collections/nlp/parts/nlp_overrides.py
RUN sed -i 's/\[all\]/\[nlp\]/g' reinstall.sh
RUN rm -rf .git && ./reinstall.sh

WORKDIR /opt

RUN pip install --upgrade-strategy only-if-needed jsonlines

RUN pip uninstall -y transformer-engine
RUN pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.1
# alternatively, if you have issues with TE 1.1, try stable instead
#RUN pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable

RUN pip uninstall -y apex

RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--permutation_search" --config-settings "--build-option=--bnp" --config-settings "--build-option=--xentropy" --config-settings "--build-option=--focal_loss" --config-settings "--build-option=--index_mul_2d" --config-settings "--build-option=--deprecated_fused_adam" --config-settings "--build-option=--fast_layer_norm" --config-settings "--build-option=--fmha" --config-settings "--build-option=--fast_multihead_attn" --config-settings "--build-option=--transducer" --config-settings "--build-option=--cudnn_gbn" --config-settings "--build-option=--fused_conv_bias_relu" .

RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==0.4.1
RUN pip install -U --no-deps protobuf==4.24.4

WORKDIR /workspace

ENV NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
ENV NVTE_MASKED_SOFTMAX_FUSION=0
