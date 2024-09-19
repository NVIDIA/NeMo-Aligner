ARG BASE_IMAGE=nvcr.io/nvidia/nemo:24.07

FROM ${BASE_IMAGE}
ARG ALIGNER_COMMIT=main

WORKDIR /opt

# NeMo Aligner
RUN <<"EOF" bash -exu
cd NeMo-Aligner
git fetch -a
git checkout -f ${ALIGNER_COMMIT}
cd -
EOF

ENV NEMO_ALIGNER_BUILD=true
