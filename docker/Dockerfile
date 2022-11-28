FROM registry.access.redhat.com/ubi8/ubi:latest as rhel-base

RUN dnf update -y \
    && dnf install -y --disableplugin=subscription-manager \
    python38 \
    python38-setuptools \
    python38-wheel \
    python38-pip \
    python38-devel \
    make \
    git \
    glibc-langpack-en \
    curl \
    gcc \
    gcc-c++ \
    unzip \
    && dnf autoremove -y \
    && dnf clean all \
    && pip3 install --upgrade --no-cache-dir pip

ADD https://github.com/ibmruntimes/semeru8-binaries/releases/download/jdk8u302-b08_openj9-0.27.0/ibm-semeru-open-8-jdk-1.8.0.302.b08-1.x86_64.rpm .
RUN dnf install -y ibm-semeru-open-8-jdk-1.8.0.302.b08-1.x86_64.rpm

#RUN locale-gen en_US.UTF-8 && /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
# Model Location

ENV MODEL_PATH "DATA/gofa20220412/models/exp_cofill_o10_act-states_bart.base/_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt1024x8-wm4000-dp0.2/ep15-seed44/checkpoint_top3-avg.pt"

# GRPC Port (so that it can be set during run time)
ENV GRPC_PORT "50051"

# Set cache paths
ENV CACHE_DIR "DATA"
ENV ROBERTA_CACHE_PATH ${CACHE_DIR}/bart.base
ENV PYTHONPATH /amr_parser/pip_modules
## Install grpc for python3

FROM rhel-base as rhel-stage1
ARG ARTIFACTORY_USERNAME
ARG ARTIFACTORY_API_KEY
ENV ARTIFACTORY_USERNAME=$ARTIFACTORY_USERNAME
ENV ARTIFACTORY_API_KEY=$ARTIFACTORY_API_KEY

# ADD . /amr_parser/
WORKDIR /amr_parser

COPY LICENSE README.md setup.py requirements.txt /amr_parser/
COPY DATA/ /amr_parser/DATA
COPY preprocess/ /amr_parser/preprocess
COPY scripts/ /amr_parser/scripts/
COPY tests/ /amr_parser/tests
COPY configs/ /amr_parser/configs/
COPY fairseq_ext/ /amr_parser/fairseq_ext/
COPY service/ /amr_parser/service
COPY transition_amr_parser/ /amr_parser/transition_amr_parser/

RUN python3 -m pip install -t ${PYTHONPATH} --upgrade pip \
    && python3 -m pip install -t ${PYTHONPATH} protobuf grpcio grpcio-tools grpcio-health-checking \
    && python3 -m pip install -t ${PYTHONPATH} statsd

# Copy code
# ADD . /amr_parser

#RUN scripts/update_config.sh \
#    --artifactory_username $ARTIFACTORY_USERNAME \
#    --artifactory_api_key $ARTIFACTORY_API_KEY \
#    --encode_username
#RUN pip install -t ${PYTHONPATH} 'pyizumo[dp]'==0.1.5 watson-sire==1.0.18 requests

ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 7.5+PTX"
# Install the packages
RUN python3 -m pip install -t ${PYTHONPATH} .
#RUN pip3 install  -t ${PYTHONPATH} -r requirements.txt
RUN rm -rf ${PYTHONPATH}/dataclasses*
RUN python3 -m pip install -t ${PYTHONPATH} torch-scatter==1.3.2

# Compile the protos
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/wordvec.proto
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/amr2.proto
# RUN chown worker:worker /amr_parser
RUN rm -rf *.zip
RUN chmod -R 777 /amr_parser

FROM golang:1.17 AS grpcurl_build
RUN go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

FROM rhel-base as amr-final
COPY --from=rhel-stage1 /amr_parser/ /amr_parser/
COPY --from=grpcurl_build /go/bin/grpcurl /usr/local/bin/grpcurl
# start the server
ENV PYTHONPATH "/amr_parser:/amr_parser/server:/amr_parser/pip_modules"

WORKDIR /amr_parser
RUN ls -l /amr_parser/DATA
RUN ls -l /amr_parser/DATA/bart.base
#RUN mkdir -p /.cache && chmod -R 777 /.cache
#RUN python3 service/amr_test.py -m ${MODEL_PATH} -c ${ROBERTA_CACHE_PATH}
CMD python3 -u service/amr_server.py --in-model ${MODEL_PATH} --roberta-cache-path ${ROBERTA_CACHE_PATH} --port ${GRPC_PORT}
