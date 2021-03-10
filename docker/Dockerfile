## install cuda 
FROM nvidia/cuda as ubuntu-pytorch
## some basic utilities
RUN apt-get -q -y update && DEBIAN_FRONTEND=noninteractive apt-get -q -y install curl vim locales lsb-release python3-pip ssh && apt-get clean
## add locale
RUN locale-gen en_US.UTF-8 && /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
## Install grpc for python3
RUN python3 -m pip install --upgrade pip && python3 -m pip install protobuf grpcio grpcio-tools && python3 -m pip install statsd

# Set up python
ADD setup.py /amr_parser/setup.py
WORKDIR /amr_parser
# Install the packages
RUN python3 -m pip install --editable .
RUN python3 -m spacy download en

# Set cache paths
ENV CACHE_DIR "cache"
ENV ROBERTA_CACHE_PATH ${CACHE_DIR}/roberta.large
# Download the roberta large model
RUN mkdir -p ${CACHE_DIR} && curl -L https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz | tar -xzv -C ${CACHE_DIR}

# Copy the code
ADD . /amr_parser

# Compile the protos
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/wordvec.proto
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/amr2.proto

# Path adjustemts, to be revised
RUN mkdir -p DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep120-seed42 && \
    ln -s /amr_parser/models/* DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep120-seed42

# Model Location
ENV MODEL_PATH "models/checkpoint_wiki.smatch_top5-avg.pt"
# GRPC Port (so that it can be set during run time)
ENV GRPC_PORT "50051"

# start the server
CMD python3 -u service/amr_server.py --in-model ${MODEL_PATH} --roberta-cache-path ${ROBERTA_CACHE_PATH} --port ${GRPC_PORT}
