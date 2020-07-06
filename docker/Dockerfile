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

# Copy code
ADD . /amr_parser
WORKDIR /amr_parser

# Model Location
ENV MODEL_PATH "models/model.epoch25.params"

# GRPC Port (so that it can be set during run time)
ENV GRPC_PORT "50051"

# Set cache paths
ENV CACHE_DIR "cache/"
ENV ROBERTA_CACHE_PATH ${CACHE_DIR}/roberta.large

# Download the roberta large model
RUN wget -P ${CACHE_DIR} https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
RUN tar -xzvf ${CACHE_DIR}/roberta.large.tar.gz -C ${CACHE_DIR}
RUN rm ${CACHE_DIR}/roberta.large.tar.gz

# Install the packages
RUN python3 -m pip install --editable .
RUN python3 -m spacy download en

# Compile the protos
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/wordvec.proto
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/amr.proto

# start the server
CMD python3 -u service/amr_server.py --in-model ${MODEL_PATH} --roberta-cache-path ${ROBERTA_CACHE_PATH} --port ${GRPC_PORT}
