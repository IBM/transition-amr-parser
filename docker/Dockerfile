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

RUN useradd -u 1001 -m worker
WORKDIR /home/worker/amr-parser
RUN chown -R worker:worker /home/worker
ENV AMR_PARSER "/home/worker/amr-parser"

# Set up python
ADD --chown=worker:worker setup.py ./
# Install the packages
RUN python3 -m pip install --upgrade pip && python3 -m pip install --editable .
RUN python3 -m spacy download en
USER worker

# Set cache paths
ENV CACHE_DIR "cache"
ENV ROBERTA_CACHE_PATH ${CACHE_DIR}/roberta.large
# Download the roberta large model
RUN mkdir -p ${CACHE_DIR} && curl -L https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz | tar -xzv -C ${CACHE_DIR}

# Copy the code
ADD --chown=worker:worker . ./

# Compile the protos
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/wordvec.proto
RUN python3 -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/amr2.proto

# Temporary fix
ENV MODEL_DIR "EXP2/exp_amr3questions_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed43"
# Path adjustemts, to be revised
RUN mkdir -p ${MODEL_DIR} && ln -s ${AMR_PARSER}/models/* ${MODEL_DIR} 

# Model Location
ENV MODEL_PATH "models/checkpoint_smatch_top3-avg.pt"
# GRPC Port (so that it can be set during run time)
ENV GRPC_PORT "50051"

# start the server
CMD python3 -u service/amr_server.py --in-model ${MODEL_PATH} --roberta-cache-path ${ROBERTA_CACHE_PATH} --port ${GRPC_PORT}
