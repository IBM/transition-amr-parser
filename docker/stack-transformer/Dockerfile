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
ENV MODEL_DIR "models/"

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
RUN python3 -m pip install torch==1.1.0 spacy==2.2.3 h5py cffi fastBPE regex sacrebleu requests 
# torch-scatter requires torch to be installed
RUN python3 -m pip install torch-scatter==1.3.2 
RUN python3 -m pip install --editable fairseq/.
RUN python3 -m pip install --editable .
RUN python3 -m spacy download en

# Compile the protos
RUN python3 -m grpc_tools.protoc -I./service/stack-transformer/  --python_out=./service/stack-transformer/ --grpc_python_out=./service/stack-transformer/ ./service/stack-transformer/amr.proto

# start the server
CMD python3 -u service/stack-transformer/amr_server.py ${MODEL_DIR} --source-lang en --target-lang actions --path ${MODEL_DIR}model.pt --machine-type AMR --model-overrides "{'pretrained_embed_dim':1024, 'task': 'translation'}" --machine-rules ${MODEL_DIR}train.rules.json --pretrained-embed roberta.large --bert-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --batch-size 64 --roberta-batch-size 10 --roberta-cache-path ${ROBERTA_CACHE_PATH} --port ${GRPC_PORT}