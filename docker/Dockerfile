# Start with nvidia pytorch image (ensures working pytorch + CUDA)
FROM nvcr.io/nvidia/pytorch:19.04-py3

# Copy code
ADD . /amr_parser
WORKDIR /amr_parser

# Model Location
ENV MODEL_PATH "models/model.epoch40.params"

# GRPC Port (so that it can be set during run time)
ENV GRPC_PORT "50051"

# Install the packages
RUN pip install --editable .
RUN python -m spacy download en

# Install grpc stuff
RUN pip install grpcio
RUN pip install grpcio-tools

# Compile the protos
RUN python -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/wordvec.proto
RUN python -m grpc_tools.protoc -I./service/  --python_out=./service/ --grpc_python_out=./service/ ./service/amr.proto

# start the server
CMD python service/amr_server.py --in-model ${MODEL_PATH} --port ${GRPC_PORT}
