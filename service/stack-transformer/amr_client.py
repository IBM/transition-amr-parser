import logging

import grpc
import torch
import json
import amr_pb2
import amr_pb2_grpc
import argparse
from transition_amr_parser.io import read_sentences

def argument_parser():
    parser = argparse.ArgumentParser(description='AMR parser')
    parser.add_argument(
        "--port",
        help="GRPC port",
        type=str
    )
    args = parser.parse_args()

    # Sanity checks
    assert args.port

    return args

def get_input_from_sentences(sentences):
    batch = []
    for sentence in sentences:
        batch.append(amr_pb2.AMRBatchInput.Sentence(tokens=sentence.split()))
    return amr_pb2.AMRBatchInput(sentences=batch)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # Argument handling
    args = argument_parser()
    channel = grpc.insecure_channel('localhost:' + args.port)
    stub = amr_pb2_grpc.AMRBatchServerStub(channel)
    sentences = read_sentences("./data/qald_dev/dev.en")
    amr_input = get_input_from_sentences(sentences)
    response = stub.process(amr_input)
    with open('tmp.amr', 'w') as fid:
        for parse in response.amr_parse:
            fid.write(parse)

if __name__ == '__main__':
    logging.basicConfig()
    run()