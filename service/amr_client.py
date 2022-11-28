import logging

import grpc
import torch
import json
import amr_pb2
import amr_pb2_grpc
import argparse

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

def get_input_from_sentence(sentence,mode):
    tokens = sentence.split()
    input_tokens = []
    for token in tokens:
        input_tokens.append(amr_pb2.AMRInput.WordInfo(token=token))
    
    if mode.lower()=='doc' or mode.lower()=='document':
        doc_mode = True
    else:
        doc_mode = False
    return amr_pb2.AMRInput(word_infos=input_tokens,doc_mode=doc_mode)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # Argument handling
    args = argument_parser()
    channel = grpc.insecure_channel('localhost:' + args.port)
    stub = amr_pb2_grpc.AMRServerStub(channel)
    sentence = input("Enter the sentence: ")
    mode = input("Enter the mode: ")
    amr_input = get_input_from_sentence(sentence,mode)
    response = stub.process(amr_input)
    print("AMR parse received: \n" + response.amr_parse)

if __name__ == '__main__':
    logging.basicConfig()
    run()