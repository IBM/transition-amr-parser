from concurrent import futures
import logging

import grpc
import torch
import json
import amr_pb2
import amr_pb2_grpc

from transition_amr_parser.stack_transformer_amr_parser import AMRParser
from fairseq import options
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='AMR parser')
    parser.add_argument(
        "--in-model",
        help="path to the AMR parsing model",
        type=str
    )
    parser.add_argument(
        "--roberta-cache-path",
        help="Path to the roberta large model",
        type=str
    )
    parser.add_argument(
        "--port",
        help="GRPC port",
        type=str
    )
    parser.add_argument(
        '--roberta-batch-size',
        type=int,
        default=10,
        help='Batch size for roberta computation (watch for OOM)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for decoding (excluding roberta)'
    )
    args = parser.parse_args()

    # Sanity checks
    assert args.in_model
    assert args.port

    return args

class Parser():

    def __init__(self, args):
        self.parser = AMRParser.from_checkpoint(checkpoint=args.in_model, roberta_cache_path=args.roberta_cache_path)
        self.batch_size = args.batch_size
        self.roberta_batch_size = args.roberta_batch_size

    def process(self, request, context):
        sentences = request.sentences
        batch = []
        for sentence in sentences:
            batch.append(sentence.tokens)
        amrs = self.parser.parse_sentences(batch, batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size)
        return amr_pb2.AMRBatchResponse(amr_parse=amrs)

def serve():
    # Argument handling
    args = argument_parser()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    amr_pb2_grpc.add_AMRBatchServerServicer_to_server(Parser(args), server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()


