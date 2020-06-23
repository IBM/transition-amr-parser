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

def add_server_arguments(parser):
    parser.add_argument(
        "--roberta-cache-path",
        help="Path to the roberta large model",
        type=str
    )
    parser.add_argument(
        "--roberta-batch-size",
        help="Batch size to compute roberta embeddings",
        type=int
    )
    parser.add_argument(
        "--port",
        help="GRPC port",
        type=str
    )
    # for pretrained external embeddings
    parser.add_argument("--pretrained-embed", default='roberta.base',
                       help="Type of pretrained embedding")
    # NOTE: Previous default "17 18 19 20 21 22 23 24"
    parser.add_argument('--bert-layers', nargs='+', type=int,
                       help='RoBERTa layers to extract (default last)')

class Parser():

    def __init__(self, args):
        self.parser = AMRParser(args)

    def process(self, request, context):
        sentences = request.sentences
        batch = []
        for sentence in sentences:
            batch.append(sentence.tokens)
        amrs = self.parser.parse_sentences(batch)
        return amr_pb2.AMRBatchResponse(amr_parse=amrs)

def serve():
    # Argument handling
    parser = options.get_interactive_generation_parser(default_task='parsing')
    options.add_optimization_args(parser)
    add_server_arguments(parser)
    args = options.parse_args_and_arch(parser)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    amr_pb2_grpc.add_AMRBatchServerServicer_to_server(Parser(args), server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()


