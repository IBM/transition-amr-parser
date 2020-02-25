from concurrent import futures
import logging

import grpc
import torch
import json
import amr_pb2
import amr_pb2_grpc

from transition_amr_parser.amr_parser import AMRParser
import transition_amr_parser.utils as utils
from transition_amr_parser.utils import print_log
from transition_amr_parser.learn import get_bert_embeddings

from fairseq.models.roberta import RobertaModel
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
    args = parser.parse_args()

    # Sanity checks
    assert args.in_model
    assert args.port

    return args

class Parser():

    def __init__(self, model_path, roberta_cache_path=None, roberta_use_gpu=False, model_use_gpu=False):
        if torch.cuda.is_available():
            roberta_use_gpu = True
            model_use_gpu = True
        self.parser = AMRParser(model_path, roberta_cache_path=roberta_cache_path, roberta_use_gpu=roberta_use_gpu, model_use_gpu=model_use_gpu)

    def process(self, request, context):
        word_tokens = request.word_infos
        tokens = [word_token.token for word_token in word_tokens]
        amr = self.parser.parse_sentence(tokens)
        return amr_pb2.AMRResponse(amr_parse=amr.toJAMRString())

def serve():
    # Argument handling
    args = argument_parser()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    amr_pb2_grpc.add_AMRServerServicer_to_server(Parser(model_path=args.in_model, roberta_cache_path=args.roberta_cache_path), server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()


