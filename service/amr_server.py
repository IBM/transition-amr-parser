from concurrent import futures
import logging
import torch

from transition_amr_parser.parse import AMRParser

import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='AMR parser')
    parser.add_argument(
        "--in-model",
        help="path to the AMR parsing model",
        type=str,
        required=True
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
        "--max-workers",
        help="Maximum nonumber of threads",
        default=10,
        type=int
    )
    parser.add_argument(
        "--debug",
        help="Sentence for local debugging",
        type=str
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.debug:
        assert args.port, "Must provide --port"

    return args

class Parser():

    def __init__(self, model_path, roberta_cache_path=None):
        self.parser = AMRParser.from_checkpoint(
            model_path,
            roberta_cache_path=roberta_cache_path
        )

    def process(self, request, context):
        word_tokens = request.word_infos
        tokens = [word_token.token for word_token in word_tokens]
        amr = self.parser.parse_sentences(tokens)[0][0]
        return amr_pb2.AMRResponse(amr_parse=amr)

    def debug_process(self, tokens):
        amr = self.parser.parse_sentences([tokens])[0][0]
        return amr


def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    model = Parser(model_path=args.in_model, roberta_cache_path=args.roberta_cache_path)
    amr_pb2_grpc.add_AMRServerServicer_to_server(model, server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':

    # Argument handling
    args = argument_parser()

    logging.basicConfig()
    if args.debug:
        model = Parser(
            model_path=args.in_model, roberta_cache_path=args.roberta_cache_path)
        print(model.debug_process(args.debug.split()))
    else:
        import grpc
        import amr_pb2
        import amr_pb2_grpc
        serve(args)
