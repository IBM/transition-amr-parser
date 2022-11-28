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
        help="Maximum nonumber of threads -- unsafe to chage from 1",
        default=1,
        type=int
    )
    parser.add_argument(
        '--roberta-batch-size',
        type=int,
        default=1,
        help='Batch size for roberta computation (watch for OOM)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for decoding (excluding roberta)'
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
    def __init__(self, args):
        self.parser = AMRParser.from_checkpoint(checkpoint=args.in_model, roberta_cache_path=args.roberta_cache_path)
        self.batch_size = args.batch_size
        self.roberta_batch_size = args.roberta_batch_size

    def process(self, request, context):
        sentences = request.sentences
        batch = []
        for sentence in sentences:
            batch.append(sentence.tokens)
        amrs = self.parser.parse_sentences(batch, batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size)[0]
        return amr2_pb2.AMRBatchResponse(amr_parse=amrs)

    def debug_process(self, tokens):
        amr = self.parser.parse_sentences([tokens], batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size)[0][0]
        return amr

def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    amr2_pb2_grpc.add_AMRBatchServerServicer_to_server(Parser(args), server)
    laddr = '[::]:' + args.port
    server.add_insecure_port(laddr)
    server.start()
    print("server listening on ", laddr)
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
        import amr2_pb2
        import amr2_pb2_grpc
        serve(args)
