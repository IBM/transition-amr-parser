from concurrent import futures
import logging


import torch
import json

from transition_amr_parser.parse import AMRParser,get_sliding_output



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
        default=512,
        help='Batch size for roberta computation (watch for OOM)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for decoding (excluding roberta)'
    )
    parser.add_argument(
        "--debug",
        help="Sentence for local debugging",
        type=str
    )
       
    parser.add_argument(
        "--window-size",
        help="size of sliding window",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--window-overlap",
        help="size of overlap between sliding windows",
	default=200,
	type=int,
    )  
    parser.add_argument(
        "--force-actions",
        help="action sequence (per token) for force decoding",
        type=str,
        default=None
	
    )
    parser.add_argument(
        "--doc-mode",
        help="perform parsing in doc-mode",
        action='store_true'
	
    )  
    parser.add_argument(
        "--beam",
        help="beam",
        type=int,
        default=1
	
    )  
    args = parser.parse_args()

    # Sanity checks
    if not args.debug:
        assert args.port, "Must provide --port"

    return args

class Parser():
    def __init__(self, args):
        self.parser = AMRParser.from_checkpoint(checkpoint=args.in_model,beam=args.beam)
        self.batch_size = args.batch_size
        self.roberta_batch_size = args.roberta_batch_size
        self.window_size = args.window_size
        self.window_overlap = args.window_overlap
        self.beam = args.beam

    def process(self, request, context):
        sentences = request.sentences
        doc_mode = request.doc_mode
        batch = []
        for sentence in sentences:
            batch.append(sentence.tokens)
        if doc_mode:
            amrs = get_sliding_output(batch,window_size=self.window_size,window_overlap=self.window_overlap,parser=self.parser,gold_amrs=None,batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size)
        else:
            amrs = self.parser.parse_sentences(batch, batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size)[0]
        return amr2_pb2.AMRBatchResponse(amr_parse=amrs)

    def debug_process(self, tokens,force_actions=None):
        amr = self.parser.parse_sentences([tokens], batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size,force_actions=force_actions)[0][0]
        return amr
    def debug_process_doc(self, tokens,force_actions=None):
        
        force_actions = eval(force_actions.strip())+[[]]
        
        assert len(tokens)==len(force_actions)-1
        amr = get_sliding_output([tokens],window_size=self.window_size,window_overlap=self.window_overlap,parser=self.parser,gold_amrs=None,batch_size=self.batch_size, roberta_batch_size=self.roberta_batch_size,force_actions=[force_actions],beam=self.beam)
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
    if args.debug and args.doc_mode:
        model = Parser(args)
        out = model.debug_process_doc(args.debug.split(),args.force_actions)
        with open('service/debug.out','w') as f:
            f.write(out[0])
    elif args.debug:
        model = Parser(args)
        print(model.debug_process(args.debug.split()),args.force_actions)
    else:
        import grpc
        import amr2_pb2
        import amr2_pb2_grpc
        serve(args)
