import os
import math
import torch
from tqdm import tqdm
import copy
import time
from datetime import timedelta

from fairseq import options, utils
from fairseq.tokenizer import tokenize_line

from transition_amr_parser.io import read_sentences
from transition_amr_parser.stack_transformer_amr_parser import AMRParser

def add_standalone_arguments(parser):
    parser.add_argument(
        "--roberta_batch_size",
        help="Batch size to compute roberta embeddings",
        default=20,
        type=int
    )
    parser.add_argument(
        "--roberta-cache-path",
        help="Path to the roberta large model",
        type=str
    )
    # for pretrained external embeddings
    parser.add_argument("--pretrained-embed", default='roberta.base',
                       help="Type of pretrained embedding")
    # NOTE: Previous default "17 18 19 20 21 22 23 24"
    parser.add_argument('--bert-layers', nargs='+', type=int,
                       help='RoBERTa layers to extract (default last)')
    

def main(args):

    # TODO: Consider getting rid of this and manually loading needed stuff if
    # the overhead is big. LEave otheriwse
    utils.import_user_module(args)

    sentences = read_sentences(args.input)
    split_sentences = []
    for sentence in sentences:
        split_sentences.append(tokenize_line(sentence))
    print(len(split_sentences))
    start = time.time()
    parser = AMRParser(args)
    end = time.time()
    print(f'Total time taken to load parser: {timedelta(seconds=float(end-start))}')
    start = time.time()
    result = parser.parse_sentences(split_sentences)
    end = time.time()
    print(len(result))
    print(f'Total time taken to parse sentences: {timedelta(seconds=float(end-start))}')

    # write annotations
    with open('tmp.amr', 'w') as fid:
        for i in range(0,len(sentences)):
            fid.write(result[i])
    
# TODO: Get rid of options parser from fairseq and task loading if it
# represents a big overhead
def cli_main():
    parser = options.get_interactive_generation_parser(default_task='parsing')
    options.add_optimization_args(parser)
    add_standalone_arguments(parser)
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()
