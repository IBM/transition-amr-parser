# Standalone AMR parser from an existing trained APT model

import os
import time
import math
import copy
import signal
import argparse
from datetime import timedelta

from ipdb import set_trace
from tqdm import tqdm
import torch
from fairseq import checkpoint_utils, utils
from fairseq.models.roberta import RobertaModel
from fairseq.tokenizer import tokenize_line

from fairseq_ext import options    # this is key to recognizing the customized arguments
from fairseq_ext.roberta.pretrained_embeddings import PretrainedEmbeddings
from fairseq_ext.data.amr_action_pointer_dataset import collate
# OR (same results) from fairseq_ext.data.amr_action_pointer_graphmp_dataset import collate
from fairseq_ext.utils import post_process_action_pointer_prediction, clean_pointer_arcs
from transition_amr_parser.amr_state_machine import AMRStateMachine, get_spacy_lemmatizer
from transition_amr_parser.amr import InvalidAMRError, get_duplicate_edges
from transition_amr_parser.utils import yellow_font
from transition_amr_parser.io import read_config_variables, read_tokenized_sentences


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Call parser from the command line'
    )
    parser.add_argument(
        '-i', '--in-tokenized-sentences',
        type=str,
        help='File with one __tokenized__ sentence per line'
    )
    parser.add_argument(
        '--service',
        action='store_true',
        help='Prompt user for sentences'
    )
    parser.add_argument(
        '-c', '--in-checkpoint',
        type=str,
        required=True,
        help='one fairseq model checkpoint (or various, separated by :)'
    )
    parser.add_argument(
        '-o', '--out-amr',
        type=str,
        help='File to store AMR in PENNMAN format'
    )
    parser.add_argument(
        '--roberta-batch-size',
        type=int,
        default=16,
        help='Batch size for roberta computation (watch for OOM)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for decoding (excluding roberta)'
    )
    # step by step parameters
    parser.add_argument(
        "--step-by-step",
        help="pause after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--set-trace",
        help="breakpoint after each action",
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    # sanity checks
    assert bool(args.in_tokenized_sentences) or bool(args.service), \
        "Must either specify --in-tokenized-sentences or set --service"

    return args


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def parse_sentences(parser, in_tokenized_sentences, batch_size, 
                    roberta_batch_size, out_amr):

    # read tokenized sentences
    sentences = read_sentences(in_tokenized_sentences)
    split_sentences = []
    for sentence in sentences:
        split_sentences.append(tokenize_line(sentence))
    print(len(split_sentences))
    
    # parse
    start = time.time()
    result, pred_dicts = parser.parse_sentences(
        split_sentences,
        batch_size=batch_size,
        roberta_batch_size=roberta_batch_size,
    )
    end = time.time()
    print(len(result))
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to parse sentences: {time_secs}')

        return results, predictions


def simple_inspector(machine):
    '''
    print the first machine
    '''
    os.system('clear')
    print(machine)
    input("")


def breakpoint_inspector(machine):
    '''
    call set_trace() on the first machine
    '''
    os.system('clear')
    print(machine)
    set_trace()


def main():

    # argument handling
    args = argument_parsing()

    # set inspector to use on action loop
    inspector = None
    if args.set_trace:
        inspector = breakpoint_inspector
    if args.step_by_step:
        inspector = simple_inspector

    # load parser
    start = time.time()
    parser = AMRParser.from_checkpoint(args.in_checkpoint, inspector=inspector)
    end = time.time()
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to load parser: {time_secs}')

    # TODO: max batch sizes could be computed from max sentence length
    if args.service:

        # set orderd exit
        signal.signal(signal.SIGINT, ordered_exit)
        signal.signal(signal.SIGTERM, ordered_exit)

        while True:
            sentence = input("Write sentence:\n")
            os.system('clear')
            if not sentence.strip():
                continue
            result = parser.parse_sentences(
                [sentence.split()],
                batch_size=args.batch_size,
                roberta_batch_size=args.roberta_batch_size,
            )
            #
            os.system('clear')
            print('\n')
            print(''.join(result[0]))

    else:

        # Parse sentences
        result = parser.parse_sentences(
            read_tokenized_sentences(args.in_tokenized_sentences),
            batch_size=args.batch_size, 
            roberta_batch_size=args.roberta_batch_size
        )

        with open(args.out_amr, 'w') as fid:
            fid.write(''.join(result[0]))


if __name__ == '__main__':
    main()
