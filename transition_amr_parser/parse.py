import time
import os
import signal
import argparse
from datetime import timedelta
from fairseq.tokenizer import tokenize_line
from transition_amr_parser.io import read_sentences
from transition_amr_parser.stack_transformer_amr_parser import AMRParser


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
        default=10,
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
    return parser.parse_args()


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
    result = parser.parse_sentences(
        split_sentences,
        batch_size=batch_size,
        roberta_batch_size=roberta_batch_size,
    )
    end = time.time()
    print(len(result))
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to parse sentences: {time_secs}')
    
    # write annotations
    if out_amr:
        with open(out_amr, 'w') as fid:
            for i in range(0, len(sentences)):
                fid.write(result[i])


def simple_inspector(machine):
    '''
    print the first machine
    '''
    os.system('clear')
    print(machine.machines[0])
    input("")


def main():

    # argument handling
    args = argument_parsing()

    inspector = None
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
            print(result[0])

    else:

        # Parse sentences
        parse_sentences(parser, args.in_tokenized_sentences, args.batch_size, 
                        args.roberta_batch_size, args.out_amr)
