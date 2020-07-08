import time
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
        required=True,
        help='File with one __tokenized__ sentence per line'
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
        required=True,
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
    return parser.parse_args()


def main():

    # argument handling
    args = argument_parsing()

    # read tokenized sentences
    sentences = read_sentences(args.in_tokenized_sentences)
    split_sentences = []
    for sentence in sentences:
        split_sentences.append(tokenize_line(sentence))
    print(len(split_sentences))

    # load parser
    start = time.time()
    parser = AMRParser.from_checkpoint(args.in_checkpoint)
    end = time.time()
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to load parser: {time_secs}')

    # TODO: max batch sizes could be computed from max sentence length

    # parse
    start = time.time()
    result = parser.parse_sentences(
        split_sentences,
        batch_size=args.batch_size,
        roberta_batch_size=args.roberta_batch_size,
    )
    end = time.time()
    print(len(result))
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to parse sentences: {time_secs}')

    # write annotations
    with open(args.out_amr, 'w') as fid:
        for i in range(0, len(sentences)):
            fid.write(result[i])
