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


# def add_standalone_arguments(parser):
#     parser.add_argument(
#         "--roberta_batch_size",
#         help="Batch size to compute roberta embeddings",
#         default=20,
#         type=int
#     )
#     parser.add_argument(
#         "--roberta-cache-path",
#         help="Path to the roberta large model",
#         type=str
#     )
#     parser.add_argument(
#         "--out-amr",
#         help="Path to the file where AMR will be tored",
#         type=str
#     )
#     # for pretrained external embeddings
#     parser.add_argument("--pretrained-embed", default='roberta.base',
#                         help="Type of pretrained embedding")
#     # NOTE: Previous default "17 18 19 20 21 22 23 24"
#     parser.add_argument('--bert-layers', nargs='+', type=int,
#                         help='RoBERTa layers to extract (default last)')


def main():

    args = argument_parsing()

    # TODO: Consider getting rid of this and manually loading needed stuff if
    # the overhead is big. LEave otheriwse
    # utils.import_user_module(args)

    # read tokenized sentences
    sentences = read_sentences(args.in_tokenized_sentences)
    split_sentences = []
    for sentence in sentences:
        split_sentences.append(tokenize_line(sentence))
    print(len(split_sentences))

    # Load parser
    start = time.time()
    parser = AMRParser.from_checkpoint(args.in_checkpoint)
    end = time.time()
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to load parser: {time_secs}')

    # Parse
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


# # TODO: Get rid of options parser from fairseq and task loading if it
# # represents a big overhead
# def cli_main():
#     parser = options.get_interactive_generation_parser()
#     options.add_optimization_args(parser)
#     add_standalone_arguments(parser)
#     args = options.parse_args_and_arch(parser)
#     main(args)


if __name__ == '__main__':
    main()
