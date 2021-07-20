import argparse
from transition_amr_parser.io import read_amr2
from align_cfg.vocab_definitions import (
    PADDING_IDX, PADDING_TOK, BOS_IDX, BOS_TOK, EOS_IDX, EOS_TOK, special_tokens
)


if __name__ == '__main__':

    # Argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-amrs", help="AMR files to extract the vocabulary from",
        nargs='+', required=True)
    parser.add_argument(
        "--out-folder", help="Folder where to store vocabulary files",
        required=True)
    args = parser.parse_args()

    # collect infor for all AMR
    tokens = set()
    graph_tokens = set()
    for amr_file in args.in_amrs:
        for amr in read_amr2(amr_file, ibm_format=True):
            # surface tokens
            tokens.update(amr.tokens)
            # graph tokens
            for _, label, _ in amr.edges:
                graph_tokens.add(label)
            graph_tokens.update(amr.nodes.values())

    # Add special symbols at the beginning
    # surface
    tokens = special_tokens + sorted(tokens)
    # graph
    # useful for linearized parse
    graph_tokens.add('(')
    graph_tokens.add(')')
    graph_tokens = special_tokens + sorted(graph_tokens)

    # write files
    print('found {} text tokens'.format(len(tokens)))
    with open(f'{args.out_folder}/ELMO_vocab.text.txt', 'w') as f:
        for tok in tokens:
            f.write(tok + '\n')
    print('found {} amr tokens'.format(len(graph_tokens)))
    with open(f'{args.out_folder}/ELMO_vocab.amr.txt', 'w') as f:
        for tok in graph_tokens:
            f.write(tok + '\n')
