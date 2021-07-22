import json

from align_cfg.vocab_definitions import (
    PADDING_IDX, PADDING_TOK, BOS_IDX, BOS_TOK, EOS_IDX, EOS_TOK, special_tokens
)
from transition_amr_parser.io import read_amr2


def main(args):

    summary = {}

    # collect information for all AMR
    tokens = set()
    graph_tokens = set()
    for amr_file in args.in_amrs:

        print('reading {}\n'.format(amr_file))

        try:
            # JAMR
            local_tokens = set()
            local_graph_tokens = set()

            for amr in read_amr2(amr_file, ibm_format=False):
                # surface tokens
                local_tokens.update(amr.tokens)
                # graph tokens
                for _, label, _ in amr.edges:
                    graph_tokens.add(label)
                local_graph_tokens.update(amr.nodes.values())

            print(f'found {len(tokens)} tokens and {len(graph_tokens)} graph tokens w/ JAMR\n')

            # Update
            tokens = set.union(tokens, local_tokens)
            graph_tokens = set.union(graph_tokens, local_graph_tokens)

            print(f'current {len(tokens)} tokens and {len(graph_tokens)} graph tokens in vocab\n')

            summary[(amr_file, 'jamr')] = f'found {len(tokens)} tokens and {len(graph_tokens)} graph tokens w/ JAMR'
        except:
            print('could not read as JAMR\n')

            summary[(amr_file, 'jamr')] = 'failed'


        try:
            # PENMAN
            local_tokens = set()
            local_graph_tokens = set()

            for amr in read_amr2(amr_file, ibm_format=False, tokenize=True):
                # surface tokens
                local_tokens.update(amr.tokens)
                # graph tokens
                for _, label, _ in amr.edges:
                    graph_tokens.add(label)
                local_graph_tokens.update(amr.nodes.values())

            print(f'found {len(tokens)} tokens and {len(graph_tokens)} graph tokens w/ PENMAN\n')

            # Update
            tokens = set.union(tokens, local_tokens)
            graph_tokens = set.union(graph_tokens, local_graph_tokens)

            print(f'current {len(tokens)} tokens and {len(graph_tokens)} graph tokens in vocab\n')

            summary[(amr_file, 'penman')] = f'found {len(tokens)} tokens and {len(graph_tokens)} graph tokens w/ PENMAN'
        except:
            print('could not read as JAMR\n')

            summary[(amr_file, 'penman')] = 'failed'

    for tok in special_tokens:
        if tok in tokens:
            tokens.remove(tok)
        if tok in graph_tokens:
            graph_tokens.remove(tok)

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

    # print summary
    print('summary\n-------')

    for k, v in summary.items():
        print(k)
        print(v)
        print('')


if __name__ == '__main__':
    import argparse

    # Argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-amrs", help="AMR files to extract the vocabulary from",
        nargs='+', required=True)
    parser.add_argument(
        "--out-folder", help="Folder where to store vocabulary files",
        required=True)
    args = parser.parse_args()

    print(json.dumps(args.__dict__))

    main(args)
