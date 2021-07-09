import os

from transition_amr_parser.io import read_amr


PADDING_IDX = 0
PADDING_TOK = '<PAD>'

BOS_IDX = 1
BOS_TOK = '<S>'

EOS_IDX = 2
EOS_TOK = '</S>'

special_tokens = [PADDING_TOK, BOS_TOK, EOS_TOK]

assert special_tokens.index(PADDING_TOK) == PADDING_IDX
assert special_tokens.index(BOS_TOK) == BOS_IDX
assert special_tokens.index(EOS_TOK) == EOS_IDX


def read_text_tokens_from_amr(files):
    tokens = set()

    for path in files:
        path = os.path.expanduser(path)
        for amr in read_amr(path).amrs:
            tokens.update(amr.tokens)

    tokens = special_tokens + sorted(tokens)

    return tokens


def read_amr_tokens_from_amr(files):
    tokens = set()

    for path in files:
        path = os.path.expanduser(path)
        for amr in read_amr(path).amrs:
            for _, label, _ in amr.edges:
                tokens.add(label)
            tokens.update(amr.nodes.values())

    # useful for linearized parse
    tokens.add('(')
    tokens.add(')')

    tokens = special_tokens + sorted(tokens)

    return tokens


if __name__ == '__main__':
    import argparse

    input_files = []

    # AMR 2.0
    input_files.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.txt'))
    input_files.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.txt'))
    input_files.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/test.txt'))

    # AMR 3.0
    input_files.append(os.path.expanduser('~/data/AMR3.0/train.txt'))
    input_files.append(os.path.expanduser('~/data/AMR3.0/dev.txt'))
    input_files.append(os.path.expanduser('~/data/AMR3.0/test.txt'))

    # MANY
    input_files.append(os.path.expanduser('/dccstor/ykt-parse/SHARED/misc/adrozdov/data/amr2+ontonotes+squad.txt'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-amr", help="AMR input file.", action="append", default=input_files)
    args = parser.parse_args()

    input_files = args.in_amr

    tokens = read_text_tokens_from_amr(input_files)
    print('found {} text tokens'.format(len(tokens)))

    with open('align_cfg/vocab.text.2021-06-30.txt', 'w') as f:
        for tok in tokens:
            f.write(tok + '\n')

    tokens = read_amr_tokens_from_amr(input_files)
    print('found {} amr tokens'.format(len(tokens)))

    with open('align_cfg/vocab.amr.2021-06-30.txt', 'w') as f:
        for tok in tokens:
            f.write(tok + '\n')

