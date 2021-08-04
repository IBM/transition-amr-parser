"""

# Create oracles

python transition_amr_parser/amr_machine.py --use-copy 1 \
    --in-aligned-amr DATA/AMR2.0/aligned/align_cfg/train.txt \
    --out-machine-config DATA/AMR2.0/aligned/align_cfg/machine_config.json \
    --out-actions DATA/AMR2.0/aligned/align_cfg/train.actions \
    --out-tokens DATA/AMR2.0/aligned/align_cfg/train.tokens \
    --absolute-stack-positions

#

python transition_amr_parser/amr_machine.py --use-copy 1 \
    --in-aligned-amr ~/data/AMR2.0/aligned/cofill/train.txt \
    --out-machine-config ~/data/AMR2.0/aligned/cofill/machine_config.json \
    --out-actions ~/data/AMR2.0/aligned/cofill/train.actions \
    --out-tokens ~/data/AMR2.0/aligned/cofill/train.tokens \
    --absolute-stack-positions

####

python transition_amr_parser/amr_machine.py --use-copy 1 \
    --in-aligned-amr DATA/AMR2.0/aligned/align_cfg/dev.txt \
    --out-machine-config DATA/AMR2.0/aligned/align_cfg/machine_config.json \
    --out-actions DATA/AMR2.0/aligned/align_cfg/dev.actions \
    --out-tokens DATA/AMR2.0/aligned/align_cfg/dev.tokens \
    --absolute-stack-positions

python transition_amr_parser/amr_machine.py \
    --in-machine-config DATA/AMR2.0/aligned/align_cfg/machine_config.json \
    --in-tokens DATA/AMR2.0/aligned/align_cfg/dev.tokens \
    --in-actions DATA/AMR2.0/aligned/align_cfg/dev.actions  \
    --out-amr DATA/AMR2.0/aligned/align_cfg/dev_oracle.amr

#

python transition_amr_parser/amr_machine.py --use-copy 1 \
    --in-aligned-amr ~/data/AMR2.0/aligned/cofill/dev.txt \
    --out-machine-config ~/data/AMR2.0/aligned/cofill/machine_config.json \
    --out-actions ~/data/AMR2.0/aligned/cofill/dev.actions \
    --out-tokens ~/data/AMR2.0/aligned/cofill/dev.tokens \
    --absolute-stack-positions

"""


import argparse
import collections
import json
import os

import numpy as np

from transition_amr_parser.io import read_amr2


def read_actions(path):
    data = []
    with open(path) as f:
        for line in f:
            actions = line.strip().split()
            data.append(actions)
    return data


class Machine:
    def __init__(self, corpus, actions):
        new_corpus = []

        for amr, seq in zip(corpus, actions):
            tokens = amr.tokens

            new_seq = []

            cursor = 0

            for a in seq:
                if a == 'SHIFT':
                    cursor += 1
                elif a == 'COPY':
                    new_seq.append(('COPY', amr.tokens[cursor]))

            new_corpus.append(new_seq)
        self.new_corpus = new_corpus


def validate(amr_corpus, ref_actions, amr2_corpus, new_actions):
    amr_corpus_, ref_actions_, amr2_corpus_, new_actions_ = [], [], [], []

    stats = collections.Counter()

    for amr1, seq1, amr2, seq2 in zip(amr_corpus, ref_actions, amr2_corpus, new_actions):
        try:
            assert len(amr1.tokens) == len(amr2.tokens), (len(amr1.tokens), len(amr2.tokens), amr1.tokens, amr2.tokens)
        except:
            stats['skip-amr-length'] += 1
            continue

        try:
            assert collections.Counter(seq1)['SHIFT'] == collections.Counter(seq2)['SHIFT'], \
                (collections.Counter(seq1)['SHIFT'], collections.Counter(seq2)['SHIFT'])
        except:
            stats['skip-shift-length'] += 1
            continue

        assert len(amr1.tokens) == collections.Counter(seq1)['SHIFT']

        stats['ok'] += 1

        amr_corpus_.append(amr1)
        amr2_corpus_.append(amr2)
        ref_actions_.append(seq1)
        new_actions_.append(seq2)

    print(stats)

    return amr_corpus_, ref_actions_, amr2_corpus_, new_actions_


def main(args):
    amr_corpus = read_amr2(args.amr, ibm_format=True)
    amr2_corpus = read_amr2(args.amr2, ibm_format=True)
    ref_actions = read_actions(args.ref)
    new_actions = read_actions(args.new)
    assert len(ref_actions) == len(new_actions)
    assert len(ref_actions) == len(amr_corpus)

    amr_corpus, ref_actions, amr2_corpus, new_actions = validate(
        amr_corpus, ref_actions, amr2_corpus, new_actions)

    m1 = Machine(amr_corpus, ref_actions)
    m2 = Machine(amr2_corpus, new_actions)

    c_copy1 = collections.Counter()
    for seq in m1.new_corpus:
        for a, tok in seq:
            if a == 'COPY':
                c_copy1[tok] += 1

    c_copy2 = collections.Counter()
    for seq in m2.new_corpus:
        for a, tok in seq:
            if a == 'COPY':
                c_copy2[tok] += 1

    names = list(c_copy1.keys())
    keys = [c_copy1[x] - c_copy2[x] for x in names]
    index = np.argsort(keys)

    for ix in index:
        x = names[ix]
        c1 = c_copy1[x]
        c2 = c_copy2[x]

        if c2 - c1 > 0:
            print('+', x, c1, c2, c2-c1)
        elif c2 - c1 < 0:
            print('-', x, c1, c2, c2-c1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amr', type=str, default=os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.txt'))
    parser.add_argument('--amr2', type=str, default=os.path.expanduser('DATA/AMR2.0/aligned/align_cfg/train.txt'))
    parser.add_argument('--ref', type=str, default=os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.actions'))
    parser.add_argument('--new', type=str, default=os.path.expanduser('DATA/AMR2.0/aligned/align_cfg/train.actions'))
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        args.amr = os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.txt')
        args.amr2 = os.path.expanduser('DATA/AMR2.0/aligned/align_cfg/dev.txt')
        args.ref = os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.actions')
        args.new = os.path.expanduser('DATA/AMR2.0/aligned/align_cfg/dev.actions')

    print(json.dumps(args.__dict__))

    main(args)
