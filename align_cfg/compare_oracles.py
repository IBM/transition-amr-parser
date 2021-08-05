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
import sys

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

    # amr_corpus, ref_actions, amr2_corpus, new_actions = validate(
    #     amr_corpus, ref_actions, amr2_corpus, new_actions)

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

    # check SHIFT
    c1 = collections.Counter()
    for seq in ref_actions:
        c1.update(seq)
    c2 = collections.Counter()
    for seq in new_actions:
        c2.update(seq)

    print('COPY {} {}'.format(c1['COPY'], c2['COPY']))
    print('SHIFT {} {}'.format(c1['SHIFT'], c2['SHIFT']))

    if c1['SHIFT'] != c2['SHIFT']:
        print('WARNING: Found different amount of SHIFTS. {} != {}'.format(c1['SHIFT'], c2['SHIFT']), file=sys.stderr)

    l1 = np.mean([len(x) for x in ref_actions])
    l2 = np.mean([len(x) for x in new_actions])
    print('AVG_LENGTH {:.3f} {:.3f}'.format(l1, l2))

    def ignore_shift(seq):
        return [x for x in seq if x != 'SHIFT']

    l1 = np.mean([len(ignore_shift(x)) for x in ref_actions])
    l2 = np.mean([len(ignore_shift(x)) for x in new_actions])
    print('AVG_LENGTH[IGNORE_SHIFT] {:.3f} {:.3f}'.format(l1, l2))

    def only_arcs(seq):
        return [x for x in seq if x.startswith('>')]

    def is_node(seq):
        return [(not x.startswith('>')) and x != 'SHIFT' for x in seq]

    def is_arc(seq):
        return [x.startswith('>') for x in seq]

    c = collections.defaultdict(list)

    for i, (seq1, seq2) in enumerate(zip(ref_actions, new_actions)):

        cum_nodes = np.cumsum(is_node(seq1))
        num_candidate_nodes = cum_nodes[is_arc(seq1)]
        c['pool_candidate_nodes_1'].append(num_candidate_nodes.sum())
        c['candidate_nodes_1'].append(num_candidate_nodes)

        cum_nodes = np.cumsum(is_node(seq2))
        num_candidate_nodes = cum_nodes[is_arc(seq2)]
        c['pool_candidate_nodes_2'].append(num_candidate_nodes.sum())
        c['candidate_nodes_2'].append(num_candidate_nodes)

    n1 = np.mean(c['pool_candidate_nodes_1'])
    n2 = np.mean(c['pool_candidate_nodes_2'])
    print('AVG_POOL_CANDIDATE_NODES {:.3f} {:.3f}'.format(n1, n2))

    diff = np.array(c['pool_candidate_nodes_1']) - np.array(c['pool_candidate_nodes_2'])
    c1 = c['pool_candidate_nodes_1']
    c2 = c['pool_candidate_nodes_2']
    argmax = np.argmax(diff)
    argmin = np.argmin(diff)
    print('MIN_DIFF_CANDIDATE_NODES {:.3f} {} {}'.format(diff.min(), c1[argmin], c2[argmin]))
    print('MAX_DIFF_CANDIDATE_NODES {:.3f} {} {}'.format(diff.max(), c1[argmax], c2[argmax]))

    import ipdb; ipdb.set_trace()
    pass

    # for i, (seq1, seq2, amr1, amr2) in enumerate(zip(ref_actions, new_actions, amr_corpus, amr2_corpus)):

    #     cum_nodes_1 = np.cumsum(is_node(seq1))
    #     num_candidate_nodes_1 = cum_nodes_1[is_arc(seq1)]

    #     cum_nodes_2 = np.cumsum(is_node(seq2))
    #     num_candidate_nodes_2 = cum_nodes_2[is_arc(seq2)]

    #     n1 = num_candidate_nodes_1.sum()
    #     n2 = num_candidate_nodes_2.sum()
    #     diff = n1 - n2

    #     if np.abs(diff) > 10:
    #         import ipdb; ipdb.set_trace()
    #         pass


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
