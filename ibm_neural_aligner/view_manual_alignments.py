"""
not found manual_dev
- overlap 150
- notfound 0
not found manual_test
- overlap 188
- notfound 12
- edinburgh_1003.8
- edinburgh_1003.9
- edinburgh_1003.3
- edinburgh_1003.4
- edinburgh_1003.7
- edinburgh_1003.10
- edinburgh_1003.1
- edinburgh_1003.2
- edinburgh_1003.5
- NATHANS_EXAMPLE
- edinburgh_1003.6
- AUSTINS_EXAMPLE
"""


import argparse
import collections
import json

from tqdm import tqdm

from transition_amr_parser.io import read_amr2


MY_GLOBALS = {}
MY_GLOBALS['found'] = set()


def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())


def get_keys(corpus):
    if isinstance(corpus, (tuple, list)):
        d = {}
        deleted = set()
        for amr in corpus:
            if amr.id in deleted:
                continue
            if amr.id in d:
                del d[amr.id]
                print(f'deleted {amr.id}')
                continue
            d[amr.id] = amr
        return get_keys(d)
    return corpus.keys()


def print_overlap(datasets, name_a, name_b):
    keys_a = set(get_keys(datasets[name_a]))
    keys_b = set(get_keys(datasets[name_b]))
    overlap = set.intersection(keys_a, keys_b)
    print(f'overlap\n- {name_a} = {len(keys_a)}\n- {name_b} = {len(keys_b)}\n- overlap = {len(overlap)}')

    # Update found.
    MY_GLOBALS['found'] = set.union(MY_GLOBALS['found'], overlap)


def check_overlap_austin_and_manual(datasets):
    print_overlap(datasets, 'austin', 'manual_dev')
    print_overlap(datasets, 'austin', 'manual_test')


def check_overlap_prince_and_manual(datasets):
    print_overlap(datasets, 'prince_amr', 'manual_dev')
    print_overlap(datasets, 'prince_amr', 'manual_test')


def check_overlap_amr3_and_manual(datasets):
    print_overlap(datasets, 'amr3_train', 'manual_dev')
    print_overlap(datasets, 'amr3_train', 'manual_test')

    print_overlap(datasets, 'amr3_dev', 'manual_dev')
    print_overlap(datasets, 'amr3_dev', 'manual_test')

    print_overlap(datasets, 'amr3_test', 'manual_dev')
    print_overlap(datasets, 'amr3_test', 'manual_test')


def check_notfound(datasets):

    for name in ['manual_dev', 'manual_test']:
        print(f'not found {name}')
        keys = set(datasets[name].keys())
        overlap = set.intersection(MY_GLOBALS['found'], keys)
        notfound = {k for k in keys if k not in overlap}
        print(f'- overlap {len(overlap)}')
        print(f'- notfound {len(notfound)}')
        for k in notfound:
            print(f'- {k}')


def main():
    paths = {}

    # This has some useful information such as node names, but it is not clear
    # which are manual alignments.
    paths['austin'] = 'ldc+little_prince.subgraph_alignments.json'

    # This does not have node names, but does have AMR ids for manually aligned AMR.
    paths['manual_dev'] = "leamr/data-release/alignments/leamr_dev.subgraph_alignments.gold.json"
    paths['manual_test'] = "leamr/data-release/alignments/leamr_test.subgraph_alignments.gold.json"

    # Path to little prince data.
    paths['prince_amr'] = 'amr-bank-struct-v1.6.dummy_align.txt'

    # Path to amr3 data.
    paths['amr3_train'] = 'DATA/AMR3.0/corpora/train.dummy_align.txt'
    paths['amr3_dev'] = 'DATA/AMR3.0/corpora/dev.dummy_align.txt'
    paths['amr3_test'] = 'DATA/AMR3.0/corpora/test.dummy_align.txt'

    for k, v in paths.items():
        print(k, v)

    datasets = {}
    datasets = {k: read_amr2(v, ibm_format=True, tokenize=False) if 'amr' in k else read_json(v) for k, v in paths.items()}

    check_overlap_austin_and_manual(datasets)
    check_overlap_prince_and_manual(datasets)
    check_overlap_amr3_and_manual(datasets)
    check_notfound(datasets)


if __name__ == '__main__':
    main()
