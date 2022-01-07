"""
At first:

    Counter({'ok': 3417, 'found_many': 629, 'maybe_ok': 22})
"""

import argparse
import collections
import copy
import json

from tqdm import tqdm

from formatter import amr_to_string

from transition_amr_parser.io import read_amr2


MY_GLOBALS = {}
MY_GLOBALS['stats'] = collections.Counter()


def is_int(x):
    try:
        _ = int(x)
        return True
    except ValueError:
        return False


def read_resolve_file(filename):
    corpus = {}

    prev_val = None

    i_alternate = 0

    with open(filename) as f:
        for line in f:
            if line.startswith('# ::id'):
                val = line.strip().split()[-1]
                prev_val = val
                corpus[val] = {'align': [], 'resolve': []}

            elif line.startswith('###'):
                _, msg, o = line.strip().split(' ', 2)
                o = json.loads(o)

                if is_int(msg):
                    corpus[prev_val]['align'].append(o)
                    corpus[prev_val]['resolve'].append([])

                else:
                    corpus[prev_val]['resolve'][-1].append(o)

    return corpus


def read_amr2_as_dict(filename):
    corpus = read_amr2(filename, ibm_format=True, tokenize=False)

    new_corpus = {}

    deleted = set()
    for amr in corpus:
        if amr.id in deleted:
            continue
        if amr.id in new_corpus:
            del new_corpus[amr.id]
            deleted.add(amr.id)
            print(f'deleted {amr.id}')
            continue
        new_corpus[amr.id] = amr

    print(filename, len(corpus), len(new_corpus))

    return new_corpus


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


def get_node_names(align, align_string):
    s = align_string[len('subgraph : '):]
    nodes_string = s.split(' => ')[1]
    node_names = nodes_string.strip().split(', ')[:len(align['nodes'])]
    return node_names


def attempt_resolve(datasets, k_amr_align_with_string, k_amr_align, k_amr_corpus, use_string=True):
    """

    If use_string is True, then don't trust node variable names.

    Otherwise, do.

    """
    amr_align_with_string = datasets[k_amr_align_with_string]
    amr_align = copy.deepcopy(datasets[k_amr_align])
    amr_corpus = datasets[k_amr_corpus]

    # Get subset.
    removed = set()
    for k in list(amr_align.keys()):
        if k not in amr_corpus or k not in amr_align_with_string:
            removed.add(k)
            del amr_align[k]

    print(f'RESOLVE {k_amr_align_with_string} {k_amr_align} {k_amr_corpus}')
    print(f'- align {len(amr_align)}')
    print(f'- removed {len(removed)}')

    for k in list(amr_align.keys()):
        amr = amr_corpus[k]
        align_list = amr_align_with_string[k]
        resolve = None

        if 'resolve' in datasets and k in datasets['resolve']:
            resolve = datasets['resolve'][k]

        if use_string:
            possible_alignments = resolve_align_with_string(amr, align_list, resolve)

        else:
            possible_alignments = resolve_align_with_variable_names(amr, align_list)


def resolve_align_with_string(amr, align_list, resolve=None):

    if resolve is not None:
        d_resolve = {}

        for a, r in zip(resolve['align'], resolve['resolve']):
            assert len(r) > 0

            for rr in r:
                assert len(rr) == 1

                for fake_node_id, possible_node_ids in rr.items():
                    d_resolve[fake_node_id] = possible_node_ids

        def get_possible(amr, fake_node_id, node_name):
            possible_node_ids = d_resolve[fake_node_id]
            for k in possible_node_ids:
                assert amr.nodes[k] == node_name
            return possible_node_ids

    else:
        def get_possible(amr, fake_node_id, node_name):
            return [k for k, v in amr.nodes.items() if v == node_name]

    fake_alignments = {k: [0] for k in amr.nodes.keys()}
    f_resolve.write(amr_to_string(amr, alignments=fake_alignments).strip() + '\n')

    seen = set()

    for i, align in enumerate(align_list):

        align_string = align['string']
        fake_node_ids = align['nodes']
        node_names = get_node_names(align, align_string)

        # Header
        f_resolve.write(f'### {i} {json.dumps(align)}' + '\n')

        # Body
        for fake_node_id, node_name in zip(fake_node_ids, node_names):
            possible_node_ids = get_possible(amr, fake_node_id, node_name)

            o = {fake_node_id: possible_node_ids}

            if len(possible_node_ids) == 1:
                msg = 'ok'
            elif len(possible_node_ids) == 0:
                msg = 'found_none'
            else:
                if fake_node_id in amr.nodes and node_name == amr.nodes[fake_node_id]:
                    msg = 'maybe_ok'
                else:
                    msg = 'found_many'

            f_resolve.write(f'### {msg} {json.dumps(o)}' + '\n')

            MY_GLOBALS['stats'][msg] += 1

        # CHECK
        assert len(fake_node_ids) == len(node_names)

        # CHECK
        node_ids = align['nodes']
        for node_id in node_ids:
            assert node_id not in seen
            seen.add(node_id)

    f_resolve.write('\n')


def resolve_align_with_variable_names(amr, align_list):

    fake_alignments = {k: [0] for k in amr.nodes.keys()}
    f_resolve.write(amr_to_string(amr, alignments=fake_alignments).strip() + '\n')

    seen = set()

    for i, align in enumerate(align_list):

        # Header
        f_resolve.write(f'### {i} {json.dumps(align)}\n')

        # Body
        node_ids = align['nodes']
        for node_id in node_ids:
            assert node_id in amr.nodes
            o = {node_id: [node_id]}
            f_resolve.write(f'### ok {json.dumps(o)}\n')

        # CHECK
        node_ids = align['nodes']
        for node_id in node_ids:
            assert node_id not in seen
            seen.add(node_id)

    f_resolve.write('\n')


def attempt_resolve_prince(datasets):
    attempt_resolve(datasets, 'austin', 'manual_dev', 'prince_amr')
    attempt_resolve(datasets, 'austin', 'manual_test', 'prince_amr')


def attempt_resolve_additional(datasets):
    attempt_resolve(datasets, 'manual_test', 'manual_test', 'additional_amr', use_string=False)


def attempt_resolve_amr3(datasets):
    attempt_resolve(datasets, 'austin', 'manual_dev', 'amr3_train')
    attempt_resolve(datasets, 'austin', 'manual_test', 'amr3_train')

    attempt_resolve(datasets, 'austin', 'manual_dev', 'amr3_dev')
    attempt_resolve(datasets, 'austin', 'manual_test', 'amr3_dev')

    attempt_resolve(datasets, 'austin', 'manual_dev', 'amr3_test')
    attempt_resolve(datasets, 'austin', 'manual_test', 'amr3_test')


def main():
    paths = {}

    # This has some useful information such as node names, but it is not clear
    # which are manual alignments.
    paths['austin'] = 'ldc+little_prince.subgraph_alignments.json'

    # This does not have node names, but does have AMR ids for manually aligned AMR.
    paths['manual_dev'] = "leamr/data-release/alignments/leamr_dev.subgraph_alignments.gold.json"
    paths['manual_test'] = "leamr/data-release/alignments/leamr_test.subgraph_alignments.gold.json"

    # Path to little prince data.
    paths['prince_amr'] = 'amr-bank-struct-v3.0.leamr.txt'
    paths['additional_amr'] = 'leamr/data-release/amrs/additional_amrs.txt'

    # Path to amr3 data.
    # paths['amr3_train'] = 'DATA/AMR3.0/corpora/train.dummy_align.txt'
    # paths['amr3_dev'] = 'DATA/AMR3.0/corpora/dev.dummy_align.txt'
    # paths['amr3_test'] = 'DATA/AMR3.0/corpora/test.dummy_align.txt'

    for k, v in paths.items():
        print(k, v)

    datasets = {}
    datasets = {k: read_amr2_as_dict(v) if 'amr' in k else read_json(v) for k, v in paths.items()}

    datasets['resolve'] = read_resolve_file('resolve.manual.txt')

    attempt_resolve_prince(datasets)
    attempt_resolve_additional(datasets)
    # attempt_resolve_amr3(datasets)

    print(MY_GLOBALS['stats'])


if __name__ == '__main__':
    f_resolve = open('resolve.txt', 'w')

    main()

    f_resolve.close()
