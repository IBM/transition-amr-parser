"""
At first:

    Counter({'ok': 3417, 'found_many': 629, 'success-True': 209, 'success-False': 141, 'maybe_ok': 22})
    prince_amr manual_dev data_manual_align/new.prince_amr.manual_dev.txt 42
    prince_amr manual_test data_manual_align/new.prince_amr.manual_test.txt 42
    additional_amr manual_test data_manual_align/new.additional_amr.manual_test.txt 12
    amr3_train manual_test data_manual_align/new.amr3_train.manual_test.txt 62
    amr3_dev manual_dev data_manual_align/new.amr3_dev.manual_dev.txt 45
    amr3_dev manual_test data_manual_align/new.amr3_dev.manual_test.txt 2
    amr3_test manual_test data_manual_align/new.amr3_test.manual_test.txt 4

After resolving Little Prince and a few others:

    Counter({'ok': 3441, 'found_many': 627, 'success-True': 220, 'success-False': 130})

    prince_amr manual_dev data_manual_align/new.prince_amr.manual_dev.txt 50
    prince_amr manual_test data_manual_align/new.prince_amr.manual_test.txt 45
    additional_amr manual_test data_manual_align/new.additional_amr.manual_test.txt 12
    amr3_train manual_test data_manual_align/new.amr3_train.manual_test.txt 62
    amr3_dev manual_dev data_manual_align/new.amr3_dev.manual_dev.txt 45
    amr3_dev manual_test data_manual_align/new.amr3_dev.manual_test.txt 2
    amr3_test manual_test data_manual_align/new.amr3_test.manual_test.txt 4
"""

import argparse
import collections
import copy
import json
import os

from tqdm import tqdm

from transition_amr_parser.io import read_amr


MY_GLOBALS = {}
MY_GLOBALS['stats'] = collections.Counter()
MY_GLOBALS['new_amr'] = collections.defaultdict(list)
MY_GLOBALS['skip_amr'] = collections.defaultdict(list)


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
    corpus = read_amr(filename)

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
            new_alignments, success = resolve_align_with_string(amr, align_list, resolve)

        else:
            new_alignments, success = resolve_align_with_variable_names(amr, align_list)

        MY_GLOBALS['stats'][f'success-{success}'] += 1

        if success:
            MY_GLOBALS['new_amr'][(k_amr_align, k_amr_corpus)].append((amr, new_alignments))
        else:
            MY_GLOBALS['skip_amr'][(k_amr_align, k_amr_corpus)].append((amr, None))


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
    amr.alignments = fake_alignments
    f_resolve.write(f'{amr.__str__()}\n')

    all_ok = True

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

            if len(possible_node_ids) != 1:
                all_ok = False

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

    f_resolve.write('\n')

    # CHECK don't align twice.
    if all_ok:
        seen = set()
        for i, align in enumerate(align_list):
            align_string = align['string']
            fake_node_ids = align['nodes']
            node_names = get_node_names(align, align_string)

            for fake_node_id, node_name in zip(fake_node_ids, node_names):
                node_id = get_possible(amr, fake_node_id, node_name)[0]
                assert node_id not in seen
                seen.add(node_id)

    # TODO: Get the final alignments!
    new_alignments = {}

    if not all_ok:
        return new_alignments, False

    for i, align in enumerate(align_list):
        tokens = align['tokens']
        align_string = align['string']
        fake_node_ids = align['nodes']
        node_names = get_node_names(align, align_string)

        for fake_node_id, node_name in zip(fake_node_ids, node_names):
            node_id = get_possible(amr, fake_node_id, node_name)[0]
            new_alignments[node_id] = tokens

    return new_alignments, True


def resolve_align_with_variable_names(amr, align_list):

    fake_alignments = {k: [0] for k in amr.nodes.keys()}
    amr.alignments = fake_alignments
    f_resolve.write(f'{amr.__str__()}\n')

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

    # Get the final alignments!
    new_alignments = {}

    for i, align in enumerate(align_list):
        tokens = align['tokens']
        for fake_node_id in align['nodes']:
            node_id = fake_node_id
            new_alignments[node_id] = tokens

    return new_alignments, True


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

def do_write(d_align, d, output_file):
    new_corpus = []
    for corpus in d:
        for k, amr in corpus.items():
            if k not in d_align:
                continue
            new_corpus.append(amr)

    print(f'writing... {output_file} {len(new_corpus)}')
    with open(output_file, 'w') as f:
        fake_alignments = {k: [0] for k in amr.nodes.keys()}
        amr.alignments = fake_alignments
        f.write(f'{amr.__str__()}\n')

def write_amr3_train(datasets, output_dir):
    d_align = set.union(set(datasets['manual_dev'].keys()), set(datasets['manual_test'].keys()))
    d = [datasets[k] for k in ['amr3_train']]
    do_write(d_align, d, os.path.join(output_dir, 'gold.amr3_train.txt'))

def write_amr3_unseen(datasets, output_dir):
    d_align = set.union(set(datasets['manual_dev'].keys()), set(datasets['manual_test'].keys()))
    d = [datasets[k] for k in ['amr3_dev', 'amr3_test']]
    do_write(d_align, d, os.path.join(output_dir, 'gold.amr3_unseen.txt'))

def write_all(datasets, output_dir):
    d_align = set.union(set(datasets['manual_dev'].keys()), set(datasets['manual_test'].keys()))
    d = [datasets[k] for k in datasets.keys() if 'amr' in k]
    do_write(d_align, d, os.path.join(output_dir, 'gold.all.txt'))

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
    paths['amr3_train'] = 'DATA/AMR3.0/aligned/cofill/train.txt'
    paths['amr3_dev'] = 'DATA/AMR3.0/aligned/cofill/dev.txt'
    paths['amr3_test'] = 'DATA/AMR3.0/aligned/cofill/test.txt'

    for k, v in paths.items():
        print(k, v)

    datasets = {}
    datasets = {k: read_amr2_as_dict(v) if 'amr' in k else read_json(v) for k, v in paths.items()}

    datasets['resolve'] = read_resolve_file('resolve.manual.txt')

    attempt_resolve_prince(datasets)
    attempt_resolve_additional(datasets)
    attempt_resolve_amr3(datasets)

    print(MY_GLOBALS['stats'])

    output_dir = 'data_manual_align'

    write_amr3_train(datasets, output_dir)
    write_amr3_unseen(datasets, output_dir)
    write_all(datasets, output_dir)

    def write_ok():
        for (k_amr_align, k_amr_corpus), corpus in MY_GLOBALS['new_amr'].items():
            os.system(f'mkdir -p {output_dir}')

            new_file = os.path.join(output_dir, f'new.{k_amr_corpus}.{k_amr_align}.txt')
            print(k_amr_corpus, k_amr_align, new_file, len(corpus))

            with open(new_file, 'w') as f:
                for amr, alignments in corpus:
                    amr.alignments = alignments
                    f.write(f'{amr.__str__()}\n')

    def print_ok_stats():
        print('ok')
        for (k_amr_align, k_amr_corpus), corpus in MY_GLOBALS['new_amr'].items():
            print(k_amr_corpus, k_amr_align, len(corpus))

    def print_missing():
        print('missing')
        for (k_amr_align, k_amr_corpus), corpus in MY_GLOBALS['skip_amr'].items():
            print(k_amr_corpus, k_amr_align, len(corpus))
            for amr, _ in corpus:
                print(f'- {amr.id}')

    def print_missing_stats():
        print('missing')
        for (k_amr_align, k_amr_corpus), corpus in MY_GLOBALS['skip_amr'].items():
            print(k_amr_corpus, k_amr_align, len(corpus))

    write_ok()
    print_missing()
    print_missing_stats()
    print_ok_stats()



if __name__ == '__main__':
    f_resolve = open('resolve.txt', 'w')

    main()

    f_resolve.close()
