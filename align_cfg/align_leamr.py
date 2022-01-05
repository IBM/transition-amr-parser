"""

ARGS:

    --align-file : Source of alignment info.
    --subset-file : Subset of alignments.
    --amr-file : Source of AMR.
    --resolve-file : (optional) Resolve ambiguities in alignments.

COMMANDS:

    a. Generate two files: 1) Resolve file, and 2) List of any completely valid alignments.

        python align.py \
            --align-file alignments.json \
            --subset-file subset.json \
            --amr-file amr.txt

    b. Incorporate resolved alignments.

        python align.py \
            --align-file alignments.json \
            --subset-file subset.json \
            --amr-file amr.txt \
            --resolve-file resolve.txt

"""

import argparse
import collections
import json

from tqdm import tqdm

from austin_amr_utils.amr_readers import AMR_Reader

from transition_amr_parser.io import read_amr2


class AlignError(ValueError):
    pass


class ValidateError(ValueError):
    pass


def parse_align(amr, align):
    for a in align:
        s = a['string'][len('subgraph : '):]
        s_tokens, s_nodes = s.split(' => ')
        tokens_ = s_tokens.strip().split()
        nodes_ = s_nodes.strip().split(', ')

        error_msg = f"{a['string']}\n{a['nodes']}\n{a['tokens']}\n{a['edges']}\n{amr.amr_string()}\n{json.dumps(a)}"

        for i, node_id in enumerate(a['nodes']):
            node_name = nodes_[i]
            check_node_name = amr.nodes[node_id]

            if check_node_name != node_name:
                new_error_msg = f'{node_id} {check_node_name} != {node_name}\n{error_msg}'

                print(f'length = {len(amr.tokens)}')
                print(new_error_msg)

                raise AlignError

    return a


class DetectAlign(object):
    def __init__(self, amr, align):
        self.amr = amr
        self.align = align

    def find_possible_nodes(self, node_id, node_name):
        n = node_id.count('.')

        possible_node_ids = []
        for x_node_id, x_node_name in self.amr.nodes.items():
            if node_name != x_node_name:
                continue
            if n != x_node_id.count('.'):
                continue
            possible_node_ids.append(x_node_id)

        return possible_node_ids

    def add_possible(self):
        align = self.align

        for a in align:
            s = a['string'][len('subgraph : '):]
            nodes_string = s.split(' => ')[1]
            nodes_ = nodes_string.strip().split(', ')

            a['node_names'] = nodes_[:len(a['nodes'])]
            a['possible'] = []

            for a_node_id, a_node_name in zip(a['nodes'], a['node_names']):
                possible_node_ids = self.find_possible_nodes(a_node_id, a_node_name)
                a['possible'].append(possible_node_ids)

    def has_ambiguity(self):
        align = self.align

        for a in align:
            for i, (a_node_id, a_possible) in enumerate(zip(a['nodes'], a['possible'])):
                if len(a_possible) > 1:
                    return True

        return False

    def validate(self):
        align = self.align

        for a in align:
            for i, (a_node_id, a_possible) in enumerate(zip(a['nodes'], a['possible'])):
                if len(a_possible) == 0:
                    raise ValidateError

    def detect(self):

        self.add_possible()

        self.validate()

        if self.has_ambiguity():
            n_ambiguous = 0

            # print AMR
            print('# AMR')
            print(self.amr.amr_string().strip())

            # print alignments
            print('# Align')

            for a in self.align:
                print(f'# Align Unit n = {len(a["node_names"])}')
                tokens = ' '.join([self.amr.tokens[x] for x in a['tokens']])

                print(f'# {tokens}')

                for a_node_name, a_possible in zip(a['node_names'], a['possible']):
                    if len(a_possible) == 1:
                        print(f'# OK {a_node_name} := {a_possible}')

                    else:
                        print(f'# XX {a_node_name} := {a_possible} :=')
                        n_ambiguous += 1

            print('')

            return n_ambiguous

        else:
            return 0



def main(args):
    align_file = args.align_file
    amr_file = args.amr_file

    reader = AMR_Reader()
    amrs = reader.load(amr_file)

    with open(args.align_file) as f:
        d_align = json.loads(f.read())

    with open(args.subset_file) as f:
        d_subset = json.loads(f.read())

    m = collections.Counter()
    for amr in tqdm(amrs):
        if amr.id not in d_align:
            m['skip-id'] += 1
            continue

        if amr.id not in d_subset:
            m['skip-subset-id'] += 1
            continue

        align = d_align[amr.id]

        detect = DetectAlign(amr, align)

        try:
            n_ambiguous = detect.detect()

        except ValidateError as e:
            m['skip-validate'] += 1
            continue

        if n_ambiguous == 0:
            m['okok'] += 1
        else:
            m['ambiguous'] += 1
            m['n_ambiguous'] += n_ambiguous

    print(m)


def compare():
    amr_file = 'amr-bank-struct-v1.6.txt'
    austin_file = 'ldc+little_prince.subgraph_alignments.json'
    gold_file = 'leamr/data-release/alignments/leamr_dev.subgraph_alignments.gold.json'

    reader = AMR_Reader()
    amrs = reader.load(amr_file)

    with open(austin_file) as f:
        d_austin = json.loads(f.read())

    with open(gold_file) as f:
        d_gold = json.loads(f.read())

    keys = [k for k in d_gold.keys() if k in d_austin]
    print('align', len(d_gold), len(keys))

    keys = [amr.id for amr in amrs if amr.id in d_gold and amr.id in d_austin]
    print('amr', len(amrs), len(keys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--align-file', type=str, default="ldc+little_prince.subgraph_alignments.json")
    parser.add_argument('--subset-file', type=str, default="leamr/data-release/alignments/leamr_dev.subgraph_alignments.gold.json")
    # parser.add_argument('--align-file', type=str, default="leamr/data-release/alignments/leamr_dev.subgraph_alignments.gold.json")
    parser.add_argument('--amr-file', type=str, default="amr-bank-struct-v1.6.txt")
    parser.add_argument('--resolve-file', type=str, default=None)
    args = parser.parse_args()

    # compare()
    main(args)

