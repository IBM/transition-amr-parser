import json
from tqdm import tqdm
import argparse
import re
import subprocess
import sys
import re
from ipdb import set_trace
from transition_amr_parser.io import read_blocks, AMR
# this requires local smatch installed --editable and touch smatch/__init__.py
from smatch.smatch import get_amr_match, get_best_match, compute_f
from smatch import amr
import smatch


def format_penman(x):
    lines = '\n'.join([x for x in x.split('\n') if x and x[0] != '#'])
    lines = re.sub(r'\~[0-9]+', '', lines)
    return lines.replace('\n', '')


def vimdiff(penman1, penman2):

    def write(file_name, content):
        with open(file_name, 'w') as fid:
            fid.write(content)

    write('_tmp1', penman1)
    write('_tmp2', penman2)
    subprocess.call(['vimdiff', '_tmp1', '_tmp2'])


def redo_match(mapping, weight_dict):

    node_match_num = 0
    edge_match_num = 0
    # i is node index in AMR 1, m is node index in AMR 2
    for i, m in enumerate(mapping):
        if m == -1:
            # no node maps to this node
            continue
        # node i in AMR 1 maps to node m in AMR 2
        current_node_pair = (i, m)
        if current_node_pair not in weight_dict:
            continue

        for key in weight_dict[current_node_pair]:
            if key == -1:
                # matching triple resulting from instance/attribute triples
                node_match_num += weight_dict[current_node_pair][key]
            # only consider node index larger than i to avoid duplicates
            # as we store both weight_dict[node_pair1][node_pair2] and
            #     weight_dict[node_pair2][node_pair1] for a relation
            elif key[0] < i:
                continue
            elif mapping[key[0]] == key[1]:
                edge_match_num += weight_dict[current_node_pair][key]

    return node_match_num, edge_match_num


def main(args):

    # read files
    gold_penmans = read_blocks(args.in_reference_amr)
    oracle_penmans = read_blocks(args.in_amr)

    # set global in module
    smatch.smatch.iteration_num = args.r + 1

    assert len(gold_penmans) == len(oracle_penmans)

    statistics = []
    sentence_smatch = []
    node_id_maps = []
    for index in tqdm(range(len(oracle_penmans))):

        # format penman ot be read by smatch penman parser
        gold_penman = format_penman(gold_penmans[index])
        oracle_penman = format_penman(oracle_penmans[index])

        # parse penman using smatchs parser
        amr1 = amr.AMR.parse_AMR_line(gold_penman)
        amr2 = amr.AMR.parse_AMR_line(oracle_penman)

        # get ids before changing prefix
        ids_a = [x[1] for x in amr1.get_triples()[0]]
        ids_b = [x[1] for x in amr2.get_triples()[0]]

        # prefix name as per original code
        prefix1 = "a"
        prefix2 = "b"
        amr1.rename_node(prefix1)
        amr2.rename_node(prefix2)

        # use smatch code to align nodes
        (instance1, attributes1, relation1) = amr1.get_triples()
        (instance2, attributes2, relation2) = amr2.get_triples()
        best_mapping, best_match_num, weight_dict = get_best_match(
            instance1, attributes1, relation1,
            instance2, attributes2, relation2,
            prefix1, prefix2
        )
        # IMPORTANT: Reset cache
        smatch.smatch.match_triple_dict = {}
        # use smatch code to compute partial scores
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
        # accumulate statistics for corpus-level normalization
        statistics.append([best_match_num, test_triple_num, gold_triple_num])
        # store sentence-level normalized score
        score = compute_f(best_match_num, test_triple_num, gold_triple_num)
        sentence_smatch.append(score)

        # store alignments
        sorted_ids_b = [ids_b[i] for i in best_mapping]
        best_id_map = dict(zip(ids_a, sorted_ids_b))
        node_id_maps.append(best_id_map)

        if args.stop_if_different and score[-1] < 1.0:
            amr1_p = AMR.from_penman(gold_penmans[index])
            amr2_p = AMR.from_penman(oracle_penmans[index])
            # for k, v in best_id_map.items(): print(amr1_p.nodes[k], amr2_p.nodes[v])
            set_trace()
            node_match_num, edge_match_num = redo_match(best_mapping, weight_dict)
            vimdiff(gold_penmans[index], oracle_penmans[index])
            set_trace()
            print()

        # for debug
        # amr1_p = AMR.from_penman(gold_penmans[i])
        # amr2_p = AMR.from_penman(oracle_penmans[i])
        # for k, v in best_id_map.items(): print(amr1_p.nodes[k], amr2_p.nodes[v])

    # TODO: Save scores and alignments


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    parser.add_argument(
        "--in-reference-amr",
        help="file with reference AMRs in penman format",
        type=str
    )
    parser.add_argument(
        "--in-amr",
        help="file with AMRs in penman format",
        type=str
    )
    parser.add_argument(
        "-r",
        help="Number of restarts",
        type=int,
        default=4
    )
    # flags
    parser.add_argument(
        "--stop-if-different",
        help="If sentence Smatch is not 1.0, breakpoint",
        action='store_true'
    )


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
