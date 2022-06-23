import smatch
from collections import defaultdict
import numpy as np
from smatch import get_best_match, compute_f
from transition_amr_parser.amr import smatch_triples_from_penman, normalize
from transition_amr_parser.io import read_penmans
from tqdm import tqdm
from ipdb import set_trace
import penman
import argparse


def smatch_and_alignments(graph_a, graph_b):

    inst_a, att_a, rel_a, ids_a = smatch_triples_from_penman(graph_a, "a")
    inst_b, att_b, rel_b, ids_b = smatch_triples_from_penman(graph_b, "b")

    best_mapping, best_match_num = get_best_match(
        inst_a, att_a, rel_a,
        inst_b, att_b, rel_b,
        "a", "b"
    )
    # IMPORTANT: Reset cache
    smatch.match_triple_dict = {}

    # use smatch code to compute partial scores
    test_triple_num = len(inst_a) + len(att_a) + len(rel_a)
    gold_triple_num = len(inst_b) + len(att_b) + len(rel_b)

    # sentence-level normalized score
    score = compute_f(best_match_num, test_triple_num, gold_triple_num)

    # recover original map
    sorted_ids_b = [ids_b[i] if i != -1 else i for i in best_mapping]

    if len(ids_a.values()) != len(ids_a.values()):
        set_trace(context=30)

    best_id_map = dict(zip(ids_a.values(), sorted_ids_b))

    return score, best_id_map


def get_aligned_triples(pivot_index, graphs, pairs):
    '''
    Given a pivot graph, get all triples aligned to pivot triples
    '''

    # collect all aligned triples
    # instances
    pivot_triples = defaultdict(list)
    nodes = {}
    for nid, _, nname in graphs[pivot_index].instances():
        pivot_triples[nid].append(normalize(nname))
        nodes[nid] = normalize(nname)
    # attributes
    for nid, role, nname in graphs[pivot_index].attributes():
        pivot_triples[(nid, role, normalize(nname))].append(True)
    # include root
    pivot_triples[(graphs[pivot_index].top, 'top', 'TOP')].append(True)
    # edges
    for src, role, trg in graphs[pivot_index].edges():
        pivot_triples[(src, trg)].append((nodes[src], role, nodes[trg]))

    aligned_triples = defaultdict(list)
    for j, graph in enumerate(graphs):
        if j == pivot_index:
            continue

        # get map of this graphs ids to the pivot graph
        if (pivot_index, j) in pairs:
            key = (pivot_index, j)
            id_map = {v: k for k, v in pairs[key]['alignments'].items()}
        else:
            key = (j, pivot_index)
            id_map = pairs[key]['alignments']

        # collect all aligned triples
        # instances
        nodes = {}
        for nid, _, nname in graph.instances():
            aligned_triples[id_map[nid]].append(normalize(nname))
            nodes[nid] = nname
        # attributes
        for nid, role, nname in graph.attributes():
            aligned_triples[(id_map[nid], role, normalize(nname))].append(True)
        # include root
        aligned_triples[(id_map[graph.top], 'top', 'TOP')].append(True)
        # edges
        for src, role, trg  in graph.edges():
            aligned_triples[(id_map[src], id_map[trg])].append(
                (nodes[src], role, nodes[trg])
            )

    set_trace(context=30)

    return pivot_triples, aligned_triples


def argument_parser():
    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    parser.add_argument(
        "--in-amr-versions",
        help="Multiple amr files with same number of AMRs",
        nargs='+',
        type=str
    )
    args = parser.parse_args()
    return args


def main(args):

    in_amr_versions = [
    #    '/dccstor/ykt-parse/SHARED/CORPORA/MBSE/AMR2.0/ontonotes/ontonotes.g2g',
        '/dccstor/ykt-parse/SHARED/CORPORA/MBSE/AMR2.0/ontonotes/ontonotes.oracle10',
        '/dccstor/ykt-parse/SHARED/CORPORA/MBSE/AMR2.0/ontonotes/ontonotes.oracle8',
        '/dccstor/ykt-parse/SHARED/CORPORA/MBSE/AMR2.0/ontonotes/ontonotes.spring2',
        '/dccstor/ykt-parse/SHARED/CORPORA/MBSE/AMR2.0/ontonotes/ontonotes.spring22'
    ]

    # generator
    tqdm_amrs = read_penmans(in_amr_versions)
    tqdm_amrs.set_description('Computing MBSE')
    for penman_versions in tqdm_amrs:

        # TODO: Keep stats to compute pivot corpus level smatch

        # read  into the penman module class
        graphs = [penman.decode(x.split('\n')) for x in penman_versions]
        # compute smatch and alignments for the upper triangle of possible
        # matches
        pairs = {}
        for i in range(len(graphs)):
            for j in range(len(graphs)):
                if j > i:
                    score, id_map = smatch_and_alignments(graphs[i], graphs[j])
                    pairs[(i, j)] = {'smatch': score, 'alignments': id_map}

        # select best average smatch as pivot
        smatch_averages = []
        for i in range(len(graphs)):
            scores = []
            for j in range(len(graphs)):
                if i != j:
                    key = tuple(sorted([i, j]))
                    scores.append(pairs[key]['smatch'][-1])
            smatch_averages.append(np.mean(scores))
        pivot_index = np.argmax(smatch_averages)

        # Get all triples aligned to pivot triples
        pivot_triples, aligned_triples = get_aligned_triples(pivot_index, graphs, pairs)

        set_trace(context=30)
        print()


if __name__ == '__main__':
    main(argument_parser())
