import smatch
from collections import defaultdict
import numpy as np
from smatch import get_best_match, compute_f
from transition_amr_parser.amr import smatch_triples_from_penman, normalize
from transition_amr_parser.io import read_penmans
from ipdb import set_trace
import penman
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    parser.add_argument(
        "--in-amrs",
        help="Multiple amr files with same number of AMRs",
        nargs='+',
        required=True,
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="Output amr ensemble",
        required=True,
        type=str
    )
    parser.add_argument(
        "-r", # "--iteration-num",
        help="number of re-starts for Smatch",
        default=10,
        type=int
    )
    args = parser.parse_args()
    return args


def counts_from_alignments(graph_a, graph_b, best_id_map):

    # get triples
    triples_a = get_triples(graph_a)
    triples_b = get_triples(graph_b)

    best_rev_id_map = {v: k for k, v in best_id_map.items()}
    hits = {'instances': 0, 'attributes': 0, 'relations': 0}

    # sanity check
    for part in triples_b.keys():
        if any(k == -1 for k in triples_b[part].keys()):
            set_trace(context=30)
            print()

    # instances
    for ref_id, ref_value in triples_b['instances'].items():
        if ref_id in best_rev_id_map:
            # node missing from decoded
            dec_id = best_rev_id_map[ref_id]
            if (
                dec_id in triples_a['instances']
                and triples_a['instances'][dec_id] == ref_value
            ):
                hits['instances'] += 1

    # attributes
    for ref_id, ref_value in triples_b['attributes'].items():
        if ref_id in best_rev_id_map:
            # node missing from decoded
            dec_id = best_rev_id_map[ref_id]
            if (
                dec_id in triples_a['attributes']
                and triples_a['attributes'][dec_id] == ref_value
            ):
                hits['attributes'] += 1

    # relations
    for ref_id, ref_value in triples_b['relations'].items():
        if ref_id[0] in best_rev_id_map and ref_id[1] in best_rev_id_map:
            # node missing from decoded
            dec_id = (best_rev_id_map[ref_id[0]], best_rev_id_map[ref_id[1]])
            if (
                dec_id in triples_a['relations']
                # NOTE: enough with edge labels matching
                and triples_a['relations'][dec_id][0][1] == ref_value[0][1]
            ):
                hits['relations'] += 1

    test_triple_num = len([x for x in triples_a.values() for y in x])
    gold_triple_num = len([x for x in triples_b.values() for y in x])
    counts = [sum(hits.values()), test_triple_num, gold_triple_num]
    scores = compute_f(*counts)

    return counts, scores


def smatch_and_alignments(graph_a, graph_b):

    # smatch aligner
    inst_a, att_a, rel_a, ids_a = smatch_triples_from_penman(graph_a, "a")
    inst_b, att_b, rel_b, ids_b = smatch_triples_from_penman(graph_b, "b")
    best_mapping, best_match_num = get_best_match(
        inst_a, att_a, rel_a,
        inst_b, att_b, rel_b,
        "a", "b"
    )
    # IMPORTANT: Reset cache
    smatch.match_triple_dict = {}
    # recover original map
    sorted_ids_b = [ids_b[i] if i != -1 else i for i in best_mapping]
    if len(ids_a.values()) != len(ids_a.values()):
        set_trace(context=30)
    best_id_map = dict(zip(ids_a.values(), sorted_ids_b))

    # sentence-level normalized score
    # use smatch code to compute partial scores
    test_triple_num = len(inst_a) + len(att_a) + len(rel_a)
    gold_triple_num = len(inst_b) + len(att_b) + len(rel_b)
    score = compute_f(best_match_num, test_triple_num, gold_triple_num)
    counts = [best_match_num, test_triple_num, gold_triple_num]

    # Sanity check: compute scores from alignments
    # counts, scores2 = counts_from_alignments(graph_a, graph_b, best_id_map)

    return counts, score, best_id_map


def get_triples(graph):

    # collect all aligned triples
    # instances
    triples = {
        'instances': defaultdict(list),
        'relations': defaultdict(list),
        'attributes': defaultdict(list),
    }
    nodes = {}
    for nid, _, nname in graph.instances():
        triples['instances'][nid].append(normalize(nname))
        nodes[nid] = normalize(nname)
    triples['instances'] = dict(triples['instances'])

    # attributes
    for nid, role, nname in graph.attributes():
        # sanity check
        assert nname is not None
        triples['attributes'][nid].append((role, normalize(nname)))

    triples['attributes'] = dict(triples['attributes'])
    # include root
    triples['attributes'][graph.top] = [('top', 'TOP')]

    # edges
    for src, role, trg in graph.edges():
        triples['relations'][(src, trg)].append((nodes[src], role, nodes[trg]))
    triples['relations'] = dict(triples['relations'])
    return dict(triples)


def get_aligned_triples(pivot_index, graphs, pairs):
    '''
    Given a pivot graph, get all triples aligned to pivot triples
    '''

    pivot_triples = get_triples(graphs[pivot_index])

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
        for src, role, trg in graph.edges():
            aligned_triples[(id_map[src], id_map[trg])].append(
                (nodes[src], role, nodes[trg])
            )

    return pivot_triples, aligned_triples


def graphene_coverage(pivot_graph, graph, best_rev_id_map):

    # get triples
    # NOTE: first is the pivot
    triples_a = get_triples(graph)
    triples_b = get_triples(pivot_graph)

    # sanity check
    for part in triples_b.keys():
        if any(k == -1 for k in triples_b[part].keys()):
            set_trace(context=30)
            print()

    votes = {'instances': {}, 'attributes': {}, 'relations': {}}

    # instances and attributes
    used_ids = []
    for part in ['instances', 'attributes']:
        for ref_id, ref_value in triples_b[part].items():
            if ref_id in best_rev_id_map:
                # node missing from decoded
                dec_id = best_rev_id_map[ref_id]
                votes[part][ref_id] = triples_a[part].get(dec_id, None)
                used_ids.append(dec_id)
            else:
                set_trace(context=30)
                print()
        # unaligned
        missing_ids = sorted(triples_a[part].keys() - set(used_ids))
        for nid in missing_ids:
            if None in votes[part]:
                votes[part][None].extend(triples_a[part][nid])
            else:
                votes[part][None] = triples_a[part][nid]

    # relations
    used_ids = []
    for ref_id, ref_value in triples_b['relations'].items():
        if ref_id[0] in best_rev_id_map and ref_id[1] in best_rev_id_map:
            # node missing from decoded
            dec_id = (best_rev_id_map[ref_id[0]], best_rev_id_map[ref_id[1]])
            votes['relations'][ref_id] = \
                triples_a['relations'].get(dec_id, None)
            used_ids.append(dec_id)
        else:
            set_trace(context=30)
            print()
    # unaligned
    missing_ids = sorted(triples_a['relations'].keys() - set(used_ids))
    for nid in missing_ids:
        if None in votes['relations']:
            votes['relations'][None].extend(triples_a['relations'][nid])
        else:
            votes['relations'][None] = triples_a['relations'][nid]

    # sanity check: recover Smatch counts

    return votes


def graphene_candidates(graphs, pairs):
    pass


def main(args):

    # set number of re-starts in Smatch
    smatch.iteration_num = args.r + 1

    # generator
    tqdm_amrs = read_penmans(args.in_amrs)
    tqdm_amrs.set_description('Computing MBSE')
    out_amrs = []
    for sent_index, penman_versions in enumerate(tqdm_amrs):

        # TODO: Keep stats to compute pivot corpus level smatch

        # read  into the penman module class
        graphs = [penman.decode(x.split('\n')) for x in penman_versions]
        # compute smatch and alignments for the upper triangle of possible
        # matches
        pairs = {}
        for i in range(len(graphs)):
            for j in range(len(graphs)):
                if j > i:
                    items = graphs[i], graphs[j]
                    counts, score, id_map = smatch_and_alignments(*items)
                    votes = graphene_coverage(*items, id_map)
                    pairs[(i, j)] = {
                        'counts': counts,
                        'smatch': score,
                        'alignments': id_map,
                        'votes': votes
                    }

        # Add graphene candidates
        # graphs, pairs = graphene_candidates(graphs, pairs)

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

        # keep the best AMR
        out_amrs.append(graphs[pivot_index])

    if args.out_amr:
        with open(args.out_amr, 'w') as fid:
            for graph in out_amrs:
                fid.write(f'{penman.encode(graph, indent=4)}\n\n')


if __name__ == '__main__':
    main(argument_parser())
