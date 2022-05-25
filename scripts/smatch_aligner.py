from tqdm import tqdm
import argparse
import re
import subprocess
from collections import defaultdict
from ipdb import set_trace
from transition_amr_parser.io import read_blocks
from transition_amr_parser.amr import AMR, get_is_atribute, normalize
# this requires local smatch installed --editable and touch smatch/__init__.py
from transition_amr_parser.clbar import yellow_font
from smatch.smatch import get_best_match, compute_f
from smatch import amr
import smatch


def format_penman(x):
    # remove comments
    lines = '\n'.join([x for x in x.split('\n') if x and x[0] != '#'])
    # remove ISI alignments if any
    lines = re.sub(r'\~[0-9,]+', '', lines)
    return lines.replace('\n', '')


def protected_read(penman, index, prefix):

    fpenman = format_penman(penman)

    damr = amr.AMR.parse_AMR_line(fpenman)
    # stop if reading failed
    if damr is None:
        set_trace(context=30)
        print()

    # original ids
    ids = [x[1] for x in damr.get_triples()[0]]

    # prefix name as per original code
    damr.rename_node(prefix)

    return damr, ids


def vimdiff(penman1, penman2):

    def write(file_name, content):
        with open(file_name, 'w') as fid:
            fid.write(content)

    write('_tmp1', penman1)
    write('_tmp2', penman2)
    subprocess.call(['vimdiff', '_tmp1', '_tmp2'])


def get_triples(amr):

    is_attribute = get_is_atribute(amr.nodes, amr.edges)
    attributes = defaultdict(list)
    relations = defaultdict(list)
    for (source, label, target) in amr.edges:
        if is_attribute[target]:
            attributes[source].append((target, label))
        else:
            relations[source].append((target, label))

    return attributes, relations


def compute_score_ourselves(gold_penman, penman, alignments, ref=None):

    gold_amr = AMR.from_penman(gold_penman)
    amr = AMR.from_penman(penman)

    is_attribute = get_is_atribute(amr.nodes, amr.edges)

    # nodes + edges + root
    # gold
    gold_vars = [n for n in gold_amr.nodes if n in alignments]
    gold_attributes = [e for e in gold_amr.edges if e[2] not in alignments]
    gold_relations = [e for e in gold_amr.edges if e[2] in alignments]
    # decoded
    dec_vars = [n for n in amr.nodes if not is_attribute[n]]
    dec_attributes = [e for e in amr.edges if is_attribute[e[2]]]
    dec_relations = [e for e in amr.edges if not is_attribute[e[2]]]
    # add root to attributes
    gold_triple_num2 = \
        len(gold_vars) + len(gold_attributes) + len(gold_relations) + 1
    test_triple_num2 = \
        len(dec_vars) + len(dec_attributes) + len(dec_relations) + 1

    # node hits
    node_hits = 0
    for gid, nid in alignments.items():

        # nodes (instances differ)
        if gold_amr.nodes[gid] != amr.nodes[nid]:
            set_trace(context=30)
            print()
        else:
            node_hits += 1

    # relations / attribute hits
    relation_hits = 0
    attribute_hits = 0
    # keep a list from which we can remove nodes to avoid double count. Also
    # replace attriburte ids by node names, as node ids for attributes is juts
    # afeature of our internal format
    dec_edges = [
        (s, l, normalize(amr.nodes[t])) if is_attribute[t] else (s, l, t)
        for (s, l, t) in amr.edges
    ]
    for gid, nid in alignments.items():
        for gtgt, label in gold_amr.children(gid):

            # attributes are not aligned, but names should match
            # if gold_is_attribute[gtgt]:  # this is an imperfect heuristic
            if gtgt not in alignments:
                child_id = normalize(gold_amr.nodes[gtgt])
            else:
                child_id = alignments[gtgt]

            # edeges may be also inverted
            direct_edge = (nid, label, child_id)
            if label.endswith('-of'):
                inverse_edge = (child_id, f'{label[:-3]}', nid)
            else:
                inverse_edge = (child_id, f'{label}-of', nid)

            if direct_edge in dec_edges:
                if gtgt not in alignments:
                    attribute_hits += 1
                else:
                    relation_hits += 1
                # remove to avoid double count
                # FIXME: Should we also remove the inverse or count by separate
                dec_edges.remove(direct_edge)

            elif inverse_edge in dec_edges:
                if gtgt not in alignments:
                    attribute_hits += 1
                else:
                    relation_hits += 1
                # remove to avoid double count
                dec_edges.remove(inverse_edge)

            else:
                # missing relation
                set_trace(context=30)
                print()

    # add root as extra attribute
    if gold_amr.root == amr.root:
        attribute_hits += 1

    best_match_num2 = node_hits + attribute_hits + relation_hits

    if ref:

        best_match_num, triples2, triples1 = ref
        instance2, attributes2, relation2 = triples2
        instance1, attributes1, relation1 = triples1

        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)

        # our counts differ
        if best_match_num2 != best_match_num:
            set_trace(context=30)
            print()
        if test_triple_num2 != test_triple_num:
            set_trace(context=30)
            print()
        if gold_triple_num2 != gold_triple_num:
            set_trace(context=30)
            print()

    return best_match_num2, test_triple_num2, gold_triple_num2


def main(args):

    # read files
    gold_penmans = read_blocks(args.in_reference_amr, return_tqdm=False)
    penmans = read_blocks(args.in_amr, return_tqdm=False)

    # set global in module
    smatch.smatch.iteration_num = args.r + 1

    assert len(gold_penmans) == len(penmans)

    statistics = []
    sentence_smatch = []
    node_id_maps = []
    for index in tqdm(range(len(penmans))):

        gold_penman = gold_penmans[index]
        penman = penmans[index]

        # FIXME:
        # Seems not to be reading :mod 277703234 in dev[97]
        # read and format. Keep original ids for later reconstruction
        amr1, ids_a = protected_read(gold_penman, index, "a")
        amr2, ids_b = protected_read(penman, index, "b")

        # use smatch code to align nodes
        (instance1, attributes1, relation1) = amr1.get_triples()
        (instance2, attributes2, relation2) = amr2.get_triples()
        best_mapping, best_match_num = get_best_match(
            instance1, attributes1, relation1,
            instance2, attributes2, relation2,
            "a", "b"
        )
        # IMPORTANT: Reset cache
        smatch.smatch.match_triple_dict = {}
        # use smatch code to compute partial scores
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)

        # store sentence-level normalized score
        score = compute_f(best_match_num, test_triple_num, gold_triple_num)

        # sanity check valid match number
        if (
            best_match_num > gold_triple_num
            or best_match_num > test_triple_num
        ):
            print(yellow_font(
                f'WARNING: Sentence {index} has Smatch above 100% '
                f'({best_match_num}, {test_triple_num}, {gold_triple_num})'
            ))

        # compute alignments using the original ids
        # {id_amr1 (gold): id_amr2 (decoded)}
        sorted_ids_b = [ids_b[i] for i in best_mapping]
        best_id_map = dict(zip(ids_a, sorted_ids_b))

#         # compute scores ourselves and compare against smatch computation
#         ref = (
#             best_match_num,
#             (instance2, attributes2, relation2),
#             (instance1, attributes1, relation1)
#         )
#         num_hits, num_guess, num_gold = compute_score_ourselves(
#             gold_penman, penman, best_id_map, None
#         )

        # stop if score is not perfect
        if (
            args.stop_if_different
            and (
                best_match_num < gold_triple_num
                or test_triple_num < gold_triple_num
            )
        ):
            print(gold_penman)
            print(penman)
            vimdiff(gold_penman, penman)

        # accumulate statistics for corpus-level normalization
        statistics.append([best_match_num, test_triple_num, gold_triple_num])
        sentence_smatch.append(score)
        node_id_maps.append(best_id_map)

    # TODO: Save scores and alignments

    # final score
    best_match_num = sum([t[0] for t in statistics])
    test_triple_num = sum([t[1] for t in statistics])
    gold_triple_num = sum([t[2] for t in statistics])
    corpus_score = compute_f(best_match_num, test_triple_num, gold_triple_num)
    print(f'Smatch: {corpus_score[2]:.4f}')


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
        default=10
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
