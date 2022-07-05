import numpy as np
import argparse
import re
import subprocess
from collections import defaultdict
from ipdb import set_trace
from transition_amr_parser.io import read_blocks, read_penmans
from transition_amr_parser.amr import (
    AMR, get_is_atribute, normalize, smatch_triples_from_penman
)
from transition_amr_parser.clbar import yellow_font
from smatch import get_best_match, compute_f
from smatch import amr
import smatch
import penman


def run_bootstrap_paired_test(scorer, counts1, counts2, restarts=1000):
    '''
    counts are a lists of lists of counts. Outer list is examples, inner
    example counts

    scorer takes the sum of counts (as many items as inner list len())
    '''

    assert len(counts1) == len(counts2)

    counts1 = np.array(counts1)
    counts2 = np.array(counts2)
    num_examples = counts1.shape[0]

    # reference score
    score1 = scorer(*list(counts1.sum(0)))[2]
    score2 = scorer(*list(counts2.sum(0)))[2]

    if score1 < score2:
        reference_score = score1
    else:
        reference_score = score2

    # scores for random swaps
    better = 0
    score_differences = np.zeros(restarts)
    for i in range(restarts):

        # score of random paired swap
        swap = (np.random.randn(num_examples) > 0).astype(float)[:, None]
        swapped_counts = swap * counts1 + (1 - swap) * counts2
        swapped_score = scorer(*list(swapped_counts.sum(0)))[2]

        # assign a point if the worse model gets better by mixing
        if swapped_score > reference_score:
            better += 1
        score_differences[i] = swapped_score - reference_score

    p = 1 - (better * 1.0 / restarts)

    return p, score_differences


def original_triples(penman, index, prefix):

    def format_penman(x):
        # remove comments
        lines = '\n'.join([x for x in x.split('\n') if x and x[0] != '#'])
        # remove ISI alignments if any
        lines = re.sub(r'\~[0-9,]+', '', lines)
        return lines.replace('\n', '')

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

    # instances, attributes and relations
    instance, attributes, relation = damr.get_triples()

    return instance, attributes, relation, ids


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


class Stats():

    def __init__(self, amr_labels, bootstrap_test, bootstrap_test_restarts,
                 out_boostrap_png):

        self.amr_labels = amr_labels
        self.bootstrap_test = bootstrap_test
        self.bootstrap_test_restarts = bootstrap_test_restarts
        self.out_boostrap_png = out_boostrap_png
        # will hold a list of dictionaries with individual and cross AMR stats
        # keys are amr_labels or just integers
        self.statistics = []

    def update(self, sent_idx, amr_idx, best_mapping, best_match_num, items_a,
               items_b):

        instance1, attributes1, relation1, ids_a = items_a
        instance2, attributes2, relation2, ids_b = items_b

        # compute alignments using the original ids
        sorted_ids_b = [ids_b[i] if i != -1 else i for i in best_mapping]
        best_id_map = dict(zip(ids_a, sorted_ids_b))

        # use smatch code to compute partial scores
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)

        # data structure with statistics
        amr_statistics = dict(
            best_id_map=best_id_map, best_match_num=best_match_num,
            test_triple_num=test_triple_num, gold_triple_num=gold_triple_num
        )

        if len(self.statistics) - 1 == sent_idx:
            # append to existing sentence
            assert amr_idx not in self.statistics[sent_idx]
            self.statistics[sent_idx][amr_idx] = amr_statistics
        else:
            # start new sentence
            self.statistics.append({amr_idx: amr_statistics})

        # sanity check valid match number
        if (
            best_match_num > gold_triple_num
            or best_match_num > test_triple_num
        ):
            print(yellow_font(
                f'WARNING: Sentence {sent_idx} has Smatch above 100% '
                f'({best_match_num}, {test_triple_num}, {gold_triple_num})'
            ))

    def bootstrap_test_all_pairs(self):

        num_amrs = len(self.statistics[0])

        # get all AMR pairings and run a test for each
        pairs = []
        for i in range(num_amrs):
            for j in range(num_amrs):
                if j > i:
                    pairs.append((i, j))

        # run over every pair
        p_value = {}
        delta = {}
        for i, j in pairs:
            p_value[(i, j)], delta[(i, j)] = run_bootstrap_paired_test(
                # F-measure
                compute_f,
                # counts
                [(
                    amrs[i]['best_match_num'],
                    amrs[i]['test_triple_num'],
                    amrs[i]['gold_triple_num']
                ) for amrs in self.statistics],
                [(
                    amrs[j]['best_match_num'],
                    amrs[j]['test_triple_num'],
                    amrs[j]['gold_triple_num']
                ) for amrs in self.statistics],
                # number of restarts
                restarts=self.bootstrap_test_restarts
            )

        if self.out_boostrap_png:

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            num_cols = min(4, len(delta.keys()))
            num_rows = np.ceil(len(delta.keys()) / num_cols)
            index = 0
            nbins = 100
            for (i, j) in delta.keys():
                plt.subplot(int(num_rows), int(num_cols), index + 1)
                plt.hist(delta[(i, j)], bins=nbins)
                plt.plot([0, 0], [0, 50], 'r--')
                if self.amr_labels:
                    plt.title(f'({self.amr_labels[i]}, {self.amr_labels[j]})')
                else:
                    plt.title(f'({i}, {j})')
                index += 1
            print(f'wrote {self.out_boostrap_png}')
            plt.savefig(self.out_boostrap_png)

        return p_value

    def compute_corpus_scores(self):

        # TODO: Save scores and alignments
        # store sentence-level normalized score
        # score = compute_f(best_match_num, test_triple_num, gold_triple_num)

        # cumulative score
        num_amrs = len(self.statistics[0])
        counts = [[0, 0, 0] for _ in range(num_amrs)]
        for amrs in self.statistics:
            for amr_idx, stats in amrs.items():
                counts[amr_idx][0] += stats['best_match_num']
                counts[amr_idx][1] += stats['test_triple_num']
                counts[amr_idx][2] += stats['gold_triple_num']

        return counts, [compute_f(*amr_counts) for amr_counts in counts]

    def __str__(self):

        # compute aggregate stats and corpus-level normalization
        counts, corpus_scores = self.compute_corpus_scores()

        num_amrs = len(self.statistics[0])
        if self.amr_labels:
            labels = self.amr_labels
        else:
            labels = [i for i in range(num_amrs)]

        # score strings
        rows = [
            ['model', '#hits', '#gold', '#tries',  'P',   'R',  'Smatch']
        ]
        for i in range(num_amrs):
            p, r, smatch = corpus_scores[i]
            rows.append([
                f'{labels[i]}', f'{counts[i][0]}', f'{counts[i][1]}',
                f'{counts[i][2]}', f'{p:.3f}', f'{r:.3f}', f'{smatch:.3f}'
            ])
        # widths
        widths = [max(len(r[n]) for r in rows) for n in range(len(rows[0]))]
        # padded display
        string = ''
        for i, row in enumerate(rows):
            padded_row = []
            for n, col in enumerate(row):
                if n == 0:
                    padded_row.append(f'{col:<{widths[n]}}')
                else:
                    padded_row.append(f'{col:^{widths[n]}}')
            string += ' '.join(padded_row) + '\n'

        if self.bootstrap_test:
            p_value = self.bootstrap_test_all_pairs()
            string += '\nboostrap paired randomized test\n'
            for (i, j), p in p_value.items():
                if self.amr_labels:
                    label_i = self.amr_labels[i]
                    label_j = self.amr_labels[j]
                    string += f'({label_i}, {label_j}) {p:.3f}\n'
                else:
                    string += f'({i}, {j}) {p:.3f}\n'

        return string


def main(args):

    # read files
    gold_penmans = read_blocks(args.in_reference_amr, return_tqdm=False)
    corpus_penmans = read_penmans(args.in_amrs)
    assert len(gold_penmans) == len(corpus_penmans)

    # initialize class storing stats
    stats = Stats(
        args.amr_labels,
        args.bootstrap_test,
        args.bootstrap_test_restarts,
        args.out_boostrap_png
    )

    # set global in module
    smatch.iteration_num = args.r + 1

    # loop over each reference sentence and one or more decoded AMRs
    for sent_index, dec_penmans in enumerate(corpus_penmans):

        # reference
        gold_penman = gold_penmans[sent_index]

        if args.penman_reader:

            # get triples from goodmami's penman
            gold_graph = penman.decode(gold_penman.split('\n'))
            items_a = smatch_triples_from_penman(gold_graph, "a")

        else:

            # get triples from amr.AMR.parse_AMR_line
            # Seems not to be reading :mod 277703234 in dev[97]
            # read and format. Keep original ids for later reconstruction
            items_a = original_triples(gold_penman, sent_index, "a")

        # loop over one or more decoded AMRs for same reference
        for amr_index, dec_penman in enumerate(dec_penmans):

            if args.penman_reader:

                # get triples from goodmami's penman
                dec_graph = penman.decode(dec_penman.split('\n'))
                items_b = smatch_triples_from_penman(dec_graph, "b")

            else:

                # get triples from amr.AMR.parse_AMR_line
                # Seems not to be reading :mod 277703234 in dev[97]
                # read and format. Keep original ids for later reconstruction
                items_b = original_triples(dec_penman, sent_index, "b")

            # compute scores and update stats
            instance1, attributes1, relation1, ids_a = items_a
            instance2, attributes2, relation2, ids_b = items_b

            # align triples using Smatch's solllution
            best_mapping, best_match_num = get_best_match(
                instance1, attributes1, relation1,
                instance2, attributes2, relation2,
                "a", "b"
            )
            # IMPORTANT: Reset cache
            smatch.match_triple_dict = {}

            # update stats
            stats.update(
                sent_index, amr_index, best_mapping, best_match_num,
                items_a, items_b
            )

    print(stats)


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    parser.add_argument(
        "--in-reference-amr",
        help="file with reference AMRs in penman format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--in-amrs",
        help="one or more files with AMRs in penman format to evaluate",
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        "--amr-labels",
        help="Labels for --in-amrs files",
        nargs='+',
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
        "--penman-reader",
        help="Read AMR into triples throgh goodmami's penman module rather"
             " than the smatch code",
        action='store_true'
    )
    parser.add_argument(
        "--bootstrap-test",
        help="Smatch Significance test, requires more than one AMR to eval.",
        action='store_true'
    )
    parser.add_argument(
        "--bootstrap-test-restarts",
        help="Number of re-starts in significance test",
        default=10000,
        type=int
    )
    parser.add_argument(
        "--out-boostrap-png",
        help="Plots for the boostrap test.",
        type=str
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
