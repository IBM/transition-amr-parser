import numpy as np
import argparse
import re
import subprocess
from collections import defaultdict
from transition_amr_parser.io import read_blocks, read_penmans
from transition_amr_parser.amr import (
    get_is_atribute, smatch_triples_from_penman
)
from transition_amr_parser.clbar import yellow_font
from smatch import get_best_match, compute_f
from smatch import amr
import smatch
import penman


def cltable(rows):
    # sanity checks
    assert isinstance(rows, list)
    assert all(isinstance(row, list) for row in rows)
    assert len(set(len(row) for row in rows)) == 1
    # formatting
    num_cols = len(rows[0])
    widths = [max(len(r[n]) for r in rows) for n in range(num_cols)]
    separator = '  '
    # get table string
    string = ''
    for row in rows:
        for n in range(num_cols):
            if n > 0:
                string += separator
            string += f'{row[n]:<{widths[n]}}'
        string += '\n'
    return string


def bootstrap_paired_is_greater_test(scorer, counts1, counts2, restarts=10000):
    '''

    Implements paired boostrap significance test after

    @Book{Nor89,
        author = {E. W. Noreen},
        title =  {Computer-Intensive Methods for Testing Hypotheses},
        publisher = {John Wiley Sons},
        year = {1989},
    }

    SCORER e.g. F1 scores given the sum of statistics for each example in the
    test set. For example F1 as used in smatch takes arg_number=3:
    (num_hits, num_predicted, num_gold)

    Normal score computation would be

        SCORE1 = SCORER(*list(COUNTS1.sum(0)))
        SCORE2 = SCORER(*list(COUNTS2.sum(0)))

    COUNTS1, COUNTS2 are numpy arrays of shape (arg_number, example_number)
    corresponding to the statistics for each example of each system

    It tests the hypothesis SCORE1 > SCORE2

    The test randomly swaps examples between both sets of counts and sees if
    this changes the previous bigger than relation (e.g. SCORE1 > SCORE2)
    '''

    assert len(counts1) == len(counts2)

    # ensure theyr are arrays
    counts1 = np.array(counts1)
    counts2 = np.array(counts2)
    num_examples = counts1.shape[0]

    # reference score, the second one is assumed to be the smaller and the
    # baseline to beat
    reference_score = scorer(*list(counts2.sum(0)))[2]

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
        raise Exception('Smatch AMR reader failed')

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

    def plot_bootstrap_test_score_differences(self, hypotheses):

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        num_cols = min(4, len(hypotheses))
        num_rows = np.ceil(len(hypotheses) / num_cols)
        index = 0
        nbins = 100
        for hypothesis in hypotheses:
            i, j = hypothesis['indices']
            plt.subplot(int(num_rows), int(num_cols), index + 1)
            plt.hist(hypothesis['score_differences'], bins=nbins)
            plt.plot([0, 0], [0, 50], 'r--')
            if hypothesis['is_greater']:
                rel = '>'
            else:
                rel = '<'
            if self.amr_labels:
                title = f'{self.amr_labels[i]} {rel} {self.amr_labels[j]}'
            else:
                title = f'{i} {rel} {j}'
            plt.title(title)
            index += 1
        print(f'wrote {self.out_boostrap_png}')
        plt.savefig(self.out_boostrap_png)

    def bootstrap_test_all_pairs(self):

        num_amrs = len(self.statistics[0])

        # run test for every pair ignoring the reverse pair
        hypotheses = []
        for i in range(num_amrs):
            for j in range(i+1, num_amrs):

                # gather statistics for each AMR
                stats1 = np.array([(
                    amrs[i]['best_match_num'],
                    amrs[i]['test_triple_num'],
                    amrs[i]['gold_triple_num']
                ) for amrs in self.statistics])
                stats2 = np.array([(
                    amrs[j]['best_match_num'],
                    amrs[j]['test_triple_num'],
                    amrs[j]['gold_triple_num']
                ) for amrs in self.statistics])

                # compute initial scores
                score1 = compute_f(*stats1.sum(0))
                score2 = compute_f(*stats2.sum(0))

                # hypotheses first is greater than second
                if score1 > score2:
                    p_value, delta = bootstrap_paired_is_greater_test(
                        compute_f, stats1, stats2,
                        restarts=self.bootstrap_test_restarts
                    )
                    hypotheses.append({
                        'indices': (i, j),
                        'is_greater': True,
                        'p_value': p_value,
                        'score_differences': delta
                    })

                else:
                    p_value, delta = bootstrap_paired_is_greater_test(
                        compute_f, stats2, stats1,
                        restarts=self.bootstrap_test_restarts
                    )
                    hypotheses.append({
                        'indices': (i, j),
                        'is_greater': False,
                        'p_value': p_value,
                        'score_differences': delta
                    })

        if self.out_boostrap_png:
            self.plot_bootstrap_test_score_differences(hypotheses)

        return hypotheses

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
        # padded display
        string = cltable(rows)

        if self.bootstrap_test:
            hypotheses = self.bootstrap_test_all_pairs()
            rows = []
            for hypothesis in hypotheses:
                i, j = hypothesis['indices']
                pv = hypothesis['p_value']
                if hypothesis['is_greater']:
                    rel = '>'
                else:
                    rel = '<'
                if hypothesis['p_value'] < 0.05:
                    sig = 'significant'
                else:
                    sig = 'not significant'
                if self.amr_labels:
                    label_i = self.amr_labels[i]
                    label_j = self.amr_labels[j]
                    rows.append(
                        [f'{label_i} {rel} {label_j}', f'{pv:.3f}', sig]
                    )
                else:
                    rows.append([f'{i} {rel} {j}', f'{pv:.3f}', sig])

            string += '\nboostrap paired randomized test\n'
            string += cltable(rows)

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
