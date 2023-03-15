import numpy as np
import os
import argparse
import re
import subprocess
from collections import defaultdict, Counter
from ipdb import set_trace
from transition_amr_parser.io import read_blocks, read_penmans
from transition_amr_parser.amr import (
    get_is_atribute, smatch_triples_from_penman
)
from transition_amr_parser.clbar import yellow_font
from smatch import get_best_match, compute_f
from smatch import amr
import smatch
import penman


def red(text):
    return "\033[%dm%s\033[0m" % (91, text)


def table_str(rows, sep='  ', force_str=False):

    if force_str:
        rows = [[str(cell) for cell in row] for row in rows]

    table_str = ''
    # TODO: remove bash color codes
    bash_special = re.compile(r'\\x1b\[\d+m|\\x1b\[0m')
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows)

    # right pad all rows to same length
    for i in range(num_rows):
        if len(rows[i]) < num_cols:
            rows[i] += ['' for _ in range(num_cols - len(rows[i]))]

    # get columns for each width
    widths = []
    for n in range(num_cols):
        widths.append(max([len(bash_special.sub('', r[n])) for r in rows]))

    for row in rows:
        for n, cell in enumerate(row):
            table_str += f'{sep * int(n > 0)}{cell:<{widths[n]}}'
        table_str += '\n'
    return table_str


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


def align_instances(instance1, instance2, best_mapping):

    # make decoded items indexable by node position
    i_map = {int(t[1][1:]): t for t in instance2}

    # store an aligned set of triples
    triple_pairs = []
    matched_instances = []
    for (dtype, nid, nname) in instance1:

        # there can only be one decoded instance assigned to one gold
        # instance
        idx = best_mapping[int(nid[1:])]

        if idx == -1:
            # deletion error
            triple_pairs.append((dtype, (nid, nname), None, 0))
            continue

        (dtype2, nid_b, nname_b) = i_map[idx]

        if nname_b == nname:

            # correct match
            triple_pairs.append(
                (dtype, (nid, nname), (nid_b, nname_b), 1)
            )

        else:

            # incorrect match
            triple_pairs.append(
                (dtype, (nid, nname), (nid_b, nname_b), 0)
            )

        # substitution or correct
        matched_instances.append((dtype2, nid_b, nname_b))

    # insertions
    missing = sorted(set(instance2) - set(matched_instances))
    for (dtype2, nid_b, nname_b) in missing:
        # insertion error
        triple_pairs.append((dtype, None, (nid_b, nname_b), 0))

    return triple_pairs


def align_attributes(attributes1, attributes2, best_mapping, fix1=True, fix2=True):

    # each node can have more than one attribute
    a_map = defaultdict(list)
    for t in attributes2:
        a_map[int(t[1][1:])].append(t)
    a_map = dict(a_map)

    # store an aligned set of triples
    triple_pairs = []
    remaining_attributes = list(attributes2)
    if fix1:
        repetitions = Counter(attributes2)
    else:
        repetitions = {a: 1 for a in attributes2}
    for (edge, nid, leaf) in attributes1:

        # instance
        idx = best_mapping[int(nid[1:])]

        if idx == -1 or idx not in a_map:
            # deletion error
            triple_pairs.append(
                ('attribute', (edge, nid, leaf), None, 0)
            )

        else:

            # there can only be one decoded instance assigned to one gold
            # instance
            triples = a_map[best_mapping[int(nid[1:])]]

            # more than one decoded triple assigned to this gold triple
            if len(triples) > 1:
                cmap = {
                    (aa, bb): i
                    for i, (aa, cc, bb) in enumerate(triples)
                    if (aa, cc, bb) in remaining_attributes
                }
                if (edge, leaf) in cmap:
                    index = cmap[(edge, leaf)]
                    triple_pairs.append(
                        (
                            'attribute',
                            (edge, nid, leaf),
                            triples[index],
                            repetitions[triples[index]]
                        )
                    )
                    if triples[index] not in remaining_attributes:
                        raise Exception()
                    remaining_attributes.remove(triples[index])
                else:
                    triple_pairs.append(
                        ('attribute', (edge, nid, leaf), None, 0)
                    )

            elif (
                (edge, leaf) == (triples[0][0], triples[0][2])
                # FIXME: tmp patch
                # and triples[0] in remaining_attributes
            ):
                if triples[0] in remaining_attributes:
                    triple_pairs.append(
                        (
                            'attribute',
                            (edge, nid, leaf),
                            triples[0],
                            repetitions[triples[0]]
                        )
                    )
                    remaining_attributes.remove(triples[0])
                elif fix2:
                    # TODO: This also double counts. Need to put a none since
                    # its duplicate
                    triple_pairs.append(
                        (
                            'attribute',
                            (edge, nid, leaf),
                            None,
                            repetitions[triples[0]]
                        )
                    )
                else:
                    raise Exception()

            elif triples[0] not in remaining_attributes:
                # TODO: this is an odd case, should it happen?
                triple_pairs.append(
                    ('attribute', (edge, nid, leaf), None, 0)
                )

            else:
                triple_pairs.append(
                    ('attribute', (edge, nid, leaf), triples[0], 0)
                )
                if triples[0] not in remaining_attributes:
                    raise Exception()
                remaining_attributes.remove(triples[0])

    # insertions
    for (edge, nid, leaf) in sorted(remaining_attributes):
        triple_pairs.append(
            ('attribute', None, (edge, nid, leaf), 0)
        )

    return triple_pairs


def align_relations(relation1, relation2, best_mapping, fix1=True, fix2=True):

    # each node pair can have more than one relation
    r_map = defaultdict(list)
    for t in relation2:
        r_map[(int(t[1][1:]), int(t[2][1:]))].append(t)
    r_map = dict(r_map)

    # store an aligned set of triples
    triple_pairs = []
    remaining_edges = list(relation2)
    # TODO: Why is this needed to reproduce results?. Also this makes Smatch
    # not symmetric
    if fix1:
        repetitions = Counter(relation2)
    else:
        repetitions = {r: 1 for r in relation2}
    for (edge, nid, nid2) in relation1:
        idx1 = best_mapping[int(nid[1:])]
        idx2 = best_mapping[int(nid2[1:])]
        key = (idx1, idx2)
        if key not in r_map:
            # deletions
            triple_pairs.append(
                ('relation', (edge, nid, nid2), None, 0)
            )

        else:

            # matches
            if len(r_map[key]) > 1:
                cmap = {
                    aa: i for i, (aa, bb, cc) in enumerate(r_map[key])
                    if (aa, bb, cc) in remaining_edges
                }
                if edge in cmap:
                    index = cmap[edge]
                    triple_pairs.append(
                        (
                            'relation',
                            (edge, nid, nid2),
                            r_map[key][index],
                            repetitions[r_map[key][index]]
                        )
                    )
                    if r_map[key][index] not in remaining_edges:
                        raise Exception()
                    remaining_edges.remove(r_map[key][index])
                else:
                    triple_pairs.append(
                        ('relation', (edge, nid, nid2), None, 0)
                    )

            elif edge == r_map[key][0][0]:

                if r_map[key][0] in remaining_edges:
                    triple_pairs.append(
                        (
                            'relation',
                            (edge, nid, nid2),
                            r_map[key][0],
                            repetitions[r_map[key][0]]
                        )
                    )
                    remaining_edges.remove(r_map[key][0])
                elif fix2:
                    # TODO: This also double counts. Need to put a none since
                    # its duplicate
                    triple_pairs.append(
                        (
                            'relation',
                            (edge, nid, nid2),
                            None,
                            repetitions[r_map[key][0]]
                        )
                    )
                else:
                    raise Exception()

            elif r_map[key][0] not in remaining_edges:
                # TODO: this is an odd case, should it happen?
                triple_pairs.append(
                    ('relation', (edge, nid, nid2), None, 0)
                )

            else:
                triple_pairs.append(
                    ('relation', (edge, nid, nid2), r_map[key][0], 0)
                )
                if r_map[key][0] not in remaining_edges:
                    raise Exception()
                remaining_edges.remove(r_map[key][0])

    # insertions
    for (edge, nid, nid2) in sorted(remaining_edges):
        triple_pairs.append(
            ('relation', None, (edge, nid, nid2), 0)
        )

    return triple_pairs


def triples_to_table(triples, colors=False):
    rows = []
    for (dtype, gold, dec, score) in triples:
        if dtype == 'instance':
            if gold is None:
                gold = ''
            else:
                gold = '/'.join(gold)
            if dec is None:
                dec = ''
            else:
                dec = '/'.join(dec)
            if score > 0 or not colors:
                rows.append((dtype, gold, dec, str(score)))
            else:
                rows.append((dtype, gold, dec, red(score)))

        elif dtype == 'attribute':
            if gold is None:
                gold = ''
            else:
                (s, l, t) = gold
                gold = ' '.join([s, f':{l}', t])
            if dec is None:
                dec = ''
            else:
                (s, l, t) = dec
                dec = ' '.join([s, f':{l}', t])
            if score > 0 or not colors:
                rows.append((dtype, gold, dec, str(score)))
            else:
                rows.append((dtype, gold, dec, red(score)))

        elif dtype == 'relation':
            if gold is None:
                gold = ''
            else:
                (s, l, t) = gold
                gold = ' '.join([s, f':{l}', t])
            if dec is None:
                dec = ''
            else:
                (s, l, t) = dec
                dec = ' '.join([s, f':{l}', t])
            if score > 0 or not colors:
                rows.append((dtype, gold, dec, str(score)))
            else:
                rows.append((dtype, gold, dec, red(score)))

        else:
            raise Exception()

    return rows


def sanity_check_aligned_triples(items_a, items_b, paired, best_match_num):
    '''
    Format triples into a table-like formal that can be printed or saved
    '''

    # unpack each variable
    instance1, attributes1, relation1, ids_a = items_a
    instance2, attributes2, relation2, ids_b = items_b
    ins_pairs, att_pairs, rel_pairs = paired

    # original number of triples of each is kept
    if ins_pairs:
        _, gold_ins, dec_ins, _ = zip(*ins_pairs)
    else:
        gold_ins = []
        dec_ins = []
    if att_pairs:
        _, gold_att, dec_att, _ = zip(*att_pairs)
    else:
        gold_att = []
        dec_att = []
    if rel_pairs:
        _, gold_rel, dec_rel, _ = zip(*rel_pairs)
    else:
        gold_rel = []
        dec_rel = []

    def clean(y):
        if y == []:
            return y
        elif [x for x in y if x is not None] == []:
            return []
        elif len([x for x in y if x is not None][0]) == 2:
            return sorted([('instance', x[0], x[1]) for x in y if x is not None])
        else:
            return sorted([x for x in y if x is not None])

    gold_ins = clean(gold_ins)
    gold_att = clean(gold_att)
    gold_rel = clean(gold_rel)
    dec_ins = clean(dec_ins)
    dec_att = clean(dec_att)
    dec_rel = clean(dec_rel)

    assert gold_ins == sorted(instance1)
    assert dec_ins == sorted(instance2)
    assert gold_att == sorted(attributes1)
    assert dec_att == sorted(attributes2)
    assert gold_rel == sorted(relation1)
    assert dec_rel == sorted(relation2)

    # number of hits match
    ins_hits = sum([x[-1] for x in ins_pairs])
    att_hits = sum([x[-1] for x in att_pairs])
    rel_hits = sum([x[-1] for x in rel_pairs])
    assert best_match_num == (ins_hits + att_hits + rel_hits)

    rows = triples_to_table(ins_pairs + att_pairs + rel_pairs)
    print(table_str(rows))


def remap_ids(inst_pairs, att_pairs, rel_pairs, ids_a, ids_b):

    # use the original ids
    # instances
    inst_pairs2 = []
    for (t, gold, dec, c) in inst_pairs:
        if gold is None:
            di, dn = dec
            dec = (ids_b[int(di[1:])], dn)
            gold = None

        elif dec is None:
            gi, gn = gold
            dec = None
            gold = (ids_a[int(gi[1:])], gn)

        else:
            gi, gn = gold
            di, dn = dec
            gold = (ids_a[int(gi[1:])], gn)
            dec = (ids_b[int(di[1:])], dn)
        inst_pairs2.append((t, gold, dec, c))
    inst_pairs = inst_pairs2

    # attributes
    att_pairs2 = []
    for (t, gold, dec, c) in att_pairs:
        if gold is None:
            ds, di, dt = dec
            dec = (ids_b[int(di[1:])], ds, dt)
            gold = None

        elif dec is None:
            gs, gi, gt = gold
            dec = None
            gold = (ids_a[int(gi[1:])], gs, gt)
        else:
            gs, gi, gt = gold
            ds, di, dt = dec
            dec = (ids_b[int(di[1:])], ds, dt)
            gold = (ids_a[int(gi[1:])], gs, gt)
        att_pairs2.append((t, gold, dec, c))
    att_pairs = att_pairs2

    # relations
    rel_pairs2 = []
    for (t, gold, dec, c) in rel_pairs:
        if gold is None:
            de, ds, dt = dec
            gold = None
            dec = (ids_b[int(di[1:])], de, ids_b[int(dt[1:])])

        elif dec is None:
            ge, gs, gt = gold
            gold = (ids_a[int(gs[1:])], ge, ids_a[int(gt[1:])])
            dec = None

        else:
            ge, gs, gt = gold
            de, ds, dt = dec
            gold = (ids_a[int(gs[1:])], ge, ids_a[int(gt[1:])])
            dec = (ids_b[int(ds[1:])], de, ids_b[int(dt[1:])])
        rel_pairs2.append((t, gold, dec, c))
    rel_pairs = rel_pairs2

    return inst_pairs, att_pairs, rel_pairs


def get_aligned_triples(items_a, items_b, best_mapping, best_match_num):

    # for (snt_idx, triple_type, gold_triple, dec_triple, correct) in aligned_triples:
    #    if not correct:

    # unpack each variables
    instance1, attributes1, relation1, ids_a = items_a
    instance2, attributes2, relation2, ids_b = items_b

    # create lists of aligned gold and decoded triples
    inst_pairs = align_instances(instance1, instance2, best_mapping)
    att_pairs = align_attributes(attributes1, attributes2, best_mapping)
    rel_pairs = align_relations(relation1, relation2, best_mapping)

    # check numbers match smatchs
    paired = inst_pairs, att_pairs, rel_pairs

    # sanity_check_aligned_triples(items_a, items_b, paired, best_match_num)

    inst_pairs, att_pairs, rel_pairs = \
        remap_ids(inst_pairs, att_pairs, rel_pairs, ids_a, ids_b)

    triple_pairs = inst_pairs + att_pairs + rel_pairs

    return triple_pairs


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

        # if sent_idx == 23:
        #    from ipdb import set_trace; set_trace(context=30)

        # extract aligned mapping
        triple_pairs = get_aligned_triples(
            items_a, items_b, best_mapping, best_match_num
        )

        instance1, attributes1, relation1, ids_a = items_a
        instance2, attributes2, relation2, ids_b = items_b

        # use smatch code to compute partial scores
        gold_triple_num = len(instance1) + len(attributes1) + len(relation1)
        test_triple_num = len(instance2) + len(attributes2) + len(relation2)

        # compute alignments using the original ids
        sorted_ids_b = [ids_b[i] if i != -1 else i for i in best_mapping]
        best_id_map = dict(zip(ids_a, sorted_ids_b))

        # data structure with statistics
        amr_statistics = dict(
            best_id_map=best_id_map,
            best_match_num=best_match_num,
            test_triple_num=test_triple_num,
            gold_triple_num=gold_triple_num,
            triple_pairs=triple_pairs
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

    def compute_corpus_scores_from_aligned_triples(self):
        # sanity check to compare with the function below

        num_amrs = len(self.statistics[0])
        counts = [[0, 0, 0] for _ in range(num_amrs)]
        for amrs in self.statistics:
            for amr_idx, stats in amrs.items():
                _, gold, dec, score = zip(*stats['triple_pairs'])
                counts[amr_idx][0] += sum(score)
                counts[amr_idx][1] += sum([x is not None for x in dec])
                counts[amr_idx][2] += sum([x is not None for x in gold])

        return counts, [compute_f(*amr_counts) for amr_counts in counts]

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
        # counts2, corpus_scores2 = self.compute_corpus_scores_from_aligned_triples()

        num_amrs = len(self.statistics[0])
        if self.amr_labels:
            labels = self.amr_labels
        else:
            labels = [i for i in range(num_amrs)]

        # score strings
        rows = [
            ['model', '#hits', '#tries', '#gold',  'P',   'R',  'Smatch']
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

    def save_aligned_triples(self, in_amrs, amr_labels,
                             out_aligned_triples_dir):

        header = ['sentence_index', 'type', 'gold_triple', 'dec_triple', 'score']
        for amr_index, amr_file  in enumerate(in_amrs):
            if amr_labels:
                label = amr_labels[amr_index]
            else:
                label = amr_index
            os.makedirs(out_aligned_triples_dir, exist_ok=True)
            path = f'{out_aligned_triples_dir}/{label}.tsv'
            with open(path, 'w') as fid:
                fid.write('\t'.join(header) + '\n')
                for snt_idx, amrs in enumerate(self.statistics):
                    rows = triples_to_table(amrs[amr_index]['triple_pairs'])
                    for row in rows:
                        row2 = list(row)
                        row2.insert(0, str(snt_idx))
                        fid.write('\t'.join(row2) + '\n')


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

        # if sent_index > 30: break

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

    # write triples
    if args.out_aligned_triples_dir:
        stats.save_aligned_triples(
            args.in_amrs, args.amr_labels, args.out_aligned_triples_dir
        )


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
    parser.add_argument(
        "--out-aligned-triples-dir",
        help="Stores gold-decoded aligned triples. One per file in --in-amrs",
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
