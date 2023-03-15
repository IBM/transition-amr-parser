import sys
from glob import glob
import os
import ipdb
import traceback
from smatch import compute_f
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    # post-mortem debugger
    ipdb.pm()


# sys.excepthook = debughook


def read_triples(path, strip_ids=False):

    def strip_id(triple, n):
        if triple[1] == 'instance':
            return n.split('/')[1]
        elif triple[1] == 'attribute':
            return ' '.join(n.split(' ')[1:])
        else:
            return ''.join(n.split(' ')[1])

    with open(path) as fid:
        data = []
        fid.readline()
        for line in fid:
            items = line.strip().split('\t')
            if strip_ids:
                gold = None if items[2] == '' else strip_id(items, items[2])
                dec = None if items[3] == '' else strip_id(items, items[3])
            else:
                gold = None if items[2] == '' else items[2]
                dec = None if items[3] == '' else items[3]

            data.append([
                int(items[0]),
                items[1],
                gold,
                dec,
                int(items[4]),
            ])
    return data


def get_smatch(data):

    best_match_num = 0
    test_triple_num = 0
    gold_triple_num = 0
    for (i, t, g, d, s) in data:
        if g is None:
            test_triple_num += 1
        elif d is None:
            gold_triple_num += 1
        else:
            test_triple_num += 1
            gold_triple_num += 1
        best_match_num += s

    return compute_f(best_match_num, test_triple_num, gold_triple_num)


def compare_triples(v052_data, v060_data):
    v052_not_del = [t for t in v052_data if t[2] is not None]
    v052_del = [t for t in v052_data if t[2] is None]
    v060_not_del = [t for t in v060_data if t[2] is not None]
    v060_del = [t for t in v060_data if t[2] is None]
    assert len(v052_not_del) == len(v060_not_del)

    loose_triples = []
    win_triples = []
    match_triples = []
    for v052, v060 in zip(v052_not_del, v060_not_del):
        if v052[-1] > v060[-1]:
            loose_triples.append((v052, v060))
        elif v052[-1] < v060[-1]:
            win_triples.append((v052, v060))
        else:
            match_triples.append((v052, v060))

    loose_counts = Counter()
    for (v052, v060) in loose_triples:
        i, t, g, d, s = v052
        i2, t2, g2, d2, s2 = v060
        if t == 'instance':
            loose_counts.update([(
                g.split('/')[1],
                None if d is None else d.split('/')[1],
                None if d2 is None else d2.split('/')[1]
            )])
        elif t == 'attribute':
            loose_counts.update([(
                ' '.join(g.split(' ')[1:]),
                None if d is None else ' '.join(d.split(' ')[1:]),
                None if d2 is None else ' '.join(d2.split(' ')[1:])
            )])
        elif t == 'relation':
            loose_counts.update([(
                ''.join(g.split(' ')[1]),
                None if d is None else ''.join(d.split(' ')[1]),
                None if d2 is None else ''.join(d2.split(' ')[1])
            )])
        else:
            raise Exception()

    from ipdb import set_trace; set_trace(context=30)
    print()

    loose_counts = Counter([(v052[2], v052[3], v060[3]) for (v052, v060) in loose_triples])
    win_counts = Counter([(v052[2], v052[3], v060[3]) for (v052, v060) in loose_triples])
    match_counts = Counter([(v052[2], v052[3], v060[3]) for (v052, v060) in loose_triples])

    from ipdb import set_trace; set_trace(context=30)
    print()


def get_triples_by_epoch(system):

    triples_by_epoch = defaultdict(list)
    for (idx, dtype, gold, decoded, score) in system:
        triples_by_epoch[idx].append((idx, dtype, gold, decoded, score))

    return triples_by_epoch


def get_scores(system):

    triples_by_sent = get_triples_by_epoch(system)
    scores = []
    for s_idx, s_triples in triples_by_sent.items():
        num_insertions = len([t for t in s_triples if t[2] is None])
        rest_triples = [t for t in s_triples if t[2] is not None]

        scores.append(
            (
                sum([t[-1] for t in rest_triples]),
                sum([t[-1] for t in rest_triples]) - num_insertions,
                (sum([t[-1] for t in rest_triples])  - num_insertions) / len(rest_triples),
                get_smatch(s_triples)[-1],
                len(rest_triples),
            )
        )

    scores = sorted(scores, key=lambda x: x[-1])

    return zip(*scores)


def get_triple_type_scores(system):

    triples_by_type = defaultdict(list)
    for (i, t, g, d, s) in system:
        if g is not None:
            triples_by_type[g].append((i, t, g, d, s))

    scores = []
    for ttype, s_triples in triples_by_type.items():
        num_insertions = len([t for t in s_triples if t[2] is None])
        rest_triples = [t for t in s_triples if t[2] is not None]

        scores.append(
            (
                sum([t[-1] for t in rest_triples]),
                sum([t[-1] for t in rest_triples]) - num_insertions,
                (sum([t[-1] for t in rest_triples])  - num_insertions) / len(rest_triples),
                get_smatch(s_triples)[-1],
                len(rest_triples),
            )
        )

    scores = sorted(scores, key=lambda x: x[-1])

    return zip(*scores)


def get_scores_files(paths, per_triple_type=False):

    results = dict()
    for path in paths:
        label = os.path.basename(path)[:-4]
        triples = read_triples(path, strip_ids=bool(per_triple_type))

        if per_triple_type:
            hits, ter, rel_ter, ssmatch, gold = get_triple_type_scores(triples)
            smatch = get_smatch(triples)[-1]
            results[label] = dict(hits=hits, ter=ter, rel_ter=rel_ter, ssmatch=ssmatch, smatch=smatch, gold=gold)
            print(f'{label} {100*smatch:.2f}')

        else:

            hits, ter, rel_ter, ssmatch, gold = get_scores(triples)
            smatch = get_smatch(triples)[-1]
            results[label] = dict(hits=hits, ter=ter, rel_ter=rel_ter, ssmatch=ssmatch, smatch=smatch, gold=gold)
            print(f'{label} {100*smatch:.2f}')

    return results


def epochs_plot():

    paths = sorted(glob('DATA/triples/v0.6.0beta-epoch*.tsv'), key=lambda x: int(x.replace('DATA/triples/v0.6.0beta-epoch', '')[:-4]))

    results = get_scores_files(paths, per_triple_type=True)

#     triples_by_sentence1 = get_triples_by_epoch(system1)
#     triples_by_sentence2 = get_triples_by_epoch(system2)
#     assert len(triples_by_sentence1) == len(triples_by_sentence2)
#     for n in range(max(triples_by_sentence1)):
#         aa = sorted([(t, g) for (t, g, d, s) in triples_by_sentence1[n] if g is not None])
#         bb = sorted([(t, g) for (t, g, d, s) in triples_by_sentence2[n] if g is not None])
#         if aa != bb:
#             from ipdb import set_trace; set_trace(context=30) # noqa
#             print()

    #plt.subplot(1, 2, 1)
    #plt.plot([min(gold1), max(gold1)], [min(gold1), max(gold1)])
    #plt.plot(gold1, hits1, 'x')
    #plt.plot(gold1, hits2, 'x')
    #plt.subplot(1, 2, 2)
    # compare_triples(list(v052_data), list(v060_data))

    sorted_labels = sorted(results.keys(), key=lambda x: int(x.replace('v0.6.0beta-epoch', '')))

    num_pairs = len(sorted_labels) - 1
    metric = 'ter'
    for n in range(num_pairs):
        plt.subplot(2, num_pairs // 2, n + 1)
        hits1 = results[sorted_labels[n]][metric]
        hits2 = results[sorted_labels[-1]][metric]
        plt.plot(hits1, hits2, 'o')
        plt.xlabel(f'{sorted_labels[n]} {metric}')
        plt.ylabel(f'{sorted_labels[-1]} {metric}')
        plt.plot([0, max(hits1)], [0, max(hits1)])
    plt.show()

    from ipdb import set_trace; set_trace(context=30) # noqa
    print()


def systems_plot(per_triple_type=False):

    # paths = ['DATA/triples/v0.5.2-epoch25.tsv', 'DATA/triples/v0.6.0beta-epoch41.tsv']
    paths = ['DATA/triples/v0.5.2-beam10.tsv', 'DATA/triples/v0.6.0beta-beam10.tsv']
    results = get_scores_files(paths, per_triple_type=per_triple_type)

    metric = 'ter'
    label1 = os.path.basename(paths[0])[:-4]
    label2 = os.path.basename(paths[1])[:-4]
    hits1 = results[label1][metric]
    gold = results[label1]['gold']
    smatch1 = results[label1]['smatch']
    smatch2 = results[label2]['smatch']
    num_gold = sum(results[label1]['gold'])
    gap = int((smatch2 - smatch1) * num_gold)

    plt.subplot(1,3,1)
    hits2 = results[label2][metric]
    plt.plot(hits1, hits2, 'o')
    plt.xlabel(f'{label1} {metric} (Smatch {smatch1:.3f})')
    plt.ylabel(f'{label2} {metric} (Smatch {smatch2:.3f})')
    plt.plot([0, max(hits1)], [0, max(hits1)])

    if per_triple_type:
        plt.title(f'dot = sentence, {num_gold} triples (gap ~{gap})')
    else:
        plt.title(f'dot = triple type {num_gold} triples (gap ~{gap})')

    plt.subplot(1,3,2)
    hits2 = results[label2][metric]
    plt.plot(hits1, gold, 'o')
    plt.xlabel(f'{label1} {metric} (Smatch {smatch1:.3f})')
    plt.ylabel(f'gold')
    plt.plot([0, max(gold)], [0, max(gold)])
    plt.title(f'system compared with gold')

    plt.subplot(1,3,3)
    hits2 = results[label2][metric]
    plt.plot(hits2, gold, 'o')
    plt.xlabel(f'{label2} {metric} (Smatch {smatch2:.3f})')
    plt.ylabel(f'gold')
    plt.plot([0, max(gold)], [0, max(gold)])
    plt.title(f'system compared with gold')
    plt.show()


if __name__ == '__main__':
    # epochs_plot()
    systems_plot(per_triple_type=True)
