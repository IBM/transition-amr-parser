import collections
import json
import os

import numpy as np

from transition_amr_parser.io import read_amr

from tqdm import tqdm

from evaluation import EvalAlignments


top_names = (
'date-entity',
'have-org-role-91',
'possible-01',
'cause-01',
'thing',
'you',
'i',
'-',
'country',
'and',
'person',
'name',
)


class xEvalAlignments(object):
    def run(self, path_gold, path_pred):
        gold = read_amr(path_gold).amrs
        pred = read_amr(path_pred).amrs

        assert len(gold) == len(pred)

        metrics = collections.defaultdict(list)

        for g, p in zip(gold, pred):
            galign, palign = g.alignments, p.alignments

            total = 0
            correct = 0
            total_2 = 0
            correct_2 = 0
            for node_id in galign.keys():
                if node_id not in palign:
                    print('pred')
                    print(p.toJAMRString())
                    print('gold')
                    print(g.toJAMRString())
                    import ipdb; ipdb.set_trace()

                pa = palign[node_id]
                ga = galign[node_id]

                total += 1

                if pa == ga:
                    correct += 1

                if g.nodes[node_id] not in ('country', 'and', 'person', 'name'):
                    total_2 += 1

                    if pa == ga:
                        correct_2 += 1

            if total == 0:
                continue

            metrics['recall'].append(correct / float(total))
            metrics['correct'].append(correct)
            metrics['total'].append(total)
            metrics['correct_2'].append(correct_2)
            metrics['total_2'].append(total_2)

        correct = np.sum(metrics['correct'])
        total = np.sum(metrics['total'])
        recall = np.mean(metrics['recall'])
        recall_overall = np.sum(metrics['correct']) / np.sum(metrics['total'])

        correct_2 = np.sum(metrics['correct_2'])
        total_2 = np.sum(metrics['total_2'])
        recall_overall_2 = np.sum(metrics['correct_2']) / np.sum(metrics['total_2'])
        n = len(metrics['recall'])

        print('EVAL')
        print('gold = {}'.format(path_gold))
        print('pred = {}'.format(path_pred))
        print('n = {} ({})'.format(n, len(gold)))
        print('recall = {}'.format(recall))
        print('recall_overall = {} ( {} / {} )'.format(recall_overall, correct, total))
        print('recall_overall_2 = {} ( {} / {} )'.format(recall_overall_2, correct_2, total_2))
        print('')

        output = {}
        output['recall'] = recall

        return output


class AlignmentsReport:
    def __init__(self):
        pass

    def run(self, path_gold, path_pred, path_out):
        fout = open(path_out, 'w')
        gold = read_amr(path_gold).amrs
        pred = read_amr(path_pred).amrs
        c = collections.Counter()
        c_total = collections.Counter()
        r = collections.Counter()

        assert len(gold) == len(pred)

        def header(g, p, i):
            fout.write('\n' + '@' * 80 + '\n\n')
            fout.write('{}\n'.format(i))
            fout.write(g.toJAMRString())
            fout.write(' '.join(['{}[{}]'.format(x, ii) for ii, x in enumerate(g.tokens)]) + '\n')

        def footer(g, p):
            fout.write('\n' + '@' * 80 + '\n\n')

        def recall(g, p):
            galign, palign = g.alignments, p.alignments

            total = 0
            correct = 0
            for node_id in galign.keys():

                pa = palign[node_id][0] - 1
                ga = galign[node_id][0] - 1

                total += 1

                if pa == ga:
                    correct += 1

            if total == 0:
                return 1
            return correct / total

        def metrics(g, p):
            m = collections.OrderedDict()
            m['length'] = len(g.tokens)
            m['n-nodes'] = len(g.nodes)
            m['dup-nodes'] = len(g.nodes) - len(set(g.nodes.values()))
            m['dup-text'] = len(g.tokens) - len(set(g.tokens))
            m['recall'] = recall(g, p)

            for k, v in m.items():
                fout.write('{} {}\n'.format(k, v))
            fout.write('\n')

        def errors(g, p):
            galign, palign = g.alignments, p.alignments

            keys = ['node_id', 'node_name', 'gold', 'pred']

            fout.write(' '.join(keys) + '\n')

            for node_id in galign.keys():
                node_name = g.nodes[node_id]

                pa = palign[node_id][0] - 1
                ga = galign[node_id][0] - 1

                o = collections.OrderedDict()
                o['node_id'] = node_id
                o['node_name'] = node_name
                o['gold'] = g.tokens[ga] + '[{}]'.format(ga)
                o['pred'] = g.tokens[pa] + '[{}]'.format(pa)
                o['match'] = pa == ga

                vals = [o[k] for k in keys]

                r['total'] += 1
                if o['match'] is False:
                    r['err'] += 1
                    fout.write(' '.join(vals) + '\n')

                    if g.tokens.count(g.tokens[ga]) >= 2:
                        c['dup'] += 1
                    if node_name in top_names:
                        c['top'] += 1
                    if node_name.startswith('"'):
                        c['ent'] += 1
                    c[node_name] += 1

                if g.tokens.count(g.tokens[ga]) >= 2:
                    c_total['dup'] += 1
                if node_name in top_names:
                    c_total['top'] += 1
                if node_name.startswith('"'):
                    c_total['ent'] += 1
                c_total[node_name] += 1


        i = 0
        for g, p in tqdm(zip(gold, pred)):
            header(g, p, i)
            metrics(g, p)
            errors(g, p)
            footer(g, p)
            i += 1

        for k, v in sorted(c_total.items(), key=lambda x: x[1]):
            vv = c[k]
            fout.write('{} {} / {}\n'.format(k, vv, v))

        fout.write('{} / {}\n'.format(r['err'], r['total']))

        fout.close()

        EvalAlignments().run(args.gold, args.pred)



if __name__ == '__main__':

    import argparse

    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_0.1_exp_50_seed_0' # model a tree-rnn
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_0.1_exp_39_seed_0' # model a
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_0.1_exp_33_seed_0' # model 0
    #pred = os.path.join(base_path, 'alignment.epoch_0.val.out.pred')
    #gold = os.path.join(base_path, 'alignment.epoch_0.val.out.gold')
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/model_0_write'
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/model_a_write'
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/count_based'
    
    #base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_20210624a_exp_17_seed_0_write'
    base_path = '/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_20210624a_exp_22_seed_0_write'
    pred = os.path.join(base_path, 'alignment.trn.out.pred')
    gold = os.path.join(base_path, 'alignment.trn.out.gold')
    out = os.path.join(base_path, 'alignment.trn.out.report')

    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=None, type=str)
    parser.add_argument('--gold', default=gold, type=str)
    parser.add_argument('--pred', default=pred, type=str)
    parser.add_argument('--out', default=out, type=str)
    args = parser.parse_args()

    if args.base is not None:
        args.pred = os.path.join(args.base, 'alignment.trn.out.pred')
        args.gold = os.path.join(args.base, 'alignment.trn.out.gold')
        args.out = os.path.join(args.base, 'alignment.trn.out.report')

    print(json.dumps(args.__dict__, sort_keys = True, indent = 4))

    report = AlignmentsReport()
    report.run(args.gold, args.pred, args.out)

