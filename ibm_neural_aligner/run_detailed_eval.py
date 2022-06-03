import argparse
import collections
import json
import os

from tqdm import tqdm

import numpy as np

from evaluation import EvalAlignments
from formatter import amr_to_pretty_format
from transition_amr_parser.io import read_amr


class CorpusRecall_WithGoldSpans_WithSomeNodes(object):

    def __init__(self):
        self.state = collections.defaultdict(list)

    def update(self, gold, pred, pred_ref):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        for node_id in gold_align.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            # Penalty for not predicting.
            if node_id not in pred_align or pred_align[node_id] is None:
                total += 1
                continue

            # Be fair.
            if node_id not in pred_ref.alignments:
                total += 1
                continue

            total += 1

            g0 = gold_align[node_id][0] - 1
            g1 = gold_align[node_id][-1] - 1

            p0 = pred_align[node_id][0] - 1
            p1 = pred_align[node_id][-1] - 1

            gset = set(range(g0, g1 + 1))
            pset = set(range(p0, p1 + 1))

            if len(set.intersection(pset, gset)) > 0:
                correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)

    def finish(self):
        total = np.sum(self.state['total']).item()
        correct = np.sum(self.state['correct']).item()
        if total:
            recall = correct / total
        else:
            recall = 0
        result = collections.OrderedDict()
        result['correct'] = correct
        result['total'] = total
        result['recall'] = recall

        return result


def main(args):
    gold = read_amr(args.gold)
    neural = read_amr(args.neural)
    cofill = read_amr(args.cofill)

    d_gold = {amr.id: amr for amr in gold}
    d_neural = {amr.id: amr for amr in neural}
    d_cofill = {amr.id: amr for amr in cofill}

    keys = [k for k in d_gold.keys() if k in d_neural and k in d_cofill]

    gold = [d_gold[k] for k in keys]
    neural = [d_neural[k] for k in keys]
    cofill = [d_cofill[k] for k in keys]

    def check_1():
        for g, p in zip(gold, neural):
            for k, v in g.nodes.items():
                assert v == p.nodes[k], (k, v, p.nodes[k], g.id)

        for g, p in zip(gold, cofill):
            for k, v in g.nodes.items():
                assert v == p.nodes[k], (k, v, p.nodes[k], g.id)

    def print_result(result, header=None):
        def format_value(val):
            if isinstance(val, float):
                return '{:.3f}'.format(val)
            return val

        underl = '-' * len(header)
        output = '{}\n{}\n'.format(header, underl)
        for k, v in result.items():
            output += '- {} = {}\n'.format(k, format_value(v))
        print(output)

    def run_eval():
        m_n = CorpusRecall_WithGoldSpans_WithSomeNodes()
        m_c = CorpusRecall_WithGoldSpans_WithSomeNodes()

        for i, (g, p_n, p_c) in tqdm(enumerate(zip(gold, neural, cofill)), desc='eval'):
            m_n.update(g, p_n, p_c)
            m_c.update(g, p_c, p_c)

        res_n = m_n.finish()
        res_c = m_c.finish()

        print_result(res_n, 'Neural')
        print_result(res_c, 'COFILL')


    check_1()
    run_eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', default=None, required=True, type=str)
    parser.add_argument('--neural', default=None, required=True, type=str)
    parser.add_argument('--cofill', default=None, required=True, type=str)
    args = parser.parse_args()

    main(args)
