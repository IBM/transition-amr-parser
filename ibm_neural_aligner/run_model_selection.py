import argparse
import collections
import os

import numpy as np

from tqdm import tqdm

import metric_utils
from amr_utils import safe_read as safe_read_


def write_amr(path, corpus):
    with open(path, 'w') as f:
        for amr in corpus:
            f.write(f'{amr.__str__()}\n')


def safe_read(path, **kwargs):
    if args.aligner_training_and_eval:
        kwargs['ibm_format'], kwargs['tokenize'] = True, False
    else:
        if args.no_jamr:
            kwargs['ibm_format'], kwargs['tokenize'] = False, True
        else:
            kwargs['ibm_format'], kwargs['tokenize'] = False, False

    return safe_read_(path, **kwargs)


class Machine:

    @staticmethod
    def read_actions(path):
        data = []
        with open(path) as f:
            for line in f:
                actions = line.strip().split()
                data.append(actions)
        return data

    @staticmethod
    def is_node(seq):
        return [(not x.startswith('>')) and x != 'SHIFT' for x in seq]

    @staticmethod
    def is_arc(seq):
        return [x.startswith('>') for x in seq]

    def __init__(self, corpus, actions):
        actions_corpus = []

        for amr, seq in zip(corpus, actions):
            tokens = amr.tokens

            new_seq = []

            cursor = 0

            for a in seq:
                a_arg = None
                if a == 'SHIFT':
                    cursor += 1
                elif a == 'COPY':
                    # TODO: This doesn't accurately reflect copy, since does not take lemmatization into account.
                    token_name = amr.tokens[cursor]
                    a_arg = token_name
                new_seq.append((a, a_arg))

            actions_corpus.append(new_seq)
        self.actions_corpus = actions_corpus


def compute_metrics(amr, seq, dseq):

    metrics = {}
    if amr.alignments is None:
        metrics['success'] = 0
        return metrics

    # Fertility.
    f = metric_utils.fertility_proxy(amr)

    # Distortion.
    d, dl = metric_utils.distortion_proxy(amr)
    assert not np.isnan(d), (d, dl, amr.alignments)

    # Copy frequency.
    ignore_nodes = ('country', '-', 'and', 'person', 'name')
    num_copy = len([None for a, a_arg in dseq if a == 'COPY' and a_arg not in ignore_nodes])

    # Pointer action.
    cum_nodes = np.cumsum(Machine.is_node(seq))
    num_candidate_nodes = cum_nodes[Machine.is_arc(seq)]
    pool_size = num_candidate_nodes.sum()

    metrics['pool_size'] = pool_size
    metrics['num_copy'] = num_copy
    metrics['fertility'] = f
    metrics['distortion'] = d
    metrics['success'] = 1
    return metrics


def print_header(header, prefix=''):
    if prefix is not None:
        print(prefix)
    divider = '=' * len(header)
    print(header)
    print(divider)


def main(args):

    assert len(args.in_amrs) == len(args.in_actions)

    # READ

    # Note: Reads each file, then begins model selection. Could be faster to
    # do selection one file at a time.

    datasets = collections.OrderedDict()

    for i, path in enumerate(args.in_amrs):

        print(i, path)

        action_path = args.in_actions[i]
        metrics = collections.defaultdict(list)
        corpus = safe_read(path)
        actions_corpus = Machine.read_actions(action_path)
        decorated_actions = Machine(corpus, actions_corpus).actions_corpus

        dinfo = {}
        dinfo['id'] = i
        dinfo['path'] = path
        dinfo['corpus'] = corpus
        dinfo['action_path'] = action_path
        dinfo['actions_corpus'] = actions_corpus
        dinfo['decorated_actions'] = decorated_actions

        datasets[i] = dinfo

    # COMPUTE METRICS

    for i, dinfo in datasets.items():
        corpus, actions_corpus, decorated_actions = \
            dinfo['corpus'], dinfo['actions_corpus'], dinfo['decorated_actions']

        metrics = collections.defaultdict(list)

        for amr, seq, dseq in tqdm(zip(corpus, actions_corpus, decorated_actions), desc='{}:metrics'.format(i)):
            for k, v in compute_metrics(amr, seq, dseq).items():
                metrics[k].append(v)

        dinfo['metrics'] = metrics

    # SUMMARY

    print_header('SUMMARY')

    for i, dinfo in datasets.items():
        print(i, dinfo['path'])
        for k, v in dinfo['metrics'].items():
            print(k, np.mean(v))

    # COMPARE

    print_header('COMPARE')

    compare_metrics = collections.OrderedDict()
    for i, dinfo1 in datasets.items():
        for j, dinfo2 in datasets.items():
            if i >= j:
                continue
            for k in dinfo1['metrics'].keys():
                diff = np.array(dinfo1['metrics'][k]) - np.array(dinfo2['metrics'][k])
                compare_metrics[(i, j, k)] = diff

    for (i, j, k), diff in compare_metrics.items():
        gt = int((diff > 0).sum())
        eq = int((diff == 0).sum())
        lt = int((diff < 0).sum())

        print(f'{i} {j} {k} <:{lt} =:{eq} >:{gt}')

    # COMBINE

    # TODO: Currently only selects model based on pool size.

    print_header('COMBINE')

    mbr_corpus = []
    mbr_action_corpus = []
    mbr_dinfo = {}
    mbr_metrics = collections.defaultdict(list)
    corpus_size = len(datasets[0]['corpus'])
    num_datasets = len(args.in_amrs)

    selection = collections.Counter()

    for i in tqdm(range(corpus_size), desc='select-models'):
        values = [datasets[j]['metrics']['pool_size'][i] for j in range(num_datasets)]
        best_idx = np.argmin(values)
        amr, seq = datasets[best_idx]['corpus'][i], datasets[best_idx]['actions_corpus'][i]
        dseq = datasets[best_idx]['decorated_actions'][i]
        mbr_corpus.append(amr)
        mbr_action_corpus.append(seq)

        selection[best_idx] += 1

        for k, v in compute_metrics(amr, seq, dseq).items():
            mbr_metrics[k].append(v)

    for k, v in mbr_metrics.items():
        print(k, np.mean(v))

    print('selection', selection)

    # WRITE

    if args.out_amr is not None:

        print('writing... {}'.format(args.out_amr))

        write_amr(args.out_amr, mbr_corpus)


def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-amrs', action='append', default=[])
    parser.add_argument('--in-actions', action='append', default=[])
    parser.add_argument('--out-amr', default='best.txt', type=str)
    parser.add_argument('--preset', default=None, type=str)
    parser.add_argument(
        "--no-jamr",
        help="If true, then read penman. Otherwise, read JAMR.",
        action='store_true',
    )
    parser.add_argument(
        "--aligner-training-and-eval",
        help="Set when training or evaluating aligner.",
        action='store_true',
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    if args.preset == 'dev':
        args.in_amrs.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.txt'))
        args.in_actions.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR2.0/aligned/ibm_neural_aligner/dev.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR2.0/aligned/ibm_neural_aligner/dev.actions'))
    if args.preset == 'train':
        args.in_amrs.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.txt'))
        args.in_actions.append(os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR2.0/aligned/ibm_neural_aligner/train.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR2.0/aligned/ibm_neural_aligner/train.actions'))

    if args.preset == 'mbr-2':
        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/train.actions'))

    if args.preset == 'mbr-3':
        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/train.actions'))

    if args.preset == 'mbr-3+rand':
        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gold-ref/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/random/train.txt'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/random/train.actions'))

    if args.preset == 'mbr-4':
        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-lstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-bilstm-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-tree_lstm_v4-write_amr3/train.actions'))

        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/train.actions'))

    if args.preset == 'gcn':
        args.in_amrs.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/alignment.trn.out.pred'))
        args.in_actions.append(os.path.expanduser('DATA/AMR3.0-aligners/best-gcn-write_amr3/train.actions'))

    if args.preset is not None:
        args.mbr_output = 'best-{}.txt'.format(args.preset)

    main(args)
