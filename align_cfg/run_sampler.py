# -*- coding: utf-8 -*-

import argparse
import collections
import copy
import json
import os
import sys

import numpy as np
import torch

from tqdm import tqdm

from amr_utils import convert_amr_to_tree, compute_pairwise_distance, get_node_ids
from amr_utils import safe_read as safe_read_
from formatter import amr_to_string, read_amr_pretty_file
from metric_utils import distortion_proxy

from transition_amr_parser.io import read_amr2


def safe_read(path, **kwargs):
    if args.aligner_training_and_eval:
        kwargs['ibm_format'], kwargs['tokenize'] = True, False
    else:
        if args.no_jamr:
            kwargs['ibm_format'], kwargs['tokenize'] = False, True
        else:
            kwargs['ibm_format'], kwargs['tokenize'] = False, False

    return safe_read_(path, **kwargs)


def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='mode_1', type=str)
    parser.add_argument('--n-samples-ckpt', default=10, type=int)
    parser.add_argument('--n-samples', default=100, type=int)
    parser.add_argument(
        "--in-amr",
        help="AMR input file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--in-amr-pretty",
        help="AMR input file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-amr",
        help="AMR input file.",
        type=str,
        default=None,
    )
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
    parser.add_argument(
        "--temp",
        help="Temperature for softmax.",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    return args


def run_mode_1(args):
    """ Sample once per sentence and write.

    """
    corpus = safe_read(args.in_amr)
    posterior_list = read_amr_pretty_file(args.in_amr_pretty, corpus)

    # sample
    new_alignments = []
    for amr, posterior in tqdm(zip(corpus, posterior_list), desc='sample'):
        text_tokens = amr.tokens
        node_ids = get_node_ids(amr)
        shape = (len(node_ids), len(text_tokens))
        assert posterior.shape == shape, (posterior.shape, shape)
        pt_posterior = torch.from_numpy(posterior).float()
        logits = (pt_posterior + 1e-8).log()

        dist = torch.distributions.multinomial.Multinomial(logits=logits / args.temp)
        sample = dist.sample().argmax(1).tolist()

        a = {node_ids[i]: [sample[i]] for i in range(len(node_ids))}
        new_alignments.append(a)

    # write
    with open(args.out_amr, 'w') as f_out:
        for amr, a in zip(corpus, new_alignments):
            f_out.write(amr_to_string(amr, alignments=a).strip() + '\n\n')


def run_mode_2(args):
    """ Sample N times per sentence and choose best.

    """
    corpus = safe_read(args.in_amr)
    posterior_list = read_amr_pretty_file(args.in_amr_pretty, corpus)

    ckpt = collections.defaultdict(list)

    def print_so_far():
        print('init', np.mean([x['d'] for x in ckpt['init']]))
        print('best_ckpt', np.mean([x['d'] for x in ckpt['best_ckpt']]))
        print('best', np.mean([x['d'] for x in ckpt['best']]))

    # sample
    for i, (amr, posterior) in enumerate(zip(corpus, posterior_list)):
        text_tokens = amr.tokens
        node_ids = get_node_ids(amr)
        shape = (len(node_ids), len(text_tokens))
        assert posterior.shape == shape, (posterior.shape, shape)
        pt_posterior = torch.from_numpy(posterior).float()
        logits = (pt_posterior + 1e-8).log()
        probs = torch.softmax(logits / args.temp, 1)

        try:
            dist = torch.distributions.multinomial.Multinomial(total_count=1, probs=probs)
        except:
            import ipdb; ipdb.set_trace()

        tree = convert_amr_to_tree(amr)
        pairwise_dist = compute_pairwise_distance(tree)
        tmp_amr = copy.deepcopy(amr)
        d = distortion_proxy(amr, pairwise_dist=pairwise_dist)[0]

        init_a_info = dict(d=d, a=amr.alignments)
        best_a_info = init_a_info
        best_a_info_ckpt = None

        for i_sample in range(args.n_samples):
            sample = dist.sample().argmax(1).tolist()

            a = {node_ids[i]: [sample[i]] for i in range(len(node_ids))}

            tmp_amr.alignments = a

            d = distortion_proxy(tmp_amr, pairwise_dist=pairwise_dist)[0]

            a_info = dict(d=d, a=a)

            if d < best_a_info['d']:
                best_a_info = a_info

            if i_sample == args.n_samples_ckpt - 1:
                best_a_info_ckpt = best_a_info

        ckpt['init'].append(init_a_info)
        ckpt['best'].append(best_a_info)
        ckpt['best_ckpt'].append(best_a_info_ckpt)

        # log
        prefix = '[{}/{} {:.3f}%]'.format(i, len(corpus), i / len(corpus) * 100)

        d_init = init_a_info['d']
        d_best = best_a_info['d']
        d_best_ckpt = best_a_info_ckpt['d']

        d_diff = d_best - d_init
        d_str = 'd:( init = {:.3f} , best = {:.3f} , diff = {:.3f} )'.format(d_init, d_best, d_diff)

        d_diff = d_best - d_best_ckpt
        d_str_ckpt = 'd_ckpt:( best_ckpt = {:.3f} , best = {:.3f} , diff = {:.3f} )'.format(d_best_ckpt, d_best, d_diff)

        print(prefix)
        print(d_str)
        print(d_str_ckpt)
        print()

        if i % 1000 == 0:
            print('=' * 80)
            print_so_far()
            print('\n\n')

    print('init', np.mean([x['d'] for x in ckpt['init']]))
    print('best_ckpt', np.mean([x['d'] for x in ckpt['best_ckpt']]))
    print('best', np.mean([x['d'] for x in ckpt['best']]))


def main(args):
    eval('run_{}(args)'.format(args.mode))


if __name__ == '__main__':
    args = argument_parser()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(args.__dict__)

    main(args)
