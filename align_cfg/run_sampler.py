# -*- coding: utf-8 -*-

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

from tqdm import tqdm

from amr_utils import get_node_ids
from amr_utils import safe_read as safe_read_
from formatter import amr_to_string, read_amr_pretty_file

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
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    return args


def main(args):
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



if __name__ == '__main__':
    args = argument_parser()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    print(args.__dict__)

    main(args)
