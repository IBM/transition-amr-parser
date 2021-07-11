import collections
import json
import os

import numpy as np

from tqdm import tqdm

from amr_utils import read_amr
from evaluation import EvalAlignments


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
    parser.add_argument('--only-MAP', action='store_true')
    args = parser.parse_args()

    if args.base is not None:
        args.pred = os.path.join(args.base, 'alignment.trn.out.pred')
        args.gold = os.path.join(args.base, 'alignment.trn.out.gold')
        args.out = os.path.join(args.base, 'alignment.trn.out.report')

    print(json.dumps(args.__dict__, sort_keys = True, indent = 4))

    EvalAlignments().run(args.gold, args.pred, only_MAP=args.only_MAP)

