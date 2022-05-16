import argparse
import json
import os

from amr_utils import safe_read
from evaluation import EvalAlignments
from formatter import amr_to_pretty_format
from transition_amr_parser.io import read_amr2


parser = argparse.ArgumentParser()
parser.add_argument('--gold', default=None, required=True, type=str)
parser.add_argument('--pred', default=None, required=True, type=str)
parser.add_argument('--out-json', default=None, type=str)
parser.add_argument('--subset', action='store_true')
parser.add_argument('--increment', action='store_true')
args = parser.parse_args()

if args.out_json is None:
    args.out_json = args.pred + '.eval.json'

print('start eval')

eval_output = EvalAlignments().run(args.gold, args.pred, flexible=True, subset=args.subset, increment=args.increment)

print(eval_output)

with open(args.out_json, 'w') as f:
    f.write(json.dumps(eval_output))
