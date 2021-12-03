import argparse
import json
import os

from amr_utils import safe_read
from evaluation import EvalAlignments
from formatter import amr_to_pretty_format, amr_to_string
from transition_amr_parser.io import read_amr2


parser = argparse.ArgumentParser()
parser.add_argument('--gold', default=None, required=True, type=str)
parser.add_argument('--pred', default=None, required=True, type=str)
args = parser.parse_args()

print('start eval')

eval_output = EvalAlignments().run(args.gold, args.pred, flexible=True)

print(eval_output)

with open(args.pred + '.eval.json', 'w') as f:
    f.write(json.dumps(eval_output))

