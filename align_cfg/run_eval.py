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

gold = safe_read(args.gold, ibm_format=True, tokenize=False)

with open(args.pred + '.gold','w') as f:
    for amr in gold:
        f.write(amr_to_string(amr).strip() + '\n\n')

pred = safe_read(args.pred, ibm_format=True, tokenize=False)

with open(args.pred + '.pred','w') as f:
    for amr in pred:
        f.write(amr_to_string(amr).strip() + '\n\n')

eval_output = EvalAlignments().run(args.pred + '.gold', args.pred + '.pred', flexible=True)

print(eval_output)

with open(args.pred + '.eval.json', 'w') as f:
    f.write(json.dumps(eval_output))

os.system('rm {}'.format(args.pred + '.gold'))
os.system('rm {}'.format(args.pred + '.pred'))

