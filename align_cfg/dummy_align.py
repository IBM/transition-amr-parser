import argparse
import copy

from formatter import amr_to_string
from transition_amr_parser.io import read_amr2


parser = argparse.ArgumentParser()
parser.add_argument('--in-amr', default=None, required=True, type=str)
parser.add_argument('--out-amr', default=None, required=True, type=str)
args = parser.parse_args()

def dummy_align(amr):
    amr = copy.deepcopy(amr)
    alignments = {}
    for k in sorted(amr.nodes.keys()):
        alignments[k] = [0]
    amr.alignments = alignments
    return amr

corpus = read_amr2(args.in_amr, ibm_format=False, tokenize=False)

with open(args.out_amr,'w') as f:
    for amr in corpus:
        amr = dummy_align(amr)
        f.write(amr_to_string(amr).strip() + '\n\n')
