import argparse
import copy

from austin_amr_utils.amr_readers import AMR_Reader
from transition_amr_parser.io import read_amr2


parser = argparse.ArgumentParser()
parser.add_argument('--in-amr', default=None, required=True, type=str)
parser.add_argument('--out-amr', default=None, required=True, type=str)
args = parser.parse_args()

corpus = AMR_Reader().load(args.in_amr)

with open(args.out_amr,'w') as f:
    for amr in corpus:
        f.write(amr.amr_string().strip() + '\n\n')
