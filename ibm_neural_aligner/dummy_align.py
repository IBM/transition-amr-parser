import argparse
import copy
from transition_amr_parser.io import read_amr


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


if __name__ == '__main__':
    corpus = read_amr(args.in_amr, jamr=False)
    with open(args.out_amr, 'w') as f:
        for amr in corpus:
            amr = dummy_align(amr)
            f.write(f'{amr.__str__()}\n')
