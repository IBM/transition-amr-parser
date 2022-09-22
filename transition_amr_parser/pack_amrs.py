import argparse
import copy
import numpy as np
from tqdm import tqdm

from docamr_io import (
    AMR,
    read_amr,
)
from ipdb import set_trace

def connect_sen_amrs(amr):

    if len(amr.roots) <= 1:
        return

    node_id = amr.add_node("document")
    amr.root = str(node_id)
    for (i,root) in enumerate(amr.roots):
        amr.edges.append((amr.root, ":snt"+str(i+1), root))


def make_packed_amrs(amrs, max_tok=400, randomize=True):
    packed_amrs = []

    keys = [k for k in amrs.keys()]
    
    indices = np.array(range(len(amrs)))
    if randomize:
        indices = np.random.permutation(len(amrs))

    amr = copy.deepcopy(amrs[keys[indices[0]]])
    for idx in indices[1:]:
        next_amr = amrs[keys[idx]]
        if len(amr.tokens) + len(next_amr.tokens) <= max_tok:
            amr = amr + copy.deepcopy(next_amr)
        else:
            connect_sen_amrs(amr)
            amr.get_sen_ends()
            packed_amrs.append(amr)
            amr = copy.deepcopy(next_amr)

    connect_sen_amrs(amr)
    amr.get_sen_ends()
    packed_amrs.append(amr)

    return packed_amrs


def main(args):

    assert args.out_amr        
    assert args.in_amr        

    amrs = read_amr(args.in_amr)

    with open(args.out_amr, 'w') as fid:
        packed = make_packed_amrs(amrs)
        for amr in packed:
            fid.write(amr.__str__())
                
def argument_parser():

    parser = argparse.ArgumentParser(description='Read AMRs and Corefs and put them together', \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--in-amr",
        help="path to AMR3 annoratations",
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="Output file containing AMR in penman format",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
