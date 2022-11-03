from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re

regex = r"--avoid-indices ([\d\s]+)"


def main(args):

    tqdm_amrs_str = read_blocks(args.in_amr)
    indices = re.findall(regex,args.arg_str)
    avoid_indices = indices[0].split()
    avoid_indices = [int(i) for i in avoid_indices]

    with open(args.out_amr, 'w') as fid:
        for idx, penman_str in enumerate(tqdm_amrs_str):
            if not idx in avoid_indices:
                fid.write(penman_str+'\n')

           


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--in-amr",
        help="In file containing AMR in penman format",  
        type=str
    )
    parser.add_argument(
        "--arg-str",
        help="the arg string containing the indices needed to be removed",  
        type=str
    )

    parser.add_argument(
        "--out-amr",
        help="out amr after removal of avois indices",
        type=str,
    )
    args = parser.parse_args()
    main(args)


