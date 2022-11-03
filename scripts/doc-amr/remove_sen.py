from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re

regex = r"--avoid-indices ([\d\s]+)"


def main(args):

    lines = open(args.in_file).readlines()
    indices = re.findall(regex,args.arg_str)
    avoid_indices = indices[0].split()
    avoid_indices = [int(i) for i in avoid_indices]

    with open(args.out_file, 'w') as fid:
        for idx, line in enumerate(lines):
            if not idx in avoid_indices:
                fid.write(line)

           


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file",
        help="In file containing sen",  
        type=str
    )
    parser.add_argument(
        "--arg-str",
        help="the arg string containing the indices needed to be removed",  
        type=str
    )

    parser.add_argument(
        "--out-file",
        help="out file after removal of avoids indices",
        type=str,
    )
    args = parser.parse_args()
    main(args)


