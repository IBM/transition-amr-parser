import argparse
from transition_amr_parser.io import read_amr


def argument_parser():

    parser = argparse.ArgumentParser(description='Tool to handle AMR')
    parser.add_argument(
        "--in-amr",
        help="input AMR files in pennman notation",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-amr",
        help="output AMR files in pennman notation",
        type=str,
    )
 
    args = parser.parse_args()

    return args


def main():

    # Argument handling
    args = argument_parser()

    # Load AMR (replace some unicode characters)
    corpus = read_amr(args.in_amr, unicode_fixes=True)
    amrs = corpus.amrs

    if args.out_amr:
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                fid.write(amr.toJAMRString())
