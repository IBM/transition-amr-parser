from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks



def main(args):

    tqdm_amrs_str = read_blocks(args.in_amr)


    with open(args.out_amr, 'a') as fid:
        for idx, penman_str in enumerate(tqdm_amrs_str):
            fid.write(penman_str+'\n')

           


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--in-amr",
        help="In file containing AMR in penman format",  
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="path to save amr",
        type=str,
    )
    args = parser.parse_args()
    main(args)