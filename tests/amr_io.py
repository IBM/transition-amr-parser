import argparse
import re
import subprocess
from collections import Counter
from transition_amr_parser.io import AMR, read_blocks
from transition_amr_parser.amr import (
    trasverse,
    ANNOTATION_ISSUES,
    normalize
)
from ipdb import set_trace


def vimdiff(penman_str, penman_str2, index=''):

    def write(file_name, content):
        with open(file_name, 'w') as fid:
            fid.write(content)

    def rm_initial_space(string):
        return '\n'.join(re.sub(' *', '', x) for x in string.split('\n'))

    if rm_initial_space(penman_str) != rm_initial_space(penman_str2):
        input(f'\nPress any key to compare sentences {index}')
        write('tmp1', penman_str)
        write('tmp2', penman_str2)
        subprocess.call(['vimdiff', 'tmp1', 'tmp2'])


def argument_parser():

    parser = argparse.ArgumentParser(
        description='Test AMR reading and writing functions'
    )
    # Single input parameters
    parser.add_argument("--in-amr", type=str, required=True,
                        help="AMR notation in penman format")
    parser.add_argument("--out-amr", type=str,
                        help="AMR notation in penman format")
    parser.add_argument("--ignore-errors", type=str,
                        choices=ANNOTATION_ISSUES.keys(),
                        help="Ignore known errors")
    parser.add_argument("--no-isi", action='store_true',
                        help="Write alignments in ::alignments")
    args = parser.parse_args()

    return args


def stop_if_duplicates(amr2):
    duplicates = [
        k for k, c in Counter(amr2.penman.triples).items() if c > 1
    ]
    if duplicates:
        return duplicates


def stop_if_triples_do_not_match(amr, amr2):

    # get triples ignoring quotes
    triples = [
        (normalize(a), b, normalize(c))
        for (a, b, c) in amr.penman.triples
    ]
    triples2 = [
        (normalize(a), b, normalize(c))
        for (a, b, c) in amr2.penman.triples
    ]

    if set(triples) != set(triples2):
        missing = set(triples) - set(triples2)
        excess = set(triples2) - set(triples)
        return missing, excess
    else:
        return [], []


def main(args):

    # for AMR2.0
    if bool(args.ignore_errors):
        annotation_error_indices = ANNOTATION_ISSUES[args.ignore_errors]
    else:
        annotation_error_indices = []

    # get tqdm iterator over strings of penman notation
    tqdm_iterator = read_blocks(args.in_amr)

    num_cycles = 0
    out_amr_penmans = []
    for index, penman_str in enumerate(tqdm_iterator):

        # read, write back and read again
        amr = AMR.from_penman(penman_str)
        if args.no_isi:
            penman_str2 = amr.to_penman(isi=False)
        else:
            penman_str2 = amr.__str__()
        out_amr_penmans.append(penman_str2)
        amr2 = AMR.from_penman(penman_str2)

        # graphs are either single root or acyclic due to "-of" inversions
        # count number of cases where assuming single root yields cycles
        edge_stacks, offending_stacks = trasverse(amr.edges, amr.root)
        num_cycles += int(bool(offending_stacks['cycle']))

        # look for duplicate edge triples
        dupes = stop_if_duplicates(amr2)
        if bool(dupes) and index not in annotation_error_indices:
            print(penman_str)
            print()
            print(penman_str2)
            print(dupes)
            print(f'[ \033[91mFAILED\033[0m ] Duplicate edges triples {index}')
            set_trace(context=30)
            vimdiff(penman_str, penman_str2, index=index)
            exit(1)

        # look for different triples
        missing, excess = stop_if_triples_do_not_match(amr, amr2)
        if (missing or excess) and index not in annotation_error_indices:
            # Apparently starting here in AMR3.0, wiki starts being quoted
            # we can ignore this one
            if (
                args.ignore_errors == 'amr3-train'
                and index > 24439
                and all([e[2] == '"-"' for e in excess])
            ):
                continue
            print(penman_str)
            print()
            print(penman_str2)
            print(missing)
            print(excess)
            print(f'[ \033[91mFAILED\033[0m ] Missing/Excess triples {index}')
            set_trace(context=30)
            vimdiff(penman_str, penman_str2, index=index)
            exit(1)

        # look for different alignments
        if amr.alignments and amr.alignments != amr2.alignments:
            print(penman_str)
            print()
            print(penman_str2)
            set_trace(context=30)
            print(f'[ \033[91mFAILED\033[0m ] Missing Alignments {index}')
            # exit(1)

        # check this is not marked as not ok, but its actually ok
        if index in annotation_error_indices:
            # NOTE that there being an error in an annotation does not
            # guarantee and error in io.
            print(f'{index} has annotation errors, but passes anyway')

    if args.out_amr:
        with open(args.out_amr, 'w') as fid:
            for penman in out_amr_penmans:
                fid.write(f'{penman}\n')

    print(f'[ \033[92mOK\033[0m ] {args.in_amr} ({args.ignore_errors})')


if __name__ == '__main__':
    main(argument_parser())
