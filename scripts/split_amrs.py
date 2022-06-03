import argparse
from tqdm import tqdm
import sys
import os
import penman
from transition_amr_parser.io import read_blocks
from ipdb import set_trace


def main():

    in_amr, max_split_size, output_basename = sys.argv[1:]
    dirname = os.path.dirname(output_basename)
    os.makedirs(dirname, exist_ok=True)

    amrs = read_blocks(in_amr, return_tqdm=False)
    max_split_size = int(max_split_size)

    num_amrs = len(amrs)
    indices = list(range(num_amrs))
    chunk_indices = [
        indices[i:i + max_split_size]
        for i in range(0, num_amrs, max_split_size)
    ]

    for chunk_n, indices in enumerate(tqdm(chunk_indices)):
        split_file = f'{output_basename}.{chunk_n}'
        with open(split_file, 'w') as fid:
            for i in indices:
                fid.write(f'{amrs[i]}\n')
        print(split_file)


if __name__ == '__main__':
    main()
