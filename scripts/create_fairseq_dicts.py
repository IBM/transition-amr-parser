import sys
import re
import argparse
from collections import Counter


def argument_parser():
    parser = argparse.ArgumentParser(description='Add fairseq dicts')
    # The idea here is to keep pretraining indices to re-use model embeddings
    # when loading
    parser.add_argument(
        "-i", "--in-pretrain-dict",
        help="Pretrainign fairseq dict (indices are preserved)",
        required=True
    )
    parser.add_argument(
        "-t", "--in-fine-tune-data",
        help="Fine tuning data, tokens tab separated",
        required=True
    )
    args = parser.parse_args()

    return args


def read_fairseq_dict(file_path):
    line_re = re.compile('(.*) ([0-9]+)')
    with open(file_path) as fid:
        items = []
        for line in fid:
            assert line_re.match(line.strip()), "Not a fairseq dict"
            word, count = line_re.match(line).groups()
            items.append([word, int(count)])
    return items


def write_fairseq_dict(file_path, out_entries):
    with open(file_path, 'w') as fid:
        for key, count in out_entries:
            fid.write(f'{key} {count}\n')


def read_data_into_dict(file_path):
    with open(file_path) as fid:
        return Counter([
            token for line in fid for token in line.strip().split('\t')
        ])


if __name__ == '__main__':

    # The idea is to respect the pretrained dict indices to be able to reuse
    # the pretrained embedings

    # Argument handling
    args = argument_parser()

    # Read fine-tune entries and construct a mapping dict
    out_entries = read_fairseq_dict(args.in_pretrain_dict)
    key_to_index = {key: index for index, (key, _) in enumerate(out_entries)}
    print(f'Read {len(out_entries)} {args.in_pretrain_dict}')

    # Sum to count or append fine tune entries.
    for entry, count in read_data_into_dict(args.in_fine_tune_data).items():
        if entry in key_to_index:
            out_entries[key_to_index[entry]][1] += count
        else:
            out_entries.append((entry, count))

    write_fairseq_dict(args.in_pretrain_dict, out_entries)
    print(f'Wrote {len(out_entries)} {args.in_pretrain_dict}')
