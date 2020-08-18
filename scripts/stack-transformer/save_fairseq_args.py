import json
import argparse
from collections import defaultdict


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser oracle')
    # Single input parameters
    parser.add_argument(
        "--fairseq-preprocess-args",
        help="command lines args for fairseq-preprocess",
        type=str,
        required=True
    )
    parser.add_argument(
        "--fairseq-train-args",
        help="command lines args for fairseq-train",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-fairseq-model-config",
        help="Config",
        type=str,
        required=True
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # Argument handling
    args = argument_parser()
    config = defaultdict(dict)
    for arg_pair in args.fairseq_preprocess_args.strip().split('\n'):
        items = arg_pair.strip().split()
        config['fairseq_preprocess_args'][items[0]] = items[1:]
    for arg_pair in args.fairseq_train_args.strip().split('\n'):
        items = arg_pair.strip().split()
        if len(items) == 1:
            config['fairseq_train_args']['data'] = items[1:]
        else:
            config['fairseq_train_args'][items[0]] = items[1:]
    # Write config
    with open(args.out_fairseq_model_config, 'w') as fid:
        fid.write(json.dumps(config))
    print(f'Saved fairseq model args to {args.out_fairseq_model_config}')
