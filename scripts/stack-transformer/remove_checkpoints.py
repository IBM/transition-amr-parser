import sys
import re
import os
from glob import glob
from collections import defaultdict
import argparse


checkpoint_re = re.compile(r'checkpoint([0-9]+)\.pt')


def argument_parser():
    parser = argparse.ArgumentParser(description='Checkpoint remover')
    parser.add_argument("model_folders", nargs='+', type=str)
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--ignore-deleted", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = argument_parser()

    for model_folder in args.model_folders:

        # Skip if it does tnot look like a model folder
        config_path =  f'{model_folder}/config.sh'
        if not os.path.isfile(config_path):
           continue 

        # look for best* kind of checkpoints
        best_checkpoints = defaultdict(list)
        for label in ['best', 'second_best', 'third_best']:
            for best_checkpoint in glob(
                f'{model_folder}/checkpoint_{label}_*.pt'
            ):
                if os.path.islink(best_checkpoint):
                    checkpoint = os.readlink(best_checkpoint)
                    # link to a checkpoint
                    original_checkpoint = \
                        f'{model_folder}/{os.readlink(best_checkpoint)}'
                    if (
                        not os.path.isfile(original_checkpoint)
                        and not args.ignore_deleted
                    ):
                        # Pointing to deleted checkpoint
                        raise Exception(
                            f'Best model was deleted: {original_checkpoint} '
                            'use --ignore-deleted to ignore'
                        )
                    best_checkpoints[label].append(checkpoint)

        checkpoints_to_save = [
            cp for labels in best_checkpoints.values() for cp in labels
        ]

        for checkpoint in checkpoints_to_save:
            if args.check:
                print(f'Would save {model_folder}/{checkpoint}')
            else:
                print(f'Saved {model_folder}/{checkpoint}')

        if len(checkpoints_to_save) < 3:
            # no best links found, better not delete anything
            continue

        # delete checkpoints if the match regex, the are not pointed to by a
        # best softlink and they have been evaluated
        for checkpoint in glob(f'{model_folder}/checkpoint*.pt'):
            cp_basename = os.path.basename(checkpoint)
            if (
                checkpoint_re.match(cp_basename) 
                and cp_basename not in checkpoints_to_save
             ):
                epoch = checkpoint_re.match(cp_basename).groups()[0] 
                actions = f'{model_folder}/epoch_tests/dec-checkpoint{epoch}.actions'
                if os.path.isfile(actions):
                    # Then remove
                    if not args.check:
                        print(f'rm {checkpoint}')
                        os.remove(checkpoint)
                    else:    
                        print(f'Would rm {checkpoint}')
