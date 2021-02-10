import sys
import re
import os
from glob import glob
from collections import defaultdict
import argparse


checkpoint_re = re.compile(r'checkpoint([0-9]+)\.pt')


def get_best_checkpoints(model_folder, file_types, ignore_deleted):
    best_checkpoints = dict()
    for file_type in file_types:
        best_checkpoints[file_type] = []
        for label in ['best', 'second_best', 'third_best']:
            link = f'{model_folder}/checkpoint_{label}_{file_type.upper()}.pt'

            # raise if best checkpoint was deleted
            if (
                os.path.islink(link)
                and not os.path.isfile(f'{model_folder}/{os.readlink(link)}')
                and not ignore_deleted
            ):
                raise Exception(
                    f'Best model was deleted: {original_checkpoint} '
                    'use --ignore-deleted to ignore'
                )

            if os.path.islink(link):
                best_checkpoints[file_type].append(
                    f'{model_folder}/{os.readlink(link)}'
                )
            else:
                return {}

    return best_checkpoints


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
            # Expected config.sh
            continue 

        # look for metrics used
        file_types = set()
        for dfile in glob(f'{model_folder}/epoch_tests/*'):
            bname = os.path.basename(dfile)
            file_types |= set(['.'.join(bname.split('.')[1:])])

        # subtract known terminations    
        file_types -= set(['en', 'actions', 'amr', 'wiki.amr', 'txt'])

        if file_types == set():
            # No results found for this model
            continue

        best_checkpoints = get_best_checkpoints(
            model_folder, 
            file_types,
            args.ignore_deleted
        )

        if any(b == [] for b in best_checkpoints.values()):
            # Some scores have no selected best checkpoints
            continue

        checkpoints_to_save = [
            os.path.basename(chp) 
            for score, chpts in best_checkpoints.items() for chp in chpts
        ]

        # sanity check, there should be 3 models to save
        sanity_checkpoints_to_save = []
        for checkpoint in glob(f'{model_folder}/checkpoint*.pt'):
            checkpoint_bname = os.path.basename(checkpoint)
            if checkpoint_bname in checkpoints_to_save:
                sanity_checkpoints_to_save.append(checkpoint)
        if len(sanity_checkpoints_to_save) < 3:
            print(
                f"Expected at least 3 models to save, skipping {model_folder}"
            )
            # no best links found, better not delete anything
            continue

        # delete checkpoints if the match regex, the are not pointed to by a
        # best softlink and they have been evaluated
        for checkpoint in glob(f'{model_folder}/checkpoint*.pt'):
            checkpoint_bname = os.path.basename(checkpoint)
            if (
                checkpoint_re.match(checkpoint_bname) 
                and checkpoint_bname not in checkpoints_to_save
             ):
                epoch = checkpoint_re.match(checkpoint_bname).groups()[0] 
                actions = f'{model_folder}/epoch_tests/dec-checkpoint{epoch}.actions'
                if os.path.isfile(actions):
                    # Then remove
                    if not args.check:
                        print(f'rm {checkpoint}')
                        os.remove(checkpoint)
                    else:    
                        print(f'Would rm {checkpoint}')
            else:
                print(f'Saving {checkpoint}')

