import sys
if int(sys.version[0]) < 3:
    print("Needs at least Python 3")
    exit(1)
import subprocess
import time
import re
import os
import argparse
from glob import glob

checkpoint_re = re.compile('.*checkpoint([0-9]+)\.pt$')


def argument_parser():
    parser = argparse.ArgumentParser(description='Tool to handle AMR')
    parser.add_argument(
        "config",
        help="config used in the experiment",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="optional seed of the experiment",
        type=str,
    )
    parser.add_argument(
        "--list-checkpoints-to-eval",
        help="return all checkpoints with pending evaluation for a seed",
        action='store_true'
    )
    parser.add_argument(
        "--list-checkpoints-ready-to-eval",
        help="return all existing checkpoints with pending evaluation for a"
             " seed",
        action='store_true'
    )
    args = parser.parse_args()
    return args


def read_config_variables(config_path):

    # Read variables into dict
    # TODO: Read all ^[A-Z_]+= in the file (?)
    bash_script = (
        f'source {config_path};'
        'echo "ALIGNED_FOLDER=$ALIGNED_FOLDER";'
        'echo "ORACLE_FOLDER=$ORACLE_FOLDER";'
        'echo "EMB_FOLDER=$EMB_FOLDER";'
        'echo "DATA_FOLDER=$DATA_FOLDER";'
        'echo "MODEL_FOLDER=$MODEL_FOLDER";'
        'echo "SEEDS=$SEEDS";'
        'echo "MAX_EPOCH=$MAX_EPOCH";'
        'echo "EVAL_INIT_EPOCH=$EVAL_INIT_EPOCH";'
        'echo "EVAL_METRIC=$EVAL_METRIC";'
        'echo "DECODING_CHECKPOINT=$DECODING_CHECKPOINT"'
    )
    config_env_vars = {}
    proc = subprocess.Popen(bash_script, stdout = subprocess.PIPE, shell=True)
    for line in proc.stdout:
        (key, _, value) = line.decode('utf-8').strip().partition("=")
        config_env_vars[key] = value

    return config_env_vars


def print_step_status(step_folder):
    if os.path.isfile(f'{step_folder}/.done'):
        print(f"[\033[92mdone\033[0m] {step_folder}")
    elif os.path.isdir(step_folder):
        print(f"[\033[91mpart\033[0m] {step_folder}")
    else:
        print(f"[\033[93mpend\033[0m] {step_folder}")


def check_model_training(seed_folder, max_epoch):

    final_checkpoint = f'{seed_folder}/checkpoint{max_epoch}.pt'
    if os.path.isfile(final_checkpoint):
        # Last epoch completed
        print(f"[\033[92m{max_epoch}/{max_epoch}\033[0m] {seed_folder}")
    else:
        # Get which epochs are completed
        epochs = [] 
        for checkpoint in glob(f'{seed_folder}/checkpoint*.pt'):
            fetch = checkpoint_re.match(checkpoint)
            if fetch:
                epochs.append(int(fetch.groups()[0]))
        if epochs:
            curr_epoch = max(epochs)
        else:    
            curr_epoch = 0
        print(f"[\033[93m{curr_epoch}/{max_epoch}\033[0m] {seed_folder}")


def get_checkpoints_to_eval(config_env_vars, seed, ready=False):
    """
    List absolute paths of checkpoints needed for evaluation. Restrict to
    existing ones if read=True
    """

    # Get variables from config
    model_folder = config_env_vars['MODEL_FOLDER']
    seed_folder = f'{model_folder}-seed{seed}'
    max_epoch = int(config_env_vars['MAX_EPOCH'])
    eval_metric = config_env_vars['EVAL_METRIC']
    eval_init_epoch = int(config_env_vars['EVAL_INIT_EPOCH'])

    val_result_re = re.compile(f'dev-checkpoint([0-9]+)\.{eval_metric}')
    validation_folder = f'{seed_folder}/epoch_tests/'
    epochs = {}
    for result in glob(f'{validation_folder}/*.{eval_metric}'):
        fetch = val_result_re.match(result)
        if fetch:
            epochs.append(int(fetch.groups()[0]))
    target_epochs = range(eval_init_epoch, max_epoch+1) 
    missing_epochs = set(target_epochs) - set(epochs)
    missing_epochs = sorted(missing_epochs, reverse=True)

    # construct paths
    checkpoints = []
    for epoch in missing_epochs:
        checkpoint = f'{seed_folder}/checkpoint{epoch}.pt'
        if os.path.isfile(checkpoint) or not ready:
            checkpoints.append(os.path.realpath(checkpoint))

    return checkpoints, target_epochs


def print_status(config_env_vars):

    # Inform about completed stages
    # pre-training ones
    print()
    for variable in ['ALIGNED_FOLDER', 'ORACLE_FOLDER', 'EMB_FOLDER', 
                     'DATA_FOLDER']:
        print_step_status(config_env_vars[variable])
    # training/eval ones
    model_folder = config_env_vars['MODEL_FOLDER']
    for seed in config_env_vars['SEEDS'].split():

        # all checkpoints trained
        seed_folder = f'{model_folder}-seed{seed}'
        max_epoch = int(config_env_vars['MAX_EPOCH'])
        check_model_training(seed_folder, max_epoch)

        # all checkpoints evaluated
        checkpoints, target_epochs = get_checkpoints_to_eval(config_env_vars, seed)
        if checkpoints:
            print(f"[\033[93m{len(target_epochs) - len(checkpoints)}/{len(target_epochs)}\033[0m] {seed_folder}")
        else:
            print(f"[\033[92m{len(target_epochs)}/{len(target_epochs)}\033[0m] {seed_folder}")

        # Final model and results
        dec_checkpoint = config_env_vars['DECODING_CHECKPOINT']
        dec_checkpoint = f'{model_folder}-seed{seed}/{dec_checkpoint}'
        if os.path.isfile(dec_checkpoint):
            print(f"[\033[92mdone\033[0m] {dec_checkpoint}")
        else:
            print(f"[\033[93mpend\033[0m] {dec_checkpoint}")
    print()


def main(args):

    # TODO: wait and release x checkpoints tops, wait until completed to
    # release more use seed as argument

    config_env_vars = read_config_variables(args.config)
    if args.list_checkpoints_ready_to_eval or args.list_checkpoints_to_eval:

        # List checkpoints pending for evaluation
        assert args.seed, "Requires --seed"
        checkpoints = get_checkpoints_to_eval(
            config_env_vars,
            args.seed, 
            ready=bool(args.list_checkpoints_ready_to_eval)
        )[0]
        for checkpoint in checkpoints:
            print(checkpoint)
            sys.stdout.flush()

    else:

        # TODO: Add support for --seed here
        print_status(config_env_vars)


if __name__ == '__main__':
    main(argument_parser())
