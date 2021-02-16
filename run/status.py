import sys
import subprocess
import re
import os
import argparse
from glob import glob


# Sanity check python3
if int(sys.version[0]) < 3:
    print("Needs at least Python 3")
    exit(1)


# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
checkpoint_re = re.compile(r'.*checkpoint([0-9]+)\.pt$')


def argument_parser():
    parser = argparse.ArgumentParser(description='Tool to check experiments')
    parser.add_argument(
        "--results",
        help="print results for all complete models",
        type=str,
    )
    parser.add_argument(
        "-c", "--config",
        help="select one experiment by a config",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="optional seed of the experiment",
        type=str,
    )
    parser.add_argument(
        "--nbest",
        help="Top-n best checkpoints to keep",
        default=5
    )
    parser.add_argument(
        "--link-best",
        help="Link best model if all checkpoints are done",
        action='store_true'
    )
    parser.add_argument(
        "--remove",
        help="Remove checkpoints that have been evaluated and are not best "
             "checkpoints",
        action='store_true'
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
    proc = subprocess.Popen(
        bash_script, stdout=subprocess.PIPE, shell=True, executable='/bin/bash'
    )
    for line in proc.stdout:
        (key, _, value) = line.decode('utf-8').strip().partition("=")
        config_env_vars[key] = value

    return config_env_vars


def print_step_status(step_folder):
    if os.path.isfile(f'{step_folder}/.done'):
        print(f"[\033[92mdone\033[0m] {step_folder}")
    elif os.path.isdir(step_folder):
        print(f"[\033[93mpart\033[0m] {step_folder}")
    else:
        print(f"[pend] {step_folder}")


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
            print(f"[\033[93m{curr_epoch}/{max_epoch}\033[0m] {seed_folder}")
        else:
            curr_epoch = 0
            print(f"[{curr_epoch}/{max_epoch}] {seed_folder}")


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

    val_result_re = re.compile(r'.*de[cv]-checkpoint([0-9]+)\.' + eval_metric)
    validation_folder = f'{seed_folder}/epoch_tests/'
    epochs = []
    for result in glob(f'{validation_folder}/*.{eval_metric}'):
        fetch = val_result_re.match(result)
        if fetch:
            epochs.append(int(fetch.groups()[0]))
    target_epochs = list(range(eval_init_epoch, max_epoch+1))
    missing_epochs = set(target_epochs) - set(epochs)
    missing_epochs = sorted(missing_epochs, reverse=True)

    # construct paths
    checkpoints = []
    for epoch in missing_epochs:
        checkpoint = f'{seed_folder}/checkpoint{epoch}.pt'
        if os.path.isfile(checkpoint) or not ready:
            checkpoints.append(os.path.realpath(checkpoint))

    return checkpoints, target_epochs


def print_status(config_env_vars, seed):

    # Inform about completed stages
    # pre-training ones
    print()
    for variable in ['ALIGNED_FOLDER', 'ORACLE_FOLDER', 'EMB_FOLDER',
                     'DATA_FOLDER']:
        print_step_status(config_env_vars[variable])
    # training/eval ones
    model_folder = config_env_vars['MODEL_FOLDER']
    if seed is None:
        seeds = config_env_vars['SEEDS'].split()
    else:
        assert seed in config_env_vars['SEEDS'].split(), \
            "{seed} is not a trained seed for the model"
        seeds = [seed]
    # loop over each model with a different random seed
    for seed in seeds:

        # all checkpoints trained
        seed_folder = f'{model_folder}-seed{seed}'
        max_epoch = int(config_env_vars['MAX_EPOCH'])
        check_model_training(seed_folder, max_epoch)

        # all checkpoints evaluated
        checkpoints, target_epochs = \
            get_checkpoints_to_eval(config_env_vars, seed)
        if checkpoints:
            delta = len(target_epochs) - len(checkpoints)
            if delta > 0:
                print(
                    f"[\033[93m{delta}/{len(target_epochs)}\033[0m] "
                    f"{seed_folder}"
                )
            else:
                print(f"[{delta}/{len(target_epochs)}] {seed_folder}")

        else:
            print(
                f"[{len(target_epochs)}/{len(target_epochs)}] "
                f"{seed_folder}"
            )

        # Final model and results
        dec_checkpoint = config_env_vars['DECODING_CHECKPOINT']
        dec_checkpoint = f'{model_folder}-seed{seed}/{dec_checkpoint}'
        if os.path.isfile(dec_checkpoint):
            print(f"[\033[92mdone\033[0m] {dec_checkpoint}")
        else:
            print(f"[pend] {dec_checkpoint}")
    print()


def get_score_from_log(file_path, score_name):

    results = None

    if 'smatch' in score_name:
        regex = smatch_results_re
    else:
        raise Exception(f'Unknown score type {score_name}')

    with open(file_path) as fid:
        for line in fid:
            if regex.match(line):
                results = regex.match(line).groups()
                results = list(map(float, results))
                break

    return results


def get_best_checkpoints(config_env_vars, seed, target_epochs, n_best=5):
    model_folder = config_env_vars['MODEL_FOLDER']
    seed_folder = f'{model_folder}-seed{seed}'
    validation_folder = f'{seed_folder}/epoch_tests/'
    eval_metric = config_env_vars['EVAL_METRIC']
    scores = []
    missing_epochs = []
    for epoch in target_epochs:
        results_file = \
            f'{validation_folder}/dec-checkpoint{epoch}.{eval_metric}'
        if not os.path.isfile(results_file):
            missing_epochs.append(epoch)
            continue
        score = get_score_from_log(results_file, eval_metric)
        # TODO: Support other scores
        scores.append((score[0], epoch))

    sorted_scores = sorted(scores, key=lambda x: x[0])
    best_n_epochs = sorted_scores[-n_best:]
    rest_epochs = sorted_scores[:-n_best]

    best_n_checkpoints = [f'checkpoint{n}.pt' for _, n in best_n_epochs]
    rest_checkpoints = sorted([
        f'{model_folder}/checkpoint{n}.pt' for _, n in rest_epochs
    ])
    best_scores = [s for s, n in best_n_epochs]

    return best_n_checkpoints, best_scores, rest_checkpoints, missing_epochs


def link_best_model(best_n_checkpoints, config_env_vars, seed, nbest):

    # link best model
    model_folder = config_env_vars['MODEL_FOLDER']
    eval_metric = config_env_vars['EVAL_METRIC']
    for n, checkpoint in enumerate(best_n_checkpoints):

        target_best = (f'{model_folder}-seed{seed}/'
                       f'checkpoint_{eval_metric}_best{nbest-n}.pt')
        source_best = checkpoint

        # get current best model (if exists)
        if os.path.islink(target_best):
            current_best = os.path.basename(os.path.realpath(target_best))
        else:
            current_best = None

        # replace link/checkpoint or create a new one
        if os.path.islink(target_best) and current_best != source_best:
            # We created a link before to a worse model, remove it
            os.remove(target_best)
        elif os.path.isfile(target_best):
            # If we ran remove_checkpoints.sh, we replaced the original
            # link by copy of the checkpoint. We dont know if this is the
            # correct checkpoint already
            os.remove(target_best)

        if (
            not os.path.islink(target_best)
            and not os.path.isfile(target_best)
        ):
            os.symlink(source_best, target_best)


def main(args):

    if args.results:
        exit(1)

    config_env_vars = read_config_variables(args.config)
    if args.list_checkpoints_ready_to_eval or args.list_checkpoints_to_eval:

        # List checkpoints that need to be evaluated to complete training. If
        # ready=True list only those checkpoints that exist already
        assert args.seed, "Requires --seed"
        checkpoints, target_epochs = get_checkpoints_to_eval(
            config_env_vars,
            args.seed,
            ready=bool(args.list_checkpoints_ready_to_eval)
        )

        # print checkpoints to be
        for checkpoint in checkpoints:
            print(checkpoint)
            sys.stdout.flush()

        listed_checkpoints = True

    else:

        # print status for this config
        print_status(config_env_vars, args.seed)
        listed_checkpoints = False

    # Link best checkpoints and/or remove the ones evaluated taht are not top-n
    if args.link_best or args.remove:

        if not listed_checkpoints:

            # link best model and/or remove evaluated models that are not the
            # best
            checkpoints, target_epochs = get_checkpoints_to_eval(
                config_env_vars,
                args.seed,
            )
            if checkpoints:
                print('\nThere are {len(checkpoint)} missing checkpoints\n')
                exit(1)

        assert args.seed, "Requires --seed"
        best_n, best_scores, rest_checkpoints, missing_epochs = \
            get_best_checkpoints(config_env_vars, args.seed, target_epochs,
                                 n_best=args.nbest)

        # link best model if all results are done
        if missing_epochs == [] and args.link_best:
            link_best_model(best_n, config_env_vars, args.seed,
                            args.nbest)

        # remove checkpoints not amoung the n-best
        for checkpoint in rest_checkpoints:
            if os.path.isfile(checkpoint):
                if not listed_checkpoints:
                    print(f'rm {checkpoint}')
                os.remove(checkpoint)


if __name__ == '__main__':
    main(argument_parser())
