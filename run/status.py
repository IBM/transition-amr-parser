import sys
from glob import glob
import subprocess
import re
import os
import argparse
from collections import defaultdict
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
        action='store_true',
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
    # Read all variables of this pattern
    variable_regex = re.compile('^ *([A-Za-z0-9_]+)=.*$')
    # find variables in text and prepare evaluation script
    bash_script = f'source {config_path};'
    with open(config_path) as fid:
        for line in fid:
            if variable_regex.match(line.strip()):
                varname = variable_regex.match(line.strip()).groups()[0]
                bash_script += f'echo "{varname}=${varname}";'
    # Execute script to get variable's value
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


def read_results(seed_folder, eval_metric, target_epochs):

    val_result_re = re.compile(r'.*de[cv]-checkpoint([0-9]+)\.' + eval_metric)
    validation_folder = f'{seed_folder}/epoch_tests/'
    epochs = []
    for result in glob(f'{validation_folder}/*.{eval_metric}'):
        fetch = val_result_re.match(result)
        if fetch:
            epochs.append(int(fetch.groups()[0]))
    missing_epochs = set(target_epochs) - set(epochs)
    missing_epochs = sorted(missing_epochs, reverse=True)

    return target_epochs, missing_epochs

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

    # read results
    target_epochs = list(range(eval_init_epoch, max_epoch+1))
    target_epochs, missing_epochs = read_results(
        seed_folder, eval_metric, target_epochs
    )

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


def display_results(models_folder):

    # Table header
    header = ['data', 'oracle', 'features', 'model', 'best', 'dev', 'dev top5-beam10']
    results = []
    for model_folder in glob('DATA/AMR2.0/models/*/*'):
        for seed_folder in glob(f'{model_folder}/*'):

            # Read config contents and seed
            config_env_vars = read_config_variables(f'{seed_folder}/config.sh')
            seed = re.match('.*-seed([0-9]+)', seed_folder).groups()[0]
    
            # get experiments info
            _, target_epochs = get_checkpoints_to_eval(
                config_env_vars,
                seed,
                ready=True
            )
            checkpoints, scores, _, missing_epochs = get_best_checkpoints(
                config_env_vars, seed, target_epochs, n_best=5
            )
            best_checkpoint, best_score = sorted(
                zip(checkpoints, scores), key=lambda x: x[1]
            )[-1]
            max_epoch = config_env_vars['MAX_EPOCH']
            best_epoch = re.match(
                'checkpoint([0-9]+).pt', best_checkpoint
            ).groups()[0]

            # get top-5 beam result
            # TODO: More granularity here. We may want to add many different
            # metrics and sets
            eval_metric = config_env_vars['EVAL_METRIC']
            sset = 'valid'
            cname = 'checkpoint_wiki.smatch_top5-avg'
            results_file = \
                f'{seed_folder}/beam10/{sset}_{cname}.pt.{eval_metric}'
            if os.path.isfile(results_file):
                best_top5_beam10_score = \
                    get_score_from_log(results_file, eval_metric)[0]
            else:    
                best_top5_beam10_score = ''
    
            # Append result
            results.append([
                config_env_vars['TASK_TAG'],
                os.path.basename(config_env_vars['ORACLE_FOLDER'][:-1]),
                os.path.basename(config_env_vars['EMB_FOLDER']),
                config_env_vars['TASK'],
                f'{best_epoch}/{max_epoch}',
                f'{best_score:.3f}',
                f'{best_top5_beam10_score:.3f}'
            ])

    # TODO: average over seeds

    # sort by last row
    results = sorted(results, key=lambda x: float(x[-2]))

    # print
    print_table([header] + results)


def print_table(rows):

    # data structure checks
    assert isinstance(rows, list)
    assert all(isinstance(row, list) for row in rows)
    assert all(isinstance(item, str) for row in rows for item in row)
    assert len(set([len(row) for row in rows])) == 1

    # find largest elemend per column
    num_col = len(rows[0]) 
    max_col_size = []
    bash_scape = re.compile('\\x1b\[\d+m|\\x1b\[0m')
    for n in range(num_col):
        max_col_size.append(
            max(len(bash_scape.sub('', row[n])) for row in rows)
        )

    # format
    print('')
    col_sep = ' '
    for row in rows:
        row_str = []
        for n, cell in enumerate(row):
            row_str.append('{:^{width}}'.format(cell, width=max_col_size[n]))
        print(col_sep.join(row_str))
    print('')

def main(args):

    if args.results:
        display_results('DATA/AMR2.0/models/')
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
