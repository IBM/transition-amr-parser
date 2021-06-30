import sys
import numpy as np
from glob import glob
import signal
import re
import os
from datetime import datetime
import argparse
from collections import defaultdict
from statistics import mean
from transition_amr_parser.io import read_config_variables, clbar
from ipdb import set_trace


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
        "--long-results",
        help="print results for all complete models, with more info",
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
        "--seed-average",
        help="Average numbers over seeds",
        action='store_true'
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

    results = [None]

    if 'smatch' in score_name:
        regex = smatch_results_re
    else:
        raise Exception(f'Unknown score type {score_name}')

    with open(file_path) as fid:
        for line in fid:
            if regex.match(line):
                results = regex.match(line).groups()
                results = [100*float(x) for x in results]
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
        if score == [None]:
            continue
        # TODO: Support other scores
        scores.append((score[0], epoch))

    sorted_scores = sorted(scores, key=lambda x: x[0])
    best_n_epochs = sorted_scores[-n_best:]
    rest_epochs = sorted_scores[:-n_best]

    best_n_checkpoints = [f'checkpoint{n}.pt' for _, n in best_n_epochs]
    rest_checkpoints = sorted([
        f'{seed_folder}/checkpoint{n}.pt' for _, n in rest_epochs
    ])
    best_scores = [s for s, n in best_n_epochs]

    return (
        best_n_checkpoints, best_scores, rest_checkpoints, missing_epochs,
        sorted_scores
    )


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


def get_average_time_between_write(files):

    timestamps = []
    for dfile in files:
        timestamps.append((
            os.path.basename(dfile),
            datetime.fromtimestamp(os.stat(dfile).st_mtime)
        ))
    timestamps = sorted(timestamps, key=lambda x: x[1])
    deltas = [
        (x[1] - y[1]).seconds / 60.
        for x, y in zip(timestamps[1:], timestamps[:-1])
    ]
    if len(deltas) < 5:
        return None
    else:
        return mean(deltas[2:-2])


def get_speed_statistics(seed_folder):

    files = []
    for checkpoint in glob(f'{seed_folder}/checkpoint*.pt'):
        if checkpoint_re.match(checkpoint):
            files.append(checkpoint)

    minutes_per_epoch = get_average_time_between_write(files)

    files = []
    for checkpoint in glob(f'{seed_folder}/epoch_tests/*.actions'):
        files.append(checkpoint)

    minutes_per_test = get_average_time_between_write(files)

    return minutes_per_epoch, minutes_per_test


def average_results(results, fields):

    average_fields = ['dev', 'top5_beam10', 'train (h)', 'test (m)']
    ignore_fields = ['best']
    concatenate_fields = ['seed']

    # collect
    result_by_seed = defaultdict(list)
    for result in results:
        key = result['model_folder']
        result_by_seed[key].append(result)

    # leave only averages
    averaged_results = []
    for seed, sresults in result_by_seed.items():
        average_result = {}
        for field in fields:
            field = field.split()[0]
            if field in average_fields:
                samples = [r[field] for r in sresults
                           if r[field] is not None]
                if samples:
                    average_result[field] = np.mean(samples)
                    # Add standard deviation
                    average_result[f'{field}-std'] = np.std(samples)
                else:
                    average_result[field] = None

            elif field in ignore_fields:
                average_result[field] = ''
            elif field in concatenate_fields:
                average_result[field] = ','.join([r[field] for r in sresults])
            else:
                average_result[field] = sresults[0][field]
        averaged_results.append(average_result)

    return averaged_results


def display_results(models_folder, config, set_seed, seed_average, longr=False):

    # Table header
    results = []

    if config:
        target_config_env_vars = read_config_variables(config)

    for model_folder in glob(f'{models_folder}/*/*'):
        for seed_folder in glob(f'{model_folder}/*'):

            if set_seed and f'seed{set_seed}' not in seed_folder:
                continue
            else:
                seed = re.match('.*-seed([0-9]+)', seed_folder).groups()[0]

            # Read config contents and seed
            config_env_vars = read_config_variables(f'{seed_folder}/config.sh')

            if (
                config
                and config_env_vars['MODEL_FOLDER']
                != target_config_env_vars['MODEL_FOLDER']
            ):
                continue

            # Get speed stats
            minutes_per_epoch, minutes_per_test = \
                get_speed_statistics(seed_folder)
            max_epoch = int(config_env_vars['MAX_EPOCH'])
            if minutes_per_epoch and minutes_per_epoch > 1:
                epoch_time = f'{minutes_per_epoch/60.*max_epoch:.1f}'
            else:
                epoch_time = 'N/A'
            if minutes_per_test and minutes_per_test > 1:
                test_time = f'{minutes_per_test:.1f}'
            else:
                test_time = 'N/A'

            # get experiments info
            _, target_epochs = get_checkpoints_to_eval(
                config_env_vars,
                seed,
                ready=True
            )
            checkpoints, scores, _, missing_epochs, sorted_scores = \
                get_best_checkpoints(
                    config_env_vars, seed, target_epochs, n_best=5
                )
            if scores == []:
                continue

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
                best_top5_beam10_score = get_score_from_log(results_file,
                                                            eval_metric)[0]
            else:
                best_top5_beam10_score = None

            # Append result
            results.append(dict(
                model_folder=model_folder,
                seed=seed,
                data=config_env_vars['TASK_TAG'],
                oracle=os.path.basename(config_env_vars['ORACLE_FOLDER'][:-1]),
                features=os.path.basename(config_env_vars['EMB_FOLDER']),
                model=config_env_vars['TASK'] + f':{seed}',
                best=f'{best_epoch}/{max_epoch}',
                dev=best_score,
                top5_beam10=best_top5_beam10_score,
                train=epoch_time,
                test=test_time,
            ))

    fields = [
        'data', 'oracle', 'features', 'model', 'best', 'dev',
        'top5_beam10', 'train (h)', 'test (m)'
    ]

    # TODO: average over seeds
    if seed_average:
        results = average_results(results, fields)

    # sort by last row
    sort_field = 'top5_beam10'

    def get_score(x):
        if x[sort_field] is None:
            return -1
        else:
            return float(x[sort_field])
    results = sorted(results, key=get_score)

    # print
    if results:
        assert all(field.split()[0] in results[0].keys() for field in fields)
        formatter = {5: '{:.1f}'.format, 6: '{:.1f}'.format}
        print_table(fields, results, formatter=formatter)

        if config and longr:
            # single model result display
            minc = .95 * min([x[0] for x in sorted_scores])
            sorted_scores = sorted(sorted_scores, key=lambda x: x[1])
            pairs = [(str(x), y) for (y, x) in sorted_scores]
            clbar(pairs, ylim=(minc, None), ncol=79, yform='{:.4f}'.format)
            print()


def len_print(string):
    if string is None:
        return 0
    else:
        bash_scape = re.compile(r'\\x1b\[\d+m|\\x1b\[0m')
        return len(bash_scape.sub('', string))


def get_cell_str(row, field, formatter):
    field2 = field.split()[0]
    cell = row[field2]
    if cell is None:
        cell = ''
    if formatter and cell != '':
        cell = formatter(cell)
    if f'{field2}-std' in row:
        std = row[f'{field2}-std']
        if formatter:
            std = formatter(std)
        cell = f'{cell} ({std})'

    return cell


def print_table(header, data, formatter):

    # data structure checks

    # find largest elemend per column
    max_col_size = []
    for n, field in enumerate(header):
        row_lens = [len(field)]
        for row in data:
            cell = get_cell_str(row, field, formatter.get(n, None))
            row_lens.append(len_print(cell))
        max_col_size.append(max(row_lens))

    # format and print
    print('')
    col_sep = ' '
    row_str = ['{:^{width}}'.format(h, width=max_col_size[n])
               for n, h in enumerate(header)]
    print(col_sep.join(row_str))
    for row in data:
        row_str = []
        for n, field in enumerate(header):
            cell = get_cell_str(row, field, formatter.get(n, None))
            row_str.append('{:^{width}}'.format(cell, width=max_col_size[n]))
        print(col_sep.join(row_str))
    print('')


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def main(args):

    # set orderd exit
    signal.signal(signal.SIGINT, ordered_exit)
    signal.signal(signal.SIGTERM, ordered_exit)

    if args.results:
        display_results('DATA/*/models/', args.config, args.seed,
                        args.seed_average)
        exit(1)
    elif args.long_results:
        display_results('DATA/*/models/', args.config, args.seed,
                        args.seed_average, longr=True)
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
                print(f'\nThere are {len(checkpoints)} missing checkpoints\n')
                exit(1)

        assert args.seed, "Requires --seed"
        best_n, best_scores, rest_checkpoints, missing_epochs, _ = \
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
