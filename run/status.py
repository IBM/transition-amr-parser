import sys
import shutil
from time import sleep
import numpy as np
from glob import glob
import signal
import re
import os
from datetime import datetime
import argparse
from collections import defaultdict, Counter
from statistics import mean
from transition_amr_parser.io import read_config_variables
from transition_amr_parser.clbar import clbar, yellow_font
from fairseq_ext.utils import remove_optimizer_state
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
        "--test",
        help="Show test results (if available)",
        action='store_true',
    )
    parser.add_argument(
        "--results",
        help="print results for all complete models",
        action='store_true',
    )
    parser.add_argument(
        "--decimals",
        help="number of decimals shown with --results",
        default=1,
        type=int
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
        "--configs",
        nargs='+',
        help="select multiple experiments by config",
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
        "--wait-finished",
        help="Print status until final model created",
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
        "--final-remove",
        help="Remove all but final checkpoint, remove also optimizer",
        action='store_true'
    )
    parser.add_argument(
        "--remove-features",
        help="Remove features",
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
    parser.add_argument(
        "--wait-checkpoint-ready-to-eval",
        help="Wait 10 seconds to check if there is a checkpoint pending to "
             "eval, return path if it exists.",
        action='store_true'
    )
    parser.add_argument(
        "--clear",
        help="Clear screen before printing status",
        action='store_true'
    )
    parser.add_argument(
        "--ignore-missing-checkpoints",
        help="When linking best checkpoints, ignore missing ones",
        action='store_true'
    )
    args = parser.parse_args()
    return args


def check_model_training(seed_folder, max_epoch, is_done):

    diplay_lines = []
    final_checkpoint = f'{seed_folder}/checkpoint{max_epoch}.pt'
    if os.path.isfile(final_checkpoint) or is_done:
        # Last epoch completed
        diplay_lines.append(
            (f"\033[92m{max_epoch}/{max_epoch}\033[0m", f"{seed_folder}")
        )
    else:
        # Get which epochs are completed
        epochs = []
        for checkpoint in glob(f'{seed_folder}/checkpoint*.pt'):
            fetch = checkpoint_re.match(checkpoint)
            if fetch:
                epochs.append(int(fetch.groups()[0]))
        if epochs:
            curr_epoch = max(epochs)
            diplay_lines.append(
                (f"\033[93m{curr_epoch}/{max_epoch}\033[0m", f"{seed_folder}")
            )
        else:
            curr_epoch = 0
            diplay_lines.append(
                (f"{curr_epoch}/{max_epoch}", f"{seed_folder}")
            )

    return diplay_lines


def read_results(seed_folder, eval_metric, target_epochs, warnings=True):

    val_result_re = re.compile(r'.*de[cv]-checkpoint([0-9]+)\.' + eval_metric)
    validation_folder = f'{seed_folder}/epoch_tests/'
    epochs = []
    faulty_scores = []
    for result in glob(f'{validation_folder}/*.{eval_metric}'):
        fetch = val_result_re.match(result)
        if fetch:
            epochs.append(int(fetch.groups()[0]))
            if os.stat(result).st_size == 0:
                faulty_scores.append(result)
    missing_epochs = set(target_epochs) - set(epochs)
    missing_epochs = sorted(missing_epochs, reverse=True)

    # Warn about faulty scores
    if faulty_scores and warnings:
        print(f'\033[93mWARNING: empty {eval_metric} file(s)\033[0m')
        for faulty in faulty_scores:
            print(faulty)
        print()

    return target_epochs, missing_epochs


def get_checkpoints_to_eval(config_env_vars, seed, ready=False, warnings=True):
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
        seed_folder, eval_metric, target_epochs, warnings=warnings
    )

    # construct paths
    checkpoints = []
    epochs = []
    for epoch in missing_epochs:
        checkpoint = f'{seed_folder}/checkpoint{epoch}.pt'
        if os.path.isfile(checkpoint) or not ready:
            checkpoints.append(os.path.realpath(checkpoint))
            epochs.append(epoch)

    return checkpoints, target_epochs, missing_epochs


def check_checkpoint_evaluation(config_env_vars, seed, seed_folder):

    checkpoints, target_epochs, _ = \
        get_checkpoints_to_eval(config_env_vars, seed)
    if checkpoints:
        delta = len(target_epochs) - len(checkpoints)
        if delta > 0:
            return (
                f"\033[93m{delta}/{len(target_epochs)}\033[0m",
                f"{seed_folder}/epoch_tests"
            ), False
        else:
            return (
                f"{delta}/{len(target_epochs)}",
                f"{seed_folder}/epoch_tests"
            ), False

    else:
        return (
            f"\033[92m{len(target_epochs)}/{len(target_epochs)}\033[0m",
            f"{seed_folder}/epoch_tests"
        ), True


def get_corrupted_checkpoints(seed_folder):

    # check for corrupted models
    checkpoints_by_size = defaultdict(list)
    for checkpoint in glob(f'{seed_folder}/*.pt'):
        if not os.path.islink(checkpoint):
            size = int(os.stat(checkpoint).st_size/1024)
            checkpoints_by_size[size].append(checkpoint)

    size_count = Counter(checkpoints_by_size.keys())
    if list(size_count.keys()):
        normal_size = max(list(size_count.keys()))
        corrupted_checkpoints = []
        for size, checkpoints in checkpoints_by_size.items():
            if size < 0.5 * float(normal_size):
                corrupted_checkpoints.extend(checkpoints)
        return corrupted_checkpoints
    else:
        return []


def print_status(config_env_vars, seed, do_clear=False, warnings=True):

    # Inform about completed stages
    # pre-training ones
    status_lines = []
    for variable in ['ALIGNED_FOLDER', 'ORACLE_FOLDER', 'EMB_FOLDER',
                     'DATA_FOLDER']:
        step_folder = config_env_vars[variable]
        if os.path.isfile(f'{step_folder}/.done'):
            status_lines.append((f"\033[92mdone\033[0m", f"{step_folder}"))
        elif os.path.isdir(step_folder):
            status_lines.append((f"\033[93mpart\033[0m", f"{step_folder}"))
        else:
            status_lines.append((f"pend", f"{step_folder}"))

    # training/eval ones
    model_folder = config_env_vars['MODEL_FOLDER']
    if seed is None:
        seeds = config_env_vars['SEEDS'].split()
    else:
        assert seed in config_env_vars['SEEDS'].split(), \
            "{seed} is not a trained seed for the model"
        seeds = [seed]
    # loop over each model with a different random seed
    finished = {}
    corrupted_checkpoints = []
    for seed in seeds:

        # default unfinished
        finished[seed] = False
        seed_folder = f'{model_folder}-seed{seed}'
        max_epoch = int(config_env_vars['MAX_EPOCH'])

        # find checkpoints with suspiciously smaller sizes
        corrupted_checkpoints.extend(get_corrupted_checkpoints(seed_folder))

        # all checkpoints evaluated
        line, is_done = check_checkpoint_evaluation(
            config_env_vars, seed, seed_folder
        )
        status_lines.append(line)

        # all checkpoints trained (or evaluated)
        status_lines.extend(
            check_model_training(seed_folder, max_epoch, is_done)
        )

        # Final model and results
        dec_checkpoint = config_env_vars['DECODING_CHECKPOINT']
        beam_size = config_env_vars['BEAM_SIZE']
        eval_metric = config_env_vars['EVAL_METRIC']
        # valid_checkpoint_wiki.smatch_top5-avg.pt
        dec_final_result = (
            f'{model_folder}-seed{seed}/beam{beam_size}/'
            f'valid_{dec_checkpoint}.{eval_metric}'
        )
        dec_checkpoint = f'{model_folder}-seed{seed}/{dec_checkpoint}'
        if os.path.isfile(dec_final_result):
            finished[seed] = True
            status_lines.append(
                (f"\033[92mdone\033[0m", f"{dec_final_result}")
            )
        else:
            status_lines.append((f"pend", f"{dec_final_result}"))

    # format lines to avoid overflowing command line size
    ncol, _ = shutil.get_terminal_size((80, 20))
    col1_width = max(len_print(x[0]) for x in status_lines) + 2
    new_statues_lines = []
    for (col1, col2) in status_lines:
        delta = col1_width + 2 + len(col2) - ncol
        # correction for scape symbols
        delta_cl = len(col1) - len_print(col1)
        if delta_cl > 0:
            width = col1_width + delta_cl
        else:
            width = col1_width

        if delta > 0:
            half_delta = delta // 2 + 4
            half_col2 = len(col2) // 2
            col2_crop = col2[:half_col2 - half_delta]
            col2_crop += ' ... '
            col2_crop += col2[half_col2 + half_delta:]
            new_statues_lines.append(f'[{col1:^{width}}] {col2_crop}')
        else:
            new_statues_lines.append(f'[{col1:^{width}}] {col2}')

    # print
    if do_clear:
        os.system('clear')
    if corrupted_checkpoints and warnings:
        print()
        print(f"\033[91mWARNING: Small checkpoints, corrupted?\033[0m")
        for ch in corrupted_checkpoints:
            print(ch)
        print()
    print('\n'.join(new_statues_lines))
    print()

    return all(finished.values())


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
    rest_checkpoints = []

    for epoch in range(int(config_env_vars['MAX_EPOCH'])):

        # store paths of checkpoint that wont need to be evaluated for deletion
        checkpoint_file = f'{seed_folder}/checkpoint{epoch}.pt'
        if epoch not in target_epochs:
            if os.path.isfile(checkpoint_file):
                rest_checkpoints.append(checkpoint_file)
            else:
                continue

        results_file = \
            f'{validation_folder}/dec-checkpoint{epoch}.{eval_metric}'
        if not os.path.isfile(results_file):
            missing_epochs.append(epoch)
            continue
        elif os.stat(results_file).st_size == 0:
            # errors may have produced an empty score file
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
    if sorted_scores:
        rest_checkpoints += sorted([
            f'{seed_folder}/checkpoint{n}.pt' for _, n in rest_epochs
        ])
    else:
        # did not start yet to score any model, better keep last checkpoint.
        # not that we delete it midway through a copy to last_checkpoint.pt
        rest_checkpoints = rest_checkpoints[:-1]

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


def average_results(results, fields, average_fields, ignore_fields,
                    concatenate_fields):

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
            # ignore everything after space
            field = field.split()[0]
            if field in average_fields:
                samples = [r[field] for r in sresults if r[field] is not None]
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


def extract_experiment_data(config_env_vars, seed, do_test):

    model_folder = config_env_vars['MODEL_FOLDER']
    seed_folder = f'{model_folder}-seed{seed}'

    # Get speed stats
    minutes_per_epoch, minutes_per_test = \
        get_speed_statistics(seed_folder)
    max_epoch = int(config_env_vars['MAX_EPOCH'])
    if minutes_per_epoch and minutes_per_epoch > 1:
        epoch_time = minutes_per_epoch/60.*max_epoch
    else:
        epoch_time = None
    if minutes_per_test and minutes_per_test > 1:
        test_time = minutes_per_test
    else:
        test_time = None

    # get dev results for each tested epoch
    _, target_epochs, _ = get_checkpoints_to_eval(
        config_env_vars,
        seed,
        ready=True
    )
    checkpoints, scores, _, missing_epochs, sorted_scores = \
        get_best_checkpoints(
            config_env_vars, seed, target_epochs, n_best=5
        )
    if scores == []:
        return {}
    # select best result and epoch
    best_checkpoint, best_score = sorted(
        zip(checkpoints, scores), key=lambda x: x[1]
    )[-1]
    max_epoch = config_env_vars['MAX_EPOCH']
    best_epoch = checkpoint_re.match(best_checkpoint).groups()[0]

    # get top-5 beam result
    # TODO: More granularity here. We may want to add many different
    # metrics and sets
    eval_metric = config_env_vars['EVAL_METRIC']
    sset = 'valid'
    cname = 'checkpoint_wiki.smatch_top5-avg'
    # beam 1
    results_file = \
        f'{seed_folder}/beam1/{sset}_{cname}.pt.{eval_metric}'
    if os.path.isfile(results_file):
        best_top5_score = get_score_from_log(results_file,
                                             eval_metric)[0]
    else:
        best_top5_score = None
    # beam 10
    results_file = \
        f'{seed_folder}/beam10/{sset}_{cname}.pt.{eval_metric}'
    if os.path.isfile(results_file):
        best_top5_beam10_score = get_score_from_log(results_file,
                                                    eval_metric)[0]
    else:
        best_top5_beam10_score = None

    # Append result
    result = dict(
        model_folder=model_folder,
        seed=seed,
        data=config_env_vars['TASK_TAG'],
        oracle=os.path.basename(config_env_vars['ORACLE_FOLDER'][:-1]),
        features=os.path.basename(config_env_vars['EMB_FOLDER']),
        model=config_env_vars['TASK'] + f':{seed}',
        best=f'{best_epoch}/{max_epoch}',
        dev=best_score,
        top5_beam10=best_top5_beam10_score,
        top5_beam1=best_top5_score,
        train=epoch_time,
        dec=test_time,
        sorted_scores=sorted_scores
    )

    if '_CONFIG_PATH' in config_env_vars:
        result['config_path'] = config_env_vars['_CONFIG_PATH']

    if do_test:
        sset = 'test'
        cname = 'checkpoint_wiki.smatch_top5-avg'
        # beam 1
        results_file = \
            f'{seed_folder}/beam1/{sset}_{cname}.pt.{eval_metric}'
        if os.path.isfile(results_file):
            best_top5_score_test = get_score_from_log(results_file,
                                                      eval_metric)[0]
        else:
            best_top5_score_test = None
        result['test_top5_beam1'] = best_top5_score_test
        # beam 10
        results_file = \
            f'{seed_folder}/beam10/{sset}_{cname}.pt.{eval_metric}'
        if os.path.isfile(results_file):
            best_top5_beam10_test = get_score_from_log(results_file,
                                                       eval_metric)[0]
        else:
            best_top5_beam10_test = None
        result['test_top5_beam10'] = best_top5_beam10_test

    return result


def get_experiment_configs(models_folder, configs, set_seed):

    # Collect paths of all experiment folders for different seeds, either by
    # config or just looking into the models folder
    config_exps = []
    if configs:

        # from configs
        for config in configs:
            config_env_vars = read_config_variables(config)
            model_folder = config_env_vars['MODEL_FOLDER']
            for seed in config_env_vars['SEEDS'].split():
                if set_seed and set_seed != seed:
                    continue
                config_env_vars['_CONFIG_PATH'] = config
                config_exps.append((config_env_vars, seed))

    else:

        model_folders = glob(f'{models_folder}/*/*')

        # from DATA folder
        for model_folder in model_folders:

            # check for backwars compatibility, depth two foldler structure
            if re.match('.*-seed([0-9]+)', model_folder):
                # set_trace(context=30)
                seed_folders = [model_folder]
            else:
                seed_folders = glob(f'{model_folder}/*')

            for seed_folder in seed_folders:
                # if config given, identify it by seed
                if set_seed and f'seed{set_seed}' not in seed_folder:
                    continue
                else:
                    seed = re.match('.*-seed([0-9]+)', seed_folder).groups()[0]
                config = f'{seed_folder}/config.sh'
                config_env_vars = read_config_variables(config)
                # if os.path.islink(config):
                #    config_env_vars['config_path'] = \
                #        f'configs/{os.path.basename(os.readlink(config))}'
                config_exps.append((config_env_vars, seed))

    return config_exps


def display_results(models_folder, configs, set_seed, seed_average, do_test,
                    longr=False, do_clear=False, decimals=1, show_config=True):

    # determine numeric_fields
    numeric_fields = [
        'dev', 'top5_beam10', 'top5_beam1', 'train (h)', 'dec (m)'
    ]
    if do_test:
        numeric_fields.extend(['test_top5_beam1', 'test_top5_beam10'])

    # collect data for each experiment as a dictionary
    results = []
    # set_trace(context=30)
    for conf, seed in get_experiment_configs(models_folder, configs, set_seed):
        result = extract_experiment_data(conf, seed, do_test)
        if result:
            results.append(result)

    if configs or (show_config and all('config_path' in r for r in results)):
        fields = ['config_path', 'best']
    else:
        fields = ['data', 'oracle', 'features', 'model', 'best']
    fields.extend(numeric_fields)

    # TODO: average over seeds
    if seed_average:
        ignore_fields = ['best']
        concatenate_fields = ['seed']
        results = average_results(results, fields, numeric_fields,
                                  ignore_fields, concatenate_fields)

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
        formatter = {
            x: ('{:.' + str(decimals) + 'f}').format
            for x in numeric_fields
        }
        print_table(fields, results, formatter=formatter, do_clear=do_clear,
                    col0_right=bool(configs))

        if configs and len(configs) == 1 and longr:
            # single model result display
            sorted_scores = results[0]['sorted_scores']
            minc = .95 * min([x[0] for x in sorted_scores])
            sorted_scores = sorted(sorted_scores, key=lambda x: x[1])
            pairs = [(str(x), y) for (y, x) in sorted_scores]
            clbar(pairs, ylim=(minc, None), ncol=79, yform='{:.4f}'.format)
            print()


def len_print(string):
    if string is None:
        return 0
    else:
        bash_scape = re.compile(r'\x1b\[\d+m|\x1b\[0m')
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

    return str(cell)


def print_table(header, data, formatter, do_clear=False, col0_right=False):

    # data structure checks

    # find largest elemend per column
    max_col_size = []
    for n, field in enumerate(header):
        row_lens = [len(field)]
        for row in data:
            cell = get_cell_str(row, field, formatter.get(field, None))
            row_lens.append(len_print(cell))
        max_col_size.append(max(row_lens))

    # format and print
    if do_clear:
        os.system('clear')
    print('')
    col_sep = ' '
    row_str = ['{:^{width}}'.format(h, width=max_col_size[n])
               for n, h in enumerate(header)]
    print(col_sep.join(row_str))
    for row in data:
        row_str = []
        for n, field in enumerate(header):
            cell = get_cell_str(row, field, formatter.get(field, None))
            if col0_right and n == 0:
                row_str.append(
                    '{:<{width}}'.format(cell, width=max_col_size[n])
                )
            else:
                row_str.append(
                    '{:^{width}}'.format(cell, width=max_col_size[n])
                )
        print(col_sep.join(row_str))
    print('')


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    # exit with error to stop other scripts coming afterwards
    exit(1)


def link_remove(args, seed, config_env_vars, ignore_missing_checkpoints=False,
                checkpoints=None, target_epochs=None):

    # in script usage mode we cant have stdout
    warnings = True
    if (
        bool(args.list_checkpoints_ready_to_eval) or
        bool(args.list_checkpoints_to_eval)
    ):
        warnings = False

    # List checkpoints that need to be evaluated to complete training. If
    # ready=True list only those checkpoints that exist already
    if checkpoints is None:
        checkpoints, target_epochs, _ = get_checkpoints_to_eval(
            config_env_vars,
            seed,
            ready=bool(args.list_checkpoints_ready_to_eval),
            warnings=warnings
        )

    # get checkpoints that still need to be created, those scored and those
    # deletable
    # TODO: Unify with code above
    best_n, best_scores, rest_checkpoints, missing_epochs, _ = \
        get_best_checkpoints(config_env_vars, seed, target_epochs,
                             n_best=args.nbest)

    missing_checkpoints = [f'checkpoint{n}.pt' for n in best_n]
    if (
        args.link_best
        and ignore_missing_checkpoints
        and bool(set(best_n) & set(missing_checkpoints))
    ):
        print(set(best_n) & set(missing_checkpoints))
        raise Exception(
            '--link-best --ignore-missing-checkpoints can not be used if any '
            'n-best is missing'
        )

    # link best model if all results are done
    if (
        (missing_epochs == [] or ignore_missing_checkpoints)
        and args.link_best
    ):
        link_best_model(best_n, config_env_vars, seed, args.nbest)
        if ignore_missing_checkpoints and missing_epochs:
            print(missing_epochs)
            print(yellow_font('WARNING: --link-best with missing checkpoints'))

    # remove checkpoints not among the n-best
    for checkpoint in rest_checkpoints:
        if os.path.isfile(checkpoint):
            if not (
                bool(args.list_checkpoints_ready_to_eval) or
                bool(args.list_checkpoints_to_eval)
            ):
                print(f'rm {checkpoint}')
            os.remove(checkpoint)


def wait_checkpoint_ready_to_eval(args):

    assert bool(args.config), "Missing config"

    config_env_vars = read_config_variables(args.config)
    if args.seed:
        seeds = [args.seed]
    else:
        seeds = config_env_vars['SEEDS'].split()
    # eval_init_epoch = int(config_env_vars['EVAL_INIT_EPOCH'])
    # TODO: Clearer naming
    checkpoints = []
    need_eval = []
    while not checkpoints:
        checkpoints = []
        need_eval = []
        for seed in seeds:
            scheckpoints, starget_epochs, sneed_eval = \
                get_checkpoints_to_eval(
                    config_env_vars,
                    seed,
                    ready=True,
                    warnings=False
                 )

            # sanity check: we did not delete checkpoints without testing them
            deleted_epochs = [
                e for e in sneed_eval if e not in starget_epochs
            ]
            if deleted_epochs:

                model_folder = config_env_vars['MODEL_FOLDER']
                seed_folder = f'{model_folder}-seed{seed}'
                print('\nCheckpoints may have been deleted before testing or '
                      'testing failed on evaluation, missing\n')
                for epoch in deleted_epochs:
                    print(f'{seed_folder}/checkpoint{epoch}.pt')
                exit(1)

            checkpoints.extend(scheckpoints)
            need_eval.extend(sneed_eval)

            # link and/or remove checkpoints
            if args.link_best or args.remove:
                link_remove(args, seed,
                            config_env_vars, checkpoints=scheckpoints,
                            target_epochs=starget_epochs)

        if need_eval == []:
            print('Finished!')
            break

        print_status(config_env_vars, None, do_clear=args.clear)
        print(
            f'Waiting for checkpoint to evaluate'
            ' (if you stop this script, I wont evaluate)'
        )
        sleep(10)


def final_remove(seed, config_env_vars, remove_optimizer=True,
                 remove_features=False):
    '''
    Remove all but the final trained model file DEC_CHECKPOINT and best metric
    '''

    model_folder = config_env_vars['MODEL_FOLDER']
    # eval_metric = config_env_vars['EVAL_METRIC']
    dec_checkpoint = config_env_vars['DECODING_CHECKPOINT']
    seed_folder = f'{model_folder}-seed{seed}'
    dec_checkpoint = f'{seed_folder}/{dec_checkpoint}'
    # target_best = f'{seed_folder}/checkpoint_{eval_metric}_best1.pt'

    # check the final models exist
    # if (
    #     not os.path.islink(target_best)
    #     or not os.path.isfile(os.path.realpath(target_best))
    # ):
    #     print(f'Can not --final-remove, missing {target_best}')
    #     return
    # else:
    #     best_metric_checkpoint = os.path.realpath(target_best)
    #     best_metric_checkpoint_link = target_best

    do_not_remove = [
        dec_checkpoint,  # best_metric_checkpoint, best_metric_checkpoint_link
    ]

    if not os.path.isfile(os.path.realpath(dec_checkpoint)):
        print('Can not --final-remove, missing {dec_checkpoint}')
        return
    else:
        dec_checkpoint = os.path.realpath(dec_checkpoint)

    # remove optimizer from final checkpoint
    if remove_optimizer:
        remove_optimizer_state(dec_checkpoint)

    # remove all other checkpoints
    for checkpoint in glob(f'{seed_folder}/*.pt'):
        if (
            os.path.realpath(checkpoint) not in do_not_remove
        ):
            print(f'rm {checkpoint}')
            os.remove(checkpoint)

    # remove shared features for this model
    # this willl also affect other models with same features!
    if remove_features:
        remove_features(config_env_vars)


def remove_features(config_env_vars):

    # also remove features
    feature_folder = config_env_vars['DATA_FOLDER']
    for dfile in glob(f'{feature_folder}/*'):
        print(f'rm {dfile}')
        os.remove(dfile)
    if os.path.isfile(f'{feature_folder}/.done'):
        print(f'rm {feature_folder}/.done')
        os.remove(f'{feature_folder}/.done')
    if os.path.isdir(feature_folder):
        print(f'rm {feature_folder}/')
        os.rmdir(feature_folder)


def main(args):

    # set ordered exit
    signal.signal(signal.SIGINT, ordered_exit)
    signal.signal(signal.SIGTERM, ordered_exit)

    if args.final_remove:

        assert args.config, "Needs --config (optional --seed)"

        # print status for this config
        config_env_vars = read_config_variables(args.config)
        if args.seed:
            seeds = [args.seed]
        else:
            seeds = config_env_vars['SEEDS'].split()

        # remove checkpoints
        for seed in seeds:
            link_remove(args, seed, config_env_vars)
            final_remove(seed, config_env_vars)

    elif args.remove_features:

        assert args.config, "Needs config"

        remove_features(read_config_variables(args.config))

    elif args.results or args.long_results:

        # results display and exit
        display_results('DATA/*/models/', args.configs, args.seed,
                        args.seed_average, args.test,
                        longr=bool(args.long_results),
                        do_clear=args.clear, decimals=args.decimals)

    elif args.wait_checkpoint_ready_to_eval:

        # wait until a checkpoint to evaluate is avaliable, inform of status in
        # the meanwhile. Optionally delete checkpoints that are evaluated or do
        # not need to be evaluated
        wait_checkpoint_ready_to_eval(args)

    elif args.list_checkpoints_ready_to_eval or args.list_checkpoints_to_eval:

        # List checkpoints that need to be evaluated to complete training. If
        # ready=True list only those checkpoints that exist already
        assert args.seed, "Requires --seed"
        config_env_vars = read_config_variables(args.config)
        checkpoints, target_epochs, _ = get_checkpoints_to_eval(
            config_env_vars,
            args.seed,
            ready=bool(args.list_checkpoints_ready_to_eval),
            warnings=False
        )

        # link and/or remove checkpoints
        if args.link_best or args.remove:
            link_remove(args, args.seed, config_env_vars,
                        args.ignore_missing_checkpoints)

        # print checkpoints to be evaluated
        for checkpoint in checkpoints:
            print(checkpoint)
            sys.stdout.flush()

    else:

        # print status for this config
        if args.config is None:
            print('\nSpecify a config with -c or use --results\n')
            exit(1)
        config_env_vars = read_config_variables(args.config)

        if args.seed:
            seeds = [args.seed]
        else:
            seeds = config_env_vars['SEEDS'].split()

        while True:
            fin = print_status(config_env_vars, args.seed, do_clear=args.clear)
            # link and/or remove checkpoints
            if args.link_best or args.remove:
                for seed in seeds:
                    link_remove(args, seed, config_env_vars)
            # exit if finished
            if not args.wait_finished or fin:
                break
            sleep(10)


if __name__ == '__main__':
    main(argument_parser())
