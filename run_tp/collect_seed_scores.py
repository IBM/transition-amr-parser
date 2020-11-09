import os
import re
import argparse
import itertools

import numpy as np

from collect_scores import collect_final_scores, SaveAndPrint


# model checkpoint dir name reges
checkpoint_dir_re = re.compile(r'models_ep([0-9]+)_seed([0-9]+)')

# beam results dir name reges
beam_dir_re = re.compile(r'beam([0-9]+)')

# results file name regex
results_re = re.compile(r'(.+)_checkpoint_(.+?)\.(.+)')  # +? for non-greedy matching; only match the first "\."
# smatch_re = re.compile(r'valid_checkpoint([0-9]+)\.smatch')
# smatch_re_wiki = re.compile(r'valid_heckpoint([0-9]+)\.wiki\.smatch')

# model names to consider
models = ['last', 'wiki-smatch_best1', 'wiki-smatch_top3-avg', 'wiki-smatch_top5-avg']
beam_sizes = [1, 5, 10]
epoch = 120
seeds = [42, 0, 315]

# models = ['wiki-smatch_top5-avg']
# beam_sizes = [10]

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def blue(string):
    return "\033[94m%s\033[0m" % string


def parse_args():
    parser = argparse.ArgumentParser(description='Collect model results over different random seeds')
    parser.add_argument('experiments', type=str, default='/dccstor/jzhou1/work/EXP/exp_debug/',
                        help='folder containing saved model checkpoints for a single training')
    parser.add_argument('--data_sets', type=str, nargs='*', default=['valid', 'test'],
                        help='data sets to collect scores')
    parser.add_argument('--models', type=str, nargs='*', default=models,
                        help='model checkpoint names to collect scores')
    parser.add_argument('--beam_sizes', type=int, nargs='*', default=beam_sizes,
                        help='beam sizes to collect scores')
    parser.add_argument('--epoch', type=int, default=epoch,
                        help='number of training epochs for a model to be considered')
    parser.add_argument('--seeds', type=int, nargs='*', default=seeds,
                        help='experimental seeds to consider score collection')
    parser.add_argument('--score_name', type=str, default='wiki.smatch',
                        help='postfix of the score files')
    parser.add_argument('--ndigits', type=int, default=2,
                        help='number of digits after the decimal point')
    # parser.add_argument('--save_name', type=str, default='collected_wiki-smatch_seed_scores.txt',
    #                     help='save name for the collection of results')
    args = parser.parse_args()
    return args


def collect_final_scores_seeds(experiment_folder, data_sets, models, beam_sizes, epoch, seeds, score_name):
    """Collect scores for a single model configuration with different random seeds to summarize for final report.

    Args:
        experiment_folder ([type]): [description]
        data_sets ([type]): [description]
        models ([type]): [description]
        beam_sizes ([type]): [description]
        seeds ([type]): [description]
        score_name ([type]): [description]
    """
    beam_results_seeds = {}
    for name in os.listdir(experiment_folder):
        if not checkpoint_dir_re.match(name):
            continue

        num_epochs, seed = checkpoint_dir_re.match(name).groups()
        # NOTE matched strings are of type str
        num_epochs = int(num_epochs)
        seed = int(seed)

        if num_epochs != epoch:
            continue

        if seed not in seeds:
            continue

        checkpoint_folder = os.path.join(experiment_folder, name)

        # beam_results is a dictionary, with key levels beam_size -> model_name -> data_set -> score
        beam_results = collect_final_scores(checkpoint_folder, data_sets, models, score_name)
        beam_results_seeds[seed] = beam_results

    # beam_results_seeds is a dictionary, with key levels seed -> beam_size -> model_name -> data_set -> score
    return beam_results_seeds


def adjust_results_dict(beam_results_seeds):
    """Adjusting the results dictionary to have key levels: data_set -> model_name -> beam_size -> seed -> score."""
    results_seeds = {}
    if beam_results_seeds:
        # get the list of all unique values at each key level
        seeds = list(beam_results_seeds.keys())
        beam_sizes = set()
        models = set()
        data_sets = set()
        for beam_dict in beam_results_seeds.values():
            beam_sizes = beam_sizes.union(set(beam_dict.keys()))
            for model_dict in beam_dict.values():
                models = models.union(set(model_dict.keys()))
                for data_dict in model_dict.values():
                    data_sets = data_sets.union(set(data_dict.keys()))
        beam_sizes = sorted(list(beam_sizes))
        models = sorted(list(models))
        data_sets = sorted(list(data_sets))

        # reorder the levels to create new dictionary
        for data_set, model, beam_size, seed in itertools.product(data_sets, models, beam_sizes, seeds):
            results_seeds.setdefault(data_set, {}).setdefault(model, {}).setdefault(beam_size, {})[seed] = \
                beam_results_seeds.get(seed, {}).get(beam_size, {}).get(model, {}).get(data_set)

    return results_seeds, data_sets, models, beam_sizes, seeds


def compute_results_summary(results_seeds, data_sets, models, beam_sizes, seeds, seed_anchor=42):
    """Compute the summary of results, including the average score and std over random seeds and the best score,
    for each model and beam size.
    """
    assert seed_anchor in seeds
    results_summary = {}
    best_beam_scores = {}    # store the best score for each beam, for each data set
    best_beam_models = {}    # store the best model under each beam size, for each data set
    for data_set in data_sets:
        results_summary[data_set] = {}
        best_beam_scores[data_set] = {}
        best_beam_models[data_set] = {}
        for model, beam_size in itertools.product(models, beam_sizes):
            # compute the summary statistics
            seed_scores = np.array([results_seeds[data_set][model][beam_size][s] for s in seeds])
            seed_scores_anchor = results_seeds[data_set][model][beam_size][seed_anchor]
            if None not in seed_scores:
                seed_scores_best = seed_scores.max()
                seed_scores_best_seed = seeds[seed_scores.argmax()]
                seed_scores_avg = seed_scores.mean()
                seed_scores_std = seed_scores.std()
            else:
                # deal with the case when the results are not complete
                seed_scores_anchor = seed_scores_anchor or 0
                seed_scores_best = 0
                seed_scores_best_seed = '-'
                seed_scores_avg = 0
                seed_scores_std = 0
                seed_scores = np.array([ss if ss is not None else 0 for ss in seed_scores])
            # set the summary results
            results_summary[data_set].setdefault((model, beam_size), {})[f'seed{seed_anchor}'] = seed_scores_anchor
            results_summary[data_set].setdefault((model, beam_size), {})['seed_best'] = seed_scores_best
            results_summary[data_set].setdefault((model, beam_size), {})['seed_best_seed'] = seed_scores_best_seed
            results_summary[data_set].setdefault((model, beam_size), {})['seed_avg'] = seed_scores_avg
            results_summary[data_set].setdefault((model, beam_size), {})['seed_std'] = seed_scores_std
            results_summary[data_set].setdefault((model, beam_size), {})['seed_all'] = seed_scores.tolist()

            # update the best results for this beam size: here we use the best scores out of different random seeds
            if best_beam_scores[data_set].setdefault(beam_size, 0) < seed_scores_best:
                best_beam_scores[data_set][beam_size] = seed_scores_best
                best_beam_models[data_set][beam_size] = model

    return results_summary, best_beam_models


def save_results_summary(experiment_folder, results_summary, best_beam_models, data_sets, score_name,
                         seed_anchor=42, ndigits=2, print_to_console=True):
    save_path = os.path.join(experiment_folder, f'summary_{score_name.replace(".", "-")}_scores.txt')
    nspace = 10
    # save and print summary results
    save_and_print = SaveAndPrint(save_path, print_to_console)

    for data in data_sets:
        save_and_print('-' * 50 + f' {data} ' + '-' * 50)
        save_and_print('\n')
        save_and_print('\n')
        save_and_print('model | beam_size' + ' ' * nspace
                       + (' ' * nspace).join([f'seed{seed_anchor}', 'seed_best (seed)', 'seed_avg (all)'])
                       )
        save_and_print('\n')
        for model_beam, results in results_summary[data].items():
            # check whether to highlight the current row if it's the best model under the current beam
            highlight = False
            if model_beam[1] in best_beam_models[data]:
                if model_beam[0] == best_beam_models[data][model_beam[1]]:
                    highlight = True

            # at test data: check whether it is the best model selected by valid data under the current beam
            hightlight_valid_to_test = False
            if data == 'test' and 'valid' in best_beam_models:
                if model_beam[1] in best_beam_models['valid']:
                    if model_beam[0] == best_beam_models['valid'][model_beam[1]]:
                        hightlight_valid_to_test = True

            # print one row of a model and a beam size
            row = (f'{model_beam[0]} | {model_beam[1]}' + ' ' * (17 + nspace - len(model_beam[0]) - 4)
                   + f'{results[f"seed{seed_anchor}"] * 100:.{ndigits}f}'
                   + ' ' * nspace + f'{results["seed_best"] * 100:.{ndigits}f} ({results["seed_best_seed"]})'
                   + ' ' * nspace + f'{results["seed_avg"] * 100:.{ndigits}f}' + ' '
                   + u'\u00B1' + f'{results["seed_std"] * 100:.{ndigits}f}' + ' '
                   + '(' + ', '.join(map(lambda x: f'{x * 100:.{ndigits}f}', results['seed_all'])) + ')'
                   )

            if hightlight_valid_to_test:
                row = row + ' ' + yellow('*')
            if highlight:
                save_and_print(red(row))
            else:
                save_and_print(row)

            save_and_print('\n')
        save_and_print('\n')
    save_and_print(f'(Note: {yellow("*")} marks the model with the best valid score under each beam size, '
                   'on the test set)')
    save_and_print('\n')
    save_and_print('\n')

    save_and_print.close()

    return


if __name__ == '__main__':
    args = parse_args()
    seed_anchor = 42

    if args.score_name != 'wiki.smatch':
        args.models = [m.replace('wiki-smatch', args.score_name.replace('.', '-')) for m in models]

    beam_results_seeds = collect_final_scores_seeds(args.experiments, args.data_sets, args.models, args.beam_sizes,
                                                    args.epoch, args.seeds, args.score_name)
    results_seeds, data_sets, models, beam_sizes, seeds = adjust_results_dict(beam_results_seeds)
    # `beam_sizes` includes all the beam sizes stored, and `args.beam_sizes` are only specified beam sizes
    results_summary, best_beam_models = compute_results_summary(results_seeds, data_sets, models, args.beam_sizes,
                                                                seeds, seed_anchor=seed_anchor)
    save_results_summary(args.experiments, results_summary, best_beam_models, args.data_sets, args.score_name,
                         seed_anchor=seed_anchor,
                         ndigits=args.ndigits,
                         print_to_console=True)

    # print(beam_results_seeds)
    # print(results_seeds)
    # print(u"\u00B1")

    # print(results_summary)

    # breakpoint()
