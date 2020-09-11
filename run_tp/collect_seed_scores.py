import os
import re
import argparse
import itertools

from collect_scores import collect_final_scores


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
seeds = [42, 43, 44]

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


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
    """Adjusting the results dictionary to have key levels: data_set -> (model_name, beam_size) -> seed -> score."""
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

    return results_seeds


if __name__ == '__main__':
    args = parse_args()

    beam_results_seeds = collect_final_scores_seeds(args.experiments, args.data_sets, args.models, args.beam_sizes,
                               args.epoch, args.seeds, args.score_name)
    results_seeds = adjust_results_dict(beam_results_seeds)

    print(beam_results_seeds)
    print(results_seeds)
    print(u"\u00B1")
