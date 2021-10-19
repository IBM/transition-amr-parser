import argparse
from glob import glob
import os
from collections import defaultdict
import json
import re
# pip install python-dateutil
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
from transition_amr_parser.io import read_config_variables
from ipdb import set_trace


def get_vectors(items, label, admit_none=False):

    def x_key(item):
        return int(item['epoch'])

    def y_reduce(items):
        if admit_none:
            return np.mean([
                float(x[label]) for x in items if x[label] is not None
            ])
        else:
            return np.mean([float(x[label]) for x in items])

    # Cluster x-axis
    x_clusters = defaultdict(list)
    for item in items:
        x_clusters[x_key(item)].append(item)
    # get xy vectors
    x = sorted(x_clusters.keys())
    y = [y_reduce(x_clusters[x_i]) for x_i in x]

    return x, y


def get_score_from_log(file_path, score_name):

    smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')

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


def read_experiment(config):

    train_info_regex = re.compile(
        r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) '
        r'\| INFO \| train \| (.*)'
    )
    valid_info_regex = re.compile(
        r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) '
        r'\| INFO \| valid \| (.*)'
    )

    config_name = os.path.basename(config)
    config_env_vars = read_config_variables(config)
    model_folder = config_env_vars['MODEL_FOLDER']
    seeds = config_env_vars['SEEDS'].split()

    exp_data = []
    for seed in seeds:
        seed_folder = f'{model_folder}-seed{seed}'

        # read info from logs
        for log_file in glob(f'{seed_folder}/tr-*.stdout'):
            with open(log_file) as fid:
                for line in fid:
                    if train_info_regex.match(line):
                        date_str, json_info = \
                            train_info_regex.match(line).groups()
                        item = json.loads(json_info)
                        item['timestamp'] = parse(date_str)
                        item['experiment_key'] = config_name
                        item['set'] = 'train'
                        item['name'] = config_name
                        exp_data.append(item)

                    elif valid_info_regex.match(line):
                        date_str, json_info = \
                            valid_info_regex.match(line).groups()
                        item = json.loads(json_info)
                        item['timestamp'] = parse(date_str)
                        item['experiment_key'] = config_name
                        item['set'] = 'valid'
                        item['name'] = config_name
                        exp_data.append(item)

        # read validation decoding scores
        eval_metric = config_env_vars['EVAL_METRIC']
        validation_folder = f'{seed_folder}/epoch_tests/'
        for epoch in range(int(config_env_vars['MAX_EPOCH'])):
            results_file = \
                f'{validation_folder}/dec-checkpoint{epoch}.{eval_metric}'
            if os.path.isfile(results_file):
                score = get_score_from_log(results_file, eval_metric)[0]
                exp_data.append({
                    'epoch': epoch,
                    'set': 'valid-dec',
                    'score': score,
                    'experiment_key': config_name,
                    'name': config_name
                })

    return exp_data


def matplotlib_render(plotting_data, out_png, title):

    # plot in matplotlib
    plt.figure(figsize=(10, 10))
    # axis with extra space for legend
    ax = plt.axes([0.1, 0.1, 0.8, 0.7])
    # second axis for Smatch
    ax_smatch = ax.twinx()
    colors = ['b', 'r', 'g']
    tags = sorted(plotting_data.keys())
    handles = []
    for i in range(len(tags)):

        color = colors[i % len(colors)]

        # train loss
        x, y = plotting_data[tags[i]]['train']
        h = ax.plot(x, y, color)[0]
        handles.append(h)

        # dev loss
        x, y = plotting_data[tags[i]]['valid']
        ax.plot(x, y, '--' + color)[0]

        # dev decoding score
        x, y = plotting_data[tags[i]]['valid-dec']
        ax_smatch.plot(x, y, color)[0]
        ax_smatch.set(ylim=(80, 85))

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax_smatch.set_ylabel('Smatch')

    plt.legend(handles, tags, bbox_to_anchor=(0, 1, 1, 0))
    if title:
        plt.title(title)
    if out_png:
        plt.savefig(out_png)
    else:
        plt.show()


def main(args):

    data = []
    for config in args.in_configs:
        data.extend(read_experiment(config))

    # Cluster by experiment
    experiments = defaultdict(list)
    for item in data:
        experiments[item['experiment_key']].append(item)

    # For each experiment collect separate data for train, valid and score
    # aggregate stats for multiple seeds and produce vectors for later
    # plotting
    plotting_data = defaultdict(dict)
    for exp_tag, exp_data in experiments.items():
        print(f'Collecting data for {exp_tag}')
        for sset in ['train', 'valid']:
            valid_data = [x for x in exp_data if x['set'] == sset]
            plotting_data[exp_tag][sset] = \
                get_vectors(valid_data, f'{sset}_loss')
        sset = 'valid-dec'
        score_data = [x for x in exp_data if x['set'] == sset]
        plotting_data[exp_tag][sset] = \
            get_vectors(score_data, 'score', admit_none=True)

    # Render picture in matplotlib
    matplotlib_render(plotting_data, args.out_png, args.title)


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR results plotter')
    # Single input parameters
    parser.add_argument(
        'in_configs',
        nargs='+',
        help="One or more config fils",
        type=str,
    )
    parser.add_argument(
        '--title',
        help="Title of plot"
    )

    parser.add_argument(
        '-o', '--out-png',
        help="Save into a file instead of plotting"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
