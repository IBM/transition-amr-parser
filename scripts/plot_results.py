import argparse
from collections import defaultdict
import re
# pip install python-dateutil
import numpy as np
import matplotlib.pyplot as plt
from transition_amr_parser.io import read_experiment
# from ipdb import set_trace


def get_vectors(items, label, admit_none=False):

    def x_key(item):
        return int(item['epoch'])

    def y_reduce(items):
        if admit_none:
            vy = [float(x[label]) for x in items if x[label] is not None]
        else:
            vy = [float(x[label]) for x in items]
        return np.mean(vy)

    def get_y_std(items):
        if admit_none:
            vy = [float(x[label]) for x in items if x[label] is not None]
        else:
            vy = [float(x[label]) for x in items]
        return np.std(vy)

    # Cluster x-axis
    x_clusters = defaultdict(list)
    for item in items:
        x_clusters[x_key(item)].append(item)
    # get xy vectors
    x = np.array(sorted(x_clusters.keys()))
    y = np.array([y_reduce(x_clusters[x_i]) for x_i in x])
    y_std = np.array([get_y_std(x_clusters[x_i]) for x_i in x])

    return x, y, y_std


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


def matplotlib_render(plotting_data, out_png, title):

    # plot in matplotlib
    plt.figure(figsize=(10, 10))
    # axis with extra space for legend
    ax = plt.axes([0.1, 0.1, 0.8, 0.7])
    # second axis for Smatch
    ax_smatch = ax.twinx()
    colors = ['b', 'r', 'g', 'm', 'y']
    tags = sorted(plotting_data.keys())
    handles = []
    for i in range(len(tags)):

        color = colors[i % len(colors)]

        # train loss
        x, y, y_std = plotting_data[tags[i]]['train']
        h = ax.plot(x, y, color)[0]
        # ax.fill_between(x, y - y_std, y + y_std, alpha=0.3)
        # h = ax.fill_between(x, y - y_std, y + y_std, color=color2, alpha=0.3)
        handles.append(h)

        # valid loss
        # x, y, _ = plotting_data[tags[i]]['valid']
        # ax.plot(x, y, '--' + color)

        # dev decoding score
        x, y, y_std = plotting_data[tags[i]]['valid-dec']
        ax_smatch.plot(x, y, color)
        ax_smatch.fill_between(x, y - y_std, y + y_std, alpha=0.3)
        ax_smatch.set(ylim=(80, 85))

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax_smatch.set_ylabel('Smatch')

    plt.legend(handles, tags, bbox_to_anchor=(0, 1, 1, 0))
    if title:
        plt.title(title)
    if out_png:
        print(f'wrote {out_png}')
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
        etime = np.median([
            x['epoch_time'] for x in exp_data if x['epoch_time']]) / (60**2)
        print(f'Collecting data for {exp_tag} ({etime:.2f} h/epoch)')
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
