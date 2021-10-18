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


def get_vectors(items, label):

    def x_key(item):
        return int(item['epoch'])

    def y_reduce(items):
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


def read_experiment(seed_folder):

    config_env_vars = read_config_variables(f'{seed_folder}/config.sh')

    experiment_key = config_env_vars['MODEL_FOLDER']

    # read info from logs
    exp_data =  []
    for log_file in glob(f'{seed_folder}/tr-*.stdout'):
        with open(log_file) as fid:
            for line in fid:
                if train_info_regex.match(line):
                    date_str, json_info = train_info_regex.match(line).groups()
                    item = json.loads(json_info)
                    item['timestamp'] = parse(date_str)
                    item['experiment_key'] = experiment_key
                    item['set'] = 'train'
                    exp_data.append(item)

                elif valid_info_regex.match(line):
                    date_str, json_info = valid_info_regex.match(line).groups()
                    item = json.loads(json_info)
                    item['timestamp'] = parse(date_str)
                    item['experiment_key'] = experiment_key
                    item['set'] = 'valid'
                    exp_data.append(item)

    # read validation decoding scores
    eval_metric = config_env_vars['EVAL_METRIC']
    validation_folder = f'{seed_folder}/epoch_tests/'
    for epoch in range(int(config_env_vars['MAX_EPOCH'])):
        results_file = f'{validation_folder}/dec-checkpoint{epoch}.{eval_metric}'
        if os.path.isfile(results_file):
            score = get_score_from_log(results_file, eval_metric)[0]
            exp_data.append({
                'epoch': epoch,
                'set': 'valid-dec',
                'score': score,
                'experiment_key': experiment_key
            })

    return exp_data


if __name__ == '__main__':

    log_files = [
        'DATA/AMR2.0/models/exp_align_cfg_o10_act-states-importance-5sample_a_bart.large/_act-pos_vmask0_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt409x20-wm4000-dp0.2/ep100-seed42/tr-amr2.0-structured-bart-large-neur-al-importance-sampling5-s42-1817085-1940038.stdout',
        #'DATA/AMR2.0/models/exp_cofill_o10_act-states_cofill_o10_act-states_bart.large/_act-pos_vmask0_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt2048x4-wm4000-dp0.2-no-voc-mask/ep100-seed42/tr-amr2.0-structured-bart-large-no-voc-mask-s42-1329124-475365.stdout',
        #'DATA/AMR2.0/models/exp_cofill_o10_act-states_cofill_o10_act-states_bart.large/_act-pos_vmask0_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt2048x4-wm4000-dp0.2-no-voc-mask/ep100-seed43/tr-amr2.0-structured-bart-large-no-voc-mask-s43-1329124-475366.stdout',
        #'DATA/AMR2.0/models/exp_cofill_o10_act-states_cofill_o10_act-states_bart.large/_act-pos_vmask0_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt2048x4-wm4000-dp0.2-no-voc-mask/ep100-seed44/tr-amr2.0-structured-bart-large-no-voc-mask-s44-1329124-475367.stdout'
    ]

    train_info_regex = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) \| INFO \| train \| (.*)')
    valid_info_regex = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) \| INFO \| valid \| (.*)')

    data = []
    for seed_folder in log_files:
        data.extend(read_experiment(seed_folder))

    # Cluster by experiment
    experiments = defaultdict(list)
    for item in data:
        experiments[item['experiment_key']].append(item)

    # For each experiment collect separate data for train, valid and score
    # aggregate stats for multiple seeds and produce vectors for later
    # plotting
    plotting_data = defaultdict(dict)
    for exp_tag, exp_data in experiments.items():
        for sset in ['train', 'valid']:
            valid_data = [x for x in exp_data if x['set'] == sset]
            plotting_data[exp_tag][sset] = \
                get_vectors(valid_data, f'{sset}_loss')

        sset = 'valid-dec'
        score_data = [x for x in exp_data if x['set'] == sset]
        plotting_data[exp_tag][sset] = get_vectors(score_data, 'score')

    fig = plt.figure(0)
    ax = plt.gca()
    ax_smatch = ax.twinx()
    colors = ['b', 'r', 'g']
    tags = sorted(plotting_data.keys())
    for i in range(len(tags)):

        color = colors[i % len(colors)]

        # train loss
        x, y = plotting_data[tags[i]]['train']
        h = ax.plot(x, y, color)[0]

        # dev loss
        x, y = plotting_data[tags[i]]['valid']
        ax.plot(x, y, '--' + color)[0]

        # dev decoding score
        x, y = plotting_data[tags[i]]['valid-dec']
        ax_smatch.plot(x, y, color)[0]
        ax_smatch.set(ylim=(80, 85))

        # plt.xlabel(args.x_label)
        # plt.ylabel(args.y_label)
        # plt.legend(handles, labels)

    plt.savefig('cosa.png')
    # plt.show()
