import os
import subprocess
import re
import argparse
from collections import defaultdict

import numpy as np

PRINT = True
checkpoint_re = re.compile('checkpoint([0-9]+).pt')
stdout_re = re.compile('dec-checkpoint([0-9]+).wiki.smatch')
dec_stdout_re = re.compile('dec.*.stdout')
results_re = re.compile('^F-score: ([0-9\.]+)')


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Organize model results'
    )
    # jbinfo args
    parser.add_argument(
        '--checkpoints',
        type=str,
        default='DATA/AMR/models/',
        help='Folder containing model folders (containing themselves checkpoints, config.sh etc)'
    )
    # jbinfo args
    parser.add_argument(
        '--seed-average',
        action='store_true',
        help='average results per seed'
    )
    parser.add_argument(
        '--no-print',
        action='store_true',
        help='do not print'
    )
    return parser.parse_args()


def get_score_from_log(file_path):
    results = None
    with open(file_path) as fid:
        for line in fid:
            if results_re.match(line):
                results = results_re.match(line).groups()
                results = list(map(float, results))
                break

    return results[0] if results else None


def collect_results(args):

    # Find decoding logs
    epoch_folders = [
        x[0] 
        for x in os.walk(args.checkpoints) 
        if 'epoch_tests' in x[0]
    ]

    items = []
    for folder in epoch_folders:

        item = {}

        # model folder
        model_folder = folder.replace('epoch_tests', '')
        model_files = os.listdir(model_folder)

        # best checkpoint dec log (and score)
        checkpoint_best_log = list(filter(dec_stdout_re.match, model_files))
        checkpoint_best_log = checkpoint_best_log[0] if checkpoint_best_log else None
        best_CE = None
        if checkpoint_best_log:
            best_CE = get_score_from_log(f'{model_folder}/{checkpoint_best_log}')

        # all checkpoints
        checkpoints = list(filter(checkpoint_re.match, model_files))

        # epoch folder
        epoch_files = os.listdir(folder)

        # epoch folder dec logs
        checkpoint_logs = list(filter(stdout_re.match, epoch_files))

        # scores
        scores = {}
        for stdout in checkpoint_logs:
            epoch = int(stdout_re.match(stdout).groups()[0])
            score = get_score_from_log(f'{folder}/{stdout}')
            if score is not None:
                scores[epoch] = score

        stdout_numbers = set([
            int(checkpoint_re.match(dfile).groups()[0])
            for dfile in checkpoints
        ])

        if not scores:
            print(f'Skipping {folder}')
            continue

        best_SMATCH = sorted(scores.items(), key=lambda x: x[1])[-1]
        missing_epochs = list(stdout_numbers - set(scores.keys()))

        # for epoch in missing_epochs:
        #     print(f'{model_folder}/checkpoint{epoch}.pt')

        item = {
            'folder': model_folder,
            'best_SMATCH': best_SMATCH[1],
            'best_SMATCH_epoch': int(best_SMATCH[0]),
            'num_missing_epochs': len(missing_epochs),
            'num': 1
        }
        if best_CE:
            item['best_CE_SMATCH'] = best_CE

        # link best SMATCH model
        target_best = f'{os.path.realpath(model_folder)}/checkpoint_best_SMATCH.pt'
        source_best = f'checkpoint{int(best_SMATCH[0])}.pt'
        # We may have created a link before to a worse model, remove it
        if os.path.isfile(target_best) and os.path.realpath(target_best) != source_best:
            os.remove(target_best)
        if not os.path.isfile(target_best):
            os.symlink(source_best, target_best)
        items.append(item)

    return items


def seed_average(items):

    # cluster by key
    clusters = defaultdict(list)
    for item in items:
        key = item['folder'].split('-seed')[0]
        clusters[key].append(item)

    # merge
    merged_items = []
    for key, cluster_items in clusters.items():

        def average(field):
            return np.mean([x[field] for x in cluster_items])

        def std(field):
            return 2*np.std([x[field] for x in cluster_items])

        def maximum(field):
            return np.max([x[field] for x in cluster_items])

        merged_items.append({
            'folder': key,
            'best_SMATCH': average('best_SMATCH'),
            'best_SMATCH_std': std('best_SMATCH'),
            'best_SMATCH_epoch': maximum('best_SMATCH_epoch'),
            'num_missing_epochs': maximum('num_missing_epochs'),
            'num': len(cluster_items)
        })

        if all(['best_CE_SMATCH' in item for item in items]):
            merged_items[-1]['best_CE_SMATCH'] = average('best_CE_SMATCH')

    return merged_items


if __name__ == '__main__':

    # ARGUMENT HANDLING
    args = argument_parsing()

    items = collect_results(args)
    
    if args.seed_average:
        items = seed_average(items)

    # get max length
    max_name_len = max(len(item['folder']) for item in items)
    
    if not args.no_print:
        print("")
        for item in sorted(items, key=lambda x: x['best_SMATCH']):
            display_str = ''
            display_str = '{:<{width}}  ({:d}) ({:d})'.format(
                item['folder'],
                item['num'],
                item['best_SMATCH_epoch'],
                width=max_name_len + 2
            )
            display_str += ' SMATCH {:2.1f}'.format(100*item['best_SMATCH'])
            if 'best_SMATCH_std' in item:
                display_str += ' ({:3.1f})'.format(100*item['best_SMATCH_std'])
            # if 'best_CE_SMATCH' in item:
            #    display_str += ' SMATCH {:2.1f}'.format(100*item['best_CE_SMATCH'])
            if 'num_missing_epochs' in item and item['num_missing_epochs'] > 0:
                display_str += ' {:d}!'.format(item['num_missing_epochs'])
            print(display_str)
        print("")
