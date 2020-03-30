import os
import subprocess
import re
import argparse
from math import sqrt
from collections import defaultdict

PRINT = True
checkpoint_re = re.compile('checkpoint([0-9]+)\.pt')
stdout_re = re.compile('dec-checkpoint([0-9]+)\.smatch')
stdout_re_wiki = re.compile('dec-checkpoint([0-9]+)\.wiki\.smatch')
dec_stdout_re = re.compile('dec.*\.stdout')
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
        '--link-best',
        action='store_true',
        help='do not link or relink best smatch model'
    )
    parser.add_argument(
        '--no-print',
        action='store_true',
        help='do not print'
    )
    return parser.parse_args()


def mean(items):
    return float(sum(items)) / len(items)


def std(items):
    mu = mean(items)
    if (len(items) - 1) == 0:
        return 0.0
    else:
        return sqrt(float(sum([(x - mu)**2 for x in items])) / (len(items) - 1))


def get_score_from_log(file_path):
    results = None
    with open(file_path) as fid:
        for line in fid:
            if results_re.match(line):
                results = results_re.match(line).groups()
                results = list(map(float, results))
                break

    return results[0] if results else None


def collect_results(args, results_regex):

    # Find folders of the form /path/to/epoch_folders
    epoch_folders = [
        x[0] 
        for x in os.walk(args.checkpoints) 
        if 'epoch_tests' in x[0]
    ]

    # loop ove those folders
    items = []
    for epoch_folder in epoch_folders:

        item = {}

        # data in {epoch_folder}/../
        # assume containing folder is the model folder
        model_folder = epoch_folder.replace('epoch_tests', '')
        model_files = os.listdir(model_folder)
        # list all checkpoints
        checkpoints = list(filter(checkpoint_re.match, model_files))
        stdout_numbers = set([
            int(checkpoint_re.match(dfile).groups()[0])
            for dfile in checkpoints
        ])

        # data in epoch_folder
        epoch_files = os.listdir(epoch_folder)
        checkpoint_logs = list(filter(results_regex.match, epoch_files))
        # scores
        scores = {}
        for stdout in checkpoint_logs:
            epoch = int(results_regex.match(stdout).groups()[0])
            score = get_score_from_log(f'{epoch_folder}/{stdout}')
            if score is not None:
                scores[epoch] = score
        if not scores:
            continue
        # get top 3 scores and epochs    
        third_best_SMATCH, second_best_SMATCH, best_SMATCH = \
            sorted(scores.items(), key=lambda x: x[1])[-3:]
        missing_epochs = list(stdout_numbers - set(scores.keys()))

        # look for weight ensemble results
        if os.path.isfile(f'{model_folder}/top3-average/valid.smatch'):
            weight_ensemble_smatch = get_score_from_log(
                f'{model_folder}/top3-average/valid.smatch'
            )
        elif os.path.isfile(f'{model_folder}/top3-average/valid.wiki.smatch'):
            weight_ensemble_smatch = get_score_from_log(
                f'{model_folder}/top3-average/valid.wiki.smatch'
            )
        else:    
            weight_ensemble_smatch = None

        items.append({
            'folder': model_folder,
            'best_SMATCH': best_SMATCH[1],
            'best_SMATCH_epoch': int(best_SMATCH[0]),
            'second_best_SMATCH': second_best_SMATCH[1],
            'second_best_SMATCH_epoch': int(second_best_SMATCH[0]),
            'third_best_SMATCH': third_best_SMATCH[1],
            'third_best_SMATCH_epoch': int(third_best_SMATCH[0]),
            'num_missing_epochs': len(missing_epochs),
            'num': 1
        })

        if weight_ensemble_smatch is not None:
            items.append({
                'folder': f'{model_folder} (pt ensemble)',
                'best_SMATCH': weight_ensemble_smatch,
                'best_SMATCH_epoch': int(best_SMATCH[0]),
                'num_missing_epochs': len(missing_epochs),
                'num': 1,
            })

    return items


def seed_average(items):

    # cluster by key
    clusters = defaultdict(list)
    for item in items:
        key = re.sub('-seed[0-9]+', '', item['folder'])
        clusters[key].append(item)

    # merge
    merged_items = []
    for key, cluster_items in clusters.items():

        def average(field):
            if any([x[field] is None for x in cluster_items]):
                return None
            else:
                return mean([x[field] for x in cluster_items])

        def stdev(field):
            return 2*std([x[field] for x in cluster_items])

        def maximum(field):
            return max([x[field] for x in cluster_items])

        merged_items.append({
            'folder': key,
            'best_SMATCH': average('best_SMATCH'),
            'best_SMATCH_std': stdev('best_SMATCH'),
            'best_SMATCH_epoch': maximum('best_SMATCH_epoch'),
            'num_missing_epochs': maximum('num_missing_epochs'),
            'num': len(cluster_items)
        })

        if all(['best_CE_SMATCH' in item for item in items]):
            merged_items[-1]['best_CE_SMATCH'] = average('best_CE_SMATCH')

    return merged_items


def print_table(args, items, pattern):

   
    # add shortname as folder removing checkpoints root, get max length of
    # name for padding print
    for item in items:
        item['shortname'] = item['folder'].replace(args.checkpoints, '')
    max_name_len = max(len(item['shortname']) for item in items)
    
    print(f'\n{pattern}')
    for item in sorted(items, key=lambda x: x['best_SMATCH']):
        display_str = ''
        display_str = '{:<{width}}  ({:d}) ({:d})'.format(
            item['shortname'],
            item['num'],
            item['best_SMATCH_epoch'],
            width=max_name_len + 2
        )
        display_str += ' SMATCH {:2.1f}'.format(100*item['best_SMATCH'])
        if 'best_SMATCH_std' in item:
            display_str += ' ({:3.1f})'.format(100*item['best_SMATCH_std'])
        if 'num_missing_epochs' in item and item['num_missing_epochs'] > 0:
            display_str += ' {:d}!'.format(item['num_missing_epochs'])
        print(display_str)
    print("")


if __name__ == '__main__':

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Separate results with and without wiki
    for result_regex in [stdout_re, stdout_re_wiki]:

        # collect results for each model
                items = collect_results(args, result_regex)

        # link best SMATCH model
        if args.link_best:
            for item in items:

                if 'third_best_SMATCH_epoch' not in intem:
                    continue

                model_folder = os.path.realpath(item['folder'])
                for rank in ['best', 'second_best', 'third_best']: 
                    epoch = item[f'{rank}_SMATCH_epoch']
                    target_best = f'{model_folder}/checkpoint_{rank}_SMATCH.pt'
                    source_best = f'checkpoint{epoch}.pt'
                    # We may have created a link before to a worse model,
                    # remove it
                    if (
                        os.path.isfile(target_best) and
                        os.path.realpath(target_best) != source_best
                    ):
                        os.remove(target_best)
                    if not os.path.isfile(target_best):
                        os.symlink(source_best, target_best)

        if items != [] and not args.no_print:
            # average over seeds
            if args.seed_average:
                items = seed_average(items)
            print_table(args, items, result_regex.pattern)
