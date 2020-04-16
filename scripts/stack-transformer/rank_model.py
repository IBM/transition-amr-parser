import os
import subprocess
import re
import argparse
from math import sqrt, ceil
from collections import defaultdict

PRINT = True
checkpoint_re = re.compile('checkpoint([0-9]+)\.pt')
las_re = re.compile('dec-checkpoint([0-9]+)\.las')
smatch_re = re.compile('dec-checkpoint([0-9]+)\.smatch')
smatch_re_wiki = re.compile('dec-checkpoint([0-9]+)\.wiki\.smatch')
smatch_results_re = re.compile('^F-score: ([0-9\.]+)')
las_results_re = re.compile('UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')
model_folder_re = re.compile('(.*)-seed([0-9]+)')


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
    parser.add_argument(
        '--min-epoch-delta',
        type=int,
        default=10,
        help='Minimum for the difference between best valid epoch and max epochs'
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


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def mean(items):
    return float(sum(items)) / len(items)


def std(items):
    mu = mean(items)
    if (len(items) - 1) == 0:
        return 0.0
    else:
        return sqrt(float(sum([(x - mu)**2 for x in items])) / (len(items) - 1))


def get_score_from_log(file_path, score_name):

    results = None

    if score_name == 'smatch':
        regex = smatch_results_re
    elif score_name == 'las':
        regex = las_results_re
    else:
        raise Exception(f'Unknown score type {score_name}')

    with open(file_path) as fid:
        for line in fid:
            if regex.match(line):
                results = regex.match(line).groups()
                results = list(map(float, results))
                break

    return results


def collect_results(args, results_regex, score_name):

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
            score = get_score_from_log(f'{epoch_folder}/{stdout}', score_name)
            if score is not None:
                scores[epoch] = score
        if not scores:
            continue

        # get top 3 scores and epochs    
        if score_name == 'las':
            sort_idx = 1
        else:    
            sort_idx = 0
        models = sorted(scores.items(), key=lambda x: x[1][sort_idx])[-3:]
        if len(models) == 3:
            third_best_score, second_best_score, best_score = models
        elif len(models) == 2:
            second_best_score, best_score = models
            third_best_score = [-1, -1]
        else:
            best_score = models[0]
            second_best_score = [-1, -1]
            third_best_score = [-1, -1]

        # Find if checkpoints have been deleted
        deleted_checkpoints = False
        for score in [best_score, second_best_score, third_best_score]:
            if score[0] == -1:
                continue
            if not os.path.isfile(f'{model_folder}/checkpoint{score[0]}.pt'):
                deleted_checkpoints = True
                break

        # find out epoch checkpoints that still need to be run
        missing_epochs = list(stdout_numbers - set(scores.keys()))

#         if 'large' in model_folder:
#             import ipdb; ipdb.set_trace(context=30)
#             print()

        # look for weight ensemble results
        if os.path.isfile(f'{model_folder}/top3-average/valid.{score_name}'):
            weight_ensemble_smatch = get_score_from_log(
                f'{model_folder}/top3-average/valid.{score_name}',
                score_name
            )
        elif os.path.isfile(f'{model_folder}/top3-average/valid.wiki.{score_name}'):
            weight_ensemble_smatch = get_score_from_log(
                f'{model_folder}/top3-average/valid.wiki.{score_name}',
                score_name
            )
        else:    
            weight_ensemble_smatch = None

        items.append({
            'folder': model_folder,
            # top scores
            f'best_{score_name}': best_score[1],
            f'second_best_{score_name}': second_best_score[1],
            f'third_best_{score_name}': third_best_score[1],
            # top score epochs 
            f'best_{score_name}_epoch': int(best_score[0]),
            f'second_best_{score_name}_epoch': int(second_best_score[0]),
            f'third_best_{score_name}_epoch': int(third_best_score[0]),
            # any top score checkpoints missing
            'deleted_checkpoints': deleted_checkpoints,
            # other
            'max_epochs': max(stdout_numbers),
            'num_missing_epochs': len(missing_epochs),
            'num': 1,
            'ensemble': False
        })

        if weight_ensemble_smatch is not None:
            items.append({
                'folder': f'{model_folder}',
                f'best_{score_name}': weight_ensemble_smatch,
                f'best_{score_name}_epoch': int(best_score[0]),
                'max_epochs': max(stdout_numbers),
                'num_missing_epochs': len(missing_epochs),
                'deleted_checkpoints': False,
                'num': 1,
                'ensemble': True
            })

    return items


def seed_average(items):
    """
    Aggregate stats for different seeds of same model
    """

    # cluster by key
    clusters = defaultdict(list)
    seeds = defaultdict(list)
    for item in items:
        key, seed = model_folder_re.match(item['folder']).groups()
        if item['ensemble']:
            key += '*'
        clusters[key].append(item)
        seeds[key].append(seed)

    # merge
    merged_items = []
    for key, cluster_items in clusters.items():

        def results_map(field, fun):
            if any([x[field] is None for x in cluster_items]):
                return None
            else:
                if isinstance(cluster_items[0][field], list):
                    num_types = len(cluster_items[0][field])
                    results = []
                    for t in range(num_types):
                        results.append(fun(
                            [x[field][t] for x in cluster_items]
                        ))
                    return results
                else:    
                    return fun([x[field] for x in cluster_items])
        def fany(field):
            return results_map(field, any)

        def average(field):
            return results_map(field, mean)

        def stdev(field):
            return results_map(field, lambda x: 2 * std(x))

        def maximum(field):
            return results_map(field, max)

        merged_items.append({
            'folder': key,
            f'best_{score_name}': average(f'best_{score_name}'),
            f'best_{score_name}_std': stdev(f'best_{score_name}'),
            f'best_{score_name}_epoch': ceil(average(f'best_{score_name}_epoch')),
            'max_epochs': ceil(average('max_epochs')),
            'num_missing_epochs': maximum('num_missing_epochs'),
            'num': len(cluster_items),
            'seeds': seeds[key],
            'deleted_checkpoints': fany('deleted_checkpoints')
        })

        if all(['best_CE_{score_name}' in item for item in items]):
            merged_items[-1][f'best_CE_{score_name}'] = \
                average(f'best_CE_{score_name}')

    return merged_items


def print_table(args, items, pattern, score_name, min_epoch_delta, 
                split_name=True):
   
    # add shortname as folder removing checkpoints root, get max length of
    # name for padding print
    max_size = [0, 0, 0]
    for item in items:
        shortname = item['folder'].replace(args.checkpoints, '')
        if split_name:
            shortname = shortname[1:] if shortname[0] == '/' else shortname
            shortname = shortname[:-1] if shortname[-1] == '/' else shortname
            pieces = shortname.split('_')
            pieces = ['_'.join(pieces[:-2])] + pieces[-2:] 
            max_size[0] = max(max_size[0], len(pieces[0]))
            max_size[1] = max(max_size[1], len(pieces[1]))
            max_size[2] = max(max_size[2], len(pieces[2]))
            item['shortname'] = pieces
        else:
            item['shortname'] = shortname

    # scale of the read results
    if score_name == 'las':
        sort_idx = 1
        scale = 1
    elif score_name == 'smatch':
        sort_idx = 0
        scale = 100
    
    print(f'\n{pattern}')
    for item in sorted(items, key=lambda x: x[f'best_{score_name}'][sort_idx]):

        # colored
        epoch_delta = item['max_epochs'] - item[f'best_{score_name}_epoch']
        convergence_epoch = '{:d}'.format(item[f'best_{score_name}_epoch'])

        # check if some checkpoint was deleted by
        folder = item['folder']
        if item['deleted_checkpoints']:
            convergence_epoch = red(f'{convergence_epoch}')
        elif epoch_delta < min_epoch_delta:
            convergence_epoch = yellow(f'{convergence_epoch}')

        shortname_str = ''
        for idx, piece in enumerate(item['shortname']):
            shortname_str += '{:<{width}} '.format(
                piece, 
                width=max_size[idx] + 1
            ) 
            
        # name, number of seeds, best epoch
        display_str = ''
        display_str = '{:s}  ({:d}) ({:s}/{:d})'.format(
            shortname_str,
            item['num'],
            convergence_epoch,
            item['max_epochs'],
        )

        if score_name == 'las':
            
            # first score
            display_str += ' {:s} {:2.1f}'.format(
                'UAS',
                scale * item[f'best_{score_name}'][0]
            )
            if f'best_{score_name}_std' in item:
                display_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][0]
                )
            # second score
            display_str += ' {:s} {:2.1f}'.format(
                'LAS',
                scale * item[f'best_{score_name}'][1]
            )
            if f'best_{score_name}_std' in item:
                display_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][1]
                )

        else:

            # first score
            display_str += ' {:s} {:2.1f}'.format(
                score_name.upper(),
                scale * item[f'best_{score_name}'][0]
            )
            if f'best_{score_name}_std' in item:
                display_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][0]
                )
 
        # missing epochs for test
        if 'num_missing_epochs' in item and item['num_missing_epochs'] > 0:
            display_str += yellow(' {:d}!'.format(item['num_missing_epochs']))
        print(display_str)
    print("")


def link_top_models(items, score_name):

    for item in items:
    
        if f'third_best_{score_name}_epoch' not in item:
            continue

        model_folder = os.path.realpath(item['folder'])
        for rank in ['best', 'second_best', 'third_best']: 
            epoch = item[f'{rank}_{score_name}_epoch']
            score_name_caps = score_name.upper()
            target_best = (f'{model_folder}/'
                           f'checkpoint_{rank}_{score_name_caps}.pt')
            source_best = f'checkpoint{epoch}.pt'
            # We may have created a link before to a worse model,
            # remove it
            if (
                os.path.islink(target_best) and
                os.path.basename(os.path.realpath(target_best)) != 
                    source_best
            ):
                os.remove(target_best)
            if not os.path.islink(target_best):
                os.symlink(source_best, target_best)


if __name__ == '__main__':

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Separate results with and without wiki
    for result_regex in [smatch_re, smatch_re_wiki, las_re]:

        # get name of score
        score_name = result_regex.pattern.split('.')[-1]

        # collect results for each model
        items = collect_results(args, result_regex, score_name)

        if items == []:
            continue

        # link best score model
        if args.link_best:
            link_top_models(items, score_name)

        if items != [] and not args.no_print:
            # average over seeds
            if args.seed_average:
                folders = [x['folder'] for x in items]
                items = seed_average(items)
            print_table(args, items, result_regex.pattern, score_name, args.min_epoch_delta)
