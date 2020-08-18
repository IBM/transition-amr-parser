import os
import glob
import re
import argparse
from math import sqrt, ceil
from collections import defaultdict

# checkpoints/folder regex
checkpoint_re = re.compile(r'checkpoint([0-9]+)\.pt')
model_folder_re = re.compile('(.*)-seed([0-9]+)')

# results file name regex
las_re = re.compile(r'dec-checkpoint([0-9]+)\.las')
smatch_re = re.compile(r'dec-checkpoint([0-9]+)\.smatch')
smatch_re_wiki = re.compile(r'dec-checkpoint([0-9]+)\.wiki\.smatch')

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


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
        help='Folder containing model folders (containing themselves '
             'checkpoints, config.sh etc)'
    )
    parser.add_argument(
        '--min-epoch-delta',
        type=int,
        default=10,
        help='Minimum for the difference between best valid epoch and max'
             ' epochs'
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
    parser.add_argument(
        '--set',
        default='valid',
        choices=['valid', 'test'],
        help='Set of the results'
    )
    parser.add_argument(
        '--no-split-name',
        action='store_true',
        help='do not split model name into components'
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
        var = float(sum([(x - mu)**2 for x in items])) / (len(items) - 1)
        return sqrt(var)


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
        models = sorted(scores.items(), key=lambda x: x[1][sort_idx])
        if len(models) >= 3:
            third_best_score, second_best_score, best_score = models[-3:]
        elif len(models) == 2:
            second_best_score, best_score = models
            third_best_score = [-1, -1]
        else:
            best_score = models[0]
            second_best_score = [-1, -1]
            third_best_score = [-1, -1]

        # get top 3, but second and third happening before last
        top3_prev = [(None, None), (None, None)]
        idx = 0
        for m in models[::-2]:
            number, score = m
            if number < models[-1][0]:
                top3_prev[idx] = m
                idx += 1
            if idx == 2:
                break

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

        items.append({
            'folder': model_folder,
            # top scores
            f'best_{score_name}': best_score[1],
            f'second_best_{score_name}': second_best_score[1],
            f'third_best_{score_name}': third_best_score[1],
            f'second_best_before_{score_name}': top3_prev[1][1],
            f'third_best_before_{score_name}': top3_prev[0][1],
            # top score epochs
            f'best_{score_name}_epoch': int(best_score[0]),
            f'second_best_{score_name}_epoch': int(second_best_score[0]),
            f'third_best_{score_name}_epoch': int(third_best_score[0]),
            f'second_best_before_{score_name}_epoch': top3_prev[1][0],
            f'third_best_before_{score_name}_epoch': top3_prev[0][0],
            # any top score checkpoints missing
            'deleted_checkpoints': deleted_checkpoints,
            # other
            'max_epochs': max(stdout_numbers),
            'num_missing_epochs': len(missing_epochs),
            'num': 1,
            'ensemble': False
        })

    return items


def get_extra_results(args, score_name):

    # Find folders of the form /path/to/epoch_folders
    epoch_folders = [
        x[0]
        for x in os.walk(args.checkpoints)
        if 'epoch_tests' in x[0]
    ]

    # loop ove those folders
    items = []
    for epoch_folder in epoch_folders:

        # data in {epoch_folder}/../
        # assume containing folder is the model folder
        model_folder = epoch_folder.replace('epoch_tests', '')

        # Extra results
        for extra_exp in glob.glob(
            f'{model_folder}/*/{args.set}*.{score_name}'
        ):

            # look for extra experiments
            exp_tag = os.path.basename(os.path.dirname(extra_exp))

            if exp_tag == 'epoch_tests':
                continue

            if os.path.isfile(
                f'{model_folder}/{exp_tag}/{args.set}.{score_name}'
            ):
                exp_smatch = get_score_from_log(
                    f'{model_folder}/{exp_tag}/{args.set}.{score_name}',
                    score_name
                )
            elif os.path.isfile(
                f'{model_folder}/{exp_tag}/{args.set}.wiki.{score_name}'
            ):
                exp_smatch = get_score_from_log(
                    f'{model_folder}/{exp_tag}/{args.set}.wiki.{score_name}',
                    score_name
                )
            else:
                exp_smatch = None

            if exp_smatch is not None:
                items.append({
                    'folder': f'{model_folder}',
                    f'best_{score_name}': exp_smatch,
                    f'best_{score_name}_epoch': 0,
                    'max_epochs': 0,
                    'num_missing_epochs': 0,
                    'deleted_checkpoints': False,
                    'num': 1,
                    'ensemble': True,
                    'extra_exp': exp_tag
                })

    return items


def seed_average(items, score_name):
    """
    Aggregate stats for different seeds of same model
    """

    # cluster by key
    clusters = defaultdict(list)
    seeds = defaultdict(list)
    for item in items:
        key, seed = model_folder_re.match(item['folder']).groups()
        if 'extra_exp' in item:
            key += ' '
            key += item['extra_exp']
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
            return results_map(field, lambda x: std(x))

        def maximum(field):
            return results_map(field, max)

        merged_items.append({
            'folder': key,
            f'best_{score_name}': average(f'best_{score_name}'),
            f'best_{score_name}_std': stdev(f'best_{score_name}'),
            f'best_{score_name}_epoch':
                ceil(average(f'best_{score_name}_epoch')),
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
    # scale of the read results
    if score_name == 'las':
        sort_idx = 1
        scale = 1
    elif score_name == 'smatch':
        sort_idx = 0
        scale = 100

    print(f'\n{pattern}')
    # Header
    if split_name:
        centering = ['<', '<', '<', '<', '^', '^']
        row = [
            'data/oracle', 'features', 'model', 'extra', 'seeds', 'best epoch'
        ]
    else:
        centering = ['<', '^', '^']
        row = ['name', 'seed', 'best epoch']
    if score_name == 'las':
        centering.extend(['^', '^'])
        row.extend(['UAS', 'LAS'])
    else:
        centering.append('^')
        row.append('SMATCH')
    # extra warning
    centering.append('<')
    row.append('')
    # style for rows
    rows = [row]

    # Loop over table rows
    for item in sorted(items, key=lambda x: x[f'best_{score_name}'][sort_idx]):

        row = []

        # name
        shortname = item['folder'].replace(args.checkpoints, '')
        shortname = shortname[1:] if shortname[0] == '/' else shortname
        shortname = shortname[:-1] if shortname[-1] == '/' else shortname
        if split_name:
            # Remove slash at start of end
            shortname = shortname[1:] if shortname[0] == '/' else shortname
            shortname = shortname[:-1] if shortname[-1] == '/' else shortname
            # ignore _ on first field
            main_pieces = shortname.split()
            if len(main_pieces) > 1:
                pieces = main_pieces[0].split('_') + [main_pieces[-1]]
            else:
                pieces = shortname.split('_') + ['']

            if 'extra_exp' in item:
                pieces[-1] += ' '.join(item['extra_exp'].split('_'))

            row.extend(pieces)
        else:
            if 'extra_exp' in item:
                shortname += ' '
                shortname += item['extra_exp']
            row.append(shortname)

        # number of seeds
        row.append('{}'.format(item['num']))

        # best epoch
        epoch_delta = item['max_epochs'] - item[f'best_{score_name}_epoch']
        convergence_epoch = '{:d}'.format(item[f'best_{score_name}_epoch'])
        # check if some checkpoint was deleted by
        if item['deleted_checkpoints']:
            convergence_epoch = red(f'{convergence_epoch}')
        elif epoch_delta < min_epoch_delta:
            convergence_epoch = yellow(f'{convergence_epoch}')
        row.append('{:s}/{:d}'.format(convergence_epoch, item['max_epochs']))

        if score_name == 'las':

            # first score
            cell_str = '{:2.1f}'.format(
                scale * item[f'best_{score_name}'][0]
            )
            if f'best_{score_name}_std' in item:
                cell_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][0]
                )
            row.append(cell_str)

            # second score
            cell_str = '{:2.1f}'.format(
                scale * item[f'best_{score_name}'][1]
            )
            if f'best_{score_name}_std' in item:
                cell_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][1]
                )
            row.append(cell_str)

        else:

            # first score
            cell_str = '{:2.1f}'.format(
                scale * item[f'best_{score_name}'][0]
            )
            if f'best_{score_name}_std' in item:
                cell_str += ' ({:3.1f})'.format(
                    scale * item[f'best_{score_name}_std'][0]
                )
            row.append(cell_str)

        # missing epochs for test
        if 'num_missing_epochs' in item and item['num_missing_epochs'] > 0:
            row.append(yellow(' {:d}!'.format(item['num_missing_epochs'])))
        else:
            row.append('')

        # collect
        assert len(centering) == len(row)
        rows.append(row)

    ptable(rows, centering)


def ptable(rows, centering):

    num_columns = len(rows[0])
    # bash scape chars (used for formatting, have length 0 on display)
    BASH_SCAPE = re.compile(r'\\x1b\[\d+m|\\x1b\[0m')
    column_widths = [
        max([len(BASH_SCAPE.sub('', row[i])) for row in rows])
        for i in range(num_columns)
    ]

    table_str = []
    col_sep = ' '
    for i,  row in enumerate(rows):
        row_str = []
        for j, cell in enumerate(row):
            # need to discount for bash scape chars
            delta = len(cell) - len(BASH_SCAPE.sub('', cell))
            if i == 0:
                # Header has all cells centered
                align = '^'
            else:
                align = centering[j]
            row_str.append(
                '{:{align}{width}} '.format(
                    cell, align=align, width=column_widths[j] + delta)
            )
        table_str.append(col_sep.join(row_str))

    row_sep = '\n'
    print(row_sep.join(table_str))
    print("")


def link_top_models(items, score_name):

    for item in items:

        if f'third_best_{score_name}_epoch' not in item:
            continue

        model_folder = os.path.realpath(item['folder'])
        # TODO: Decide if we want this disabled or not
        # for rank in ['best', 'second_best', 'third_best',
        # 'second_best_before', 'third_best_before']:
        for rank in ['best', 'second_best', 'third_best']:
            epoch = item[f'{rank}_{score_name}_epoch']

            # skip if no model found
            if epoch == -1:
                continue

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


def main():

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Separate results with and without wiki
    for result_regex in [smatch_re, smatch_re_wiki, las_re]:

        # get name of score
        score_name = result_regex.pattern.split('.')[-1]

        # collect results for each model
        items = []
        if args.set == 'valid':
            items = collect_results(args, result_regex, score_name)
        items.extend(get_extra_results(args), score_name)

        if items == []:
            continue

        # link best score model
        if args.link_best:
            link_top_models(items, score_name)

        if items != [] and not args.no_print:
            # average over seeds
            if args.seed_average:
                items = seed_average(items, score_name)
            print_table(args, items, result_regex.pattern, score_name,
                        args.min_epoch_delta,
                        split_name=not args.no_split_name)


if __name__ == '__main__':
    main()
