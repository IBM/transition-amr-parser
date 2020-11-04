import os
import glob
import re
import argparse
from math import sqrt, ceil
from collections import defaultdict

# checkpoints/folder regex
model_folder_re = re.compile('(.*)-seed([0-9]+)')

# results file name regex
# smatch_re = re.compile(r'dec-checkpoint([0-9]+)\.smatch')
# smatch_re_wiki = re.compile(r'dec-checkpoint([0-9]+)\.wiki\.smatch')

# checkpoints/folder regex
checkpoint_re = re.compile(r'checkpoint([0-9]+)\.pt')
results_regex = re.compile(r'dec-checkpoint([0-9]+)\.(.+)')

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


config_var_regex = re.compile(r'^([^=]+)=([^ ]+).*$')


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
        '--score-names',
        nargs='+',
        default=['smatch', 'wiki.smatch'],
        help='Set of the results'
    )
    parser.add_argument(
        '--no-split-name',
        action='store_true',
        help='do not split model name into components'
    )
    parser.add_argument("--ignore-deleted", action='store_true')
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


def get_scores_from_folder(epoch_folder, score_name):
    '''
    Get score files from an epoch folder (<model_folder>/epoch_folder/)
    '''

    # Get results available in this folder 
    scores = {}
    epoch_numbers = []
    for dfile in os.listdir(epoch_folder):

        # if not a results file, skip
        if not results_regex.match(dfile): 
            continue

        epoch_number, sname = results_regex.match(dfile).groups()

        # store epoch number
        epoch_numbers.append(int(epoch_number))

        if sname != score_name:
            continue

        # get score
        score = get_score_from_log(f'{epoch_folder}/{dfile}', score_name)
        if score is not None:
            scores[int(epoch_number)] = score

    return scores, max(epoch_numbers) if epoch_numbers else -1


def get_score_from_log(file_path, score_name):

    results = None

    if 'smatch' in score_name:
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


def rank_scores(scores, score_name):

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

    return best_score, second_best_score, third_best_score, top3_prev


def get_max_epoch_from_config(model_folder):
    max_epoch = None
    with open(f'{model_folder}/config.sh') as fid:
        for line in fid.readlines():
            if config_var_regex.match(line.strip()):
                name, value = config_var_regex.match(line.strip()).groups()
                if name == 'MAX_EPOCH':
                    max_epoch = int(value)
                    break
    return max_epoch


def collect_checkpoint_results(epoch_folders, score_name):

    # loop over those folders
    items = []
    for epoch_folder in epoch_folders:

        # data in {epoch_folder}/../
        # assume containing folder is the model folder
        model_folder = epoch_folder.replace('epoch_tests', '')
        if not os.path.isdir(model_folder):
            continue

        # Get checkpoints available for this model
        checkpoints = list(filter(checkpoint_re.match, os.listdir(model_folder)))

        # Get the scores from the result files
        scores, max_score_epoch = \
            get_scores_from_folder(epoch_folder, score_name)

        # If no scores skip this folder
        if not scores:
            continue

        # get top 3 scores and epochs
        best_score, second_best_score, third_best_score, top3_prev  = \
            rank_scores(scores, score_name)

        # Find if checkpoints have been deleted and there is no copy (e.g. as
        # done by remove_checkpoints.sh)
        deleted_checkpoints = False
        for label, score in zip(
            ['best', 'second_best', 'third_best'], 
            [best_score, second_best_score, third_best_score]
        ):
            if score[0] == -1:
                continue

            saved_checkpoint_name = (
                f'{model_folder}/checkpoint_{label}_{score_name.upper()}.pt'
            )

            if (
                f'checkpoint{score[0]}.pt' not in checkpoints and
                not os.path.isfile(saved_checkpoint_name) 
            ):
                deleted_checkpoints = True
                break

        # find out epoch checkpoints that still need to be run
        stdout_numbers = [
            int(checkpoint_re.match(x).groups()[0]) for x in checkpoints
        ] 

        # find out the maximum number of epochs
        max_epochs = get_max_epoch_from_config(model_folder)
        missing_epochs = list(set(stdout_numbers) - set(scores.keys()))

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
            'max_epochs': max_epochs,
            'num_missing_epochs': len(missing_epochs),
            'num': 1,
            'ensemble': False
        })

    return items


def get_extra_results(epoch_folders, sset, score_name):

    # loop over those folders
    items = []
    for epoch_folder in epoch_folders:

        # data in {epoch_folder}/../
        # assume containing folder is the model folder
        model_folder = epoch_folder.replace('epoch_tests', '')

        # Extra results
        for extra_exp in glob.glob(
            f'{model_folder}/*/{sset}.{score_name}'
        ):

            # look for extra experiments
            exp_tag = os.path.basename(os.path.dirname(extra_exp))

            if exp_tag == 'epoch_tests':
                continue

            exp_smatch = get_score_from_log(extra_exp, score_name)

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


def get_basic_table_info(items, checkpoints, score_name, split_name):

    # check if split will work, other wise override split_name
    if split_name and any([
        len(get_shortname(item, checkpoints).split()[0].split('_')) != 4 
        for item in items
    ]):
        warn = yellow('WARNING:')
        split_name = False
        print(f'\n{warn} Model name not well formmatted (spureous _ ).'
               ' Fix name or use --no-split-name\n')
    
    # add shortname as folder removing checkpoints root, get max length of
    # name for padding print
    # scale of the read results
    if score_name == 'las':
        sort_idx = 1
        scale = 1
    elif 'smatch' in score_name:
        sort_idx = 0
        scale = 100

    # Header
    if split_name:
        centering = ['<', '<', '<', '<', '<', '^', '^']
        row = [
            'data', 'oracle', 'features', 'model', 'extra', 'seeds', 'best epoch'
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

    return rows, centering, scale, sort_idx, split_name


def get_shortname(item, checkpoints):
    # name
    shortname = item['folder'].replace(checkpoints, '')
    # if we give model folder direcly, shortname will be empty, use the
    # containing folder
    if shortname == '':
        if checkpoints[-1] == '/':
            shortname = os.path.basename(checkpoints[:-1])
        else:    
            shortname = os.path.basename(checkpoints)
    else:    
        shortname = shortname[1:] if shortname[0] == '/' else shortname
    return shortname


def get_name_rows(split_name, item, checkpoints):

    row = []
    shortname = get_shortname(item, checkpoints)

    if split_name and len(shortname.split()[0].split('_')) != 4:
        split_name = False
    
    if split_name:
    
        # Remove slash at start of end
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

    return row


def get_score_rows(score_name, item, scale):

    row = []

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

    return row


def print_table(checkpoints, items, score_name, min_epoch_delta,
                split_name=True):

    rows, centering, scale, sort_idx, split_name = \
        get_basic_table_info(items, checkpoints, score_name, split_name)

    # Loop over table rows
    items = sorted(items, key=lambda x: x[f'best_{score_name}'][sort_idx])
    for item in items:

        # rows pertaining node name
        row = get_name_rows(split_name, item, checkpoints)

        # number of seeds row
        row.append('{}'.format(item['num']))

        # best epoch row
        epoch_delta = item['max_epochs'] - item[f'best_{score_name}_epoch']
        convergence_epoch = '{:d}'.format(item[f'best_{score_name}_epoch'])
        # check if some checkpoint was deleted by
        if item['deleted_checkpoints']:
            convergence_epoch = red(f'{convergence_epoch}')
        elif epoch_delta < min_epoch_delta:
            convergence_epoch = yellow(f'{convergence_epoch}')
        row.append('{:s}/{:d}'.format(convergence_epoch, item['max_epochs']))

        # score row
        row.extend(get_score_rows(score_name, item, scale))

        # missing epochs for test
        if 'num_missing_epochs' in item and item['num_missing_epochs'] > 0:
            row.append(yellow(' {:d}!'.format(item['num_missing_epochs'])))
        else:
            row.append('')

        # collect
        rows.append(row)

    print(f'\n{score_name}')
    ptable(rows, centering)


def ptable(rows, centering):

    num_columns = len(rows[0])
    # bash scape chars (used for formatting, have length 0 on display)
    BASH_SCAPE = re.compile(r'\x1b\[\d+m|\x1b\[0m')
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


def link_top_models(items, score_name, ignore_deleted):

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

            # get names and paths
            score_name_caps = score_name.upper()
            target_best = (f'{model_folder}/'
                           f'checkpoint_{rank}_{score_name_caps}.pt')
            source_best = f'checkpoint{epoch}.pt'

            # if the best checkpoint does not exist but we have not saved it as
            # a file, we are in trouble
            if (
                not os.path.isfile(f'{model_folder}/{source_best}')
                and not os.path.isfile(target_best)
            ):
                if ignore_deleted:
                    continue
                else:
                    raise Exception(
                        f'Best model is {model_folder}/{source_best}, however'
                        ', the checkpoint seems to have been removed' 
                    )

            # get current best model (if exists)
            if os.path.islink(target_best):
                current_best = os.path.basename(os.path.realpath(target_best))
            else:
                current_best =  None

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


def main():

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Find folders of the form /path/to/epoch_folders/ for all checkpoints
    epoch_folders = [
        x[0]
        for x in os.walk(args.checkpoints)
        if 'epoch_tests' in x[0]
    ]

    # Separate results with and without wiki
    for score_name in args.score_names:

        # collect results for each model. For validation we have las N
        # checkpoints, which we use to determine the best model
        items = []
        if args.set == 'valid':
            items = collect_checkpoint_results(epoch_folders, score_name)
        # collect extra results such as beam or weight average experiments
        items.extend(get_extra_results(epoch_folders, args.set, score_name))

        if items == []:
            continue

        # link best score model
        if args.link_best:
            link_top_models(items, score_name, args.ignore_deleted)

        if items != [] and not args.no_print:
            # average over seeds
            if args.seed_average:
                items = seed_average(items, score_name)
            print_table(args.checkpoints, items,
                        score_name, args.min_epoch_delta, split_name=not
                        args.no_split_name)


if __name__ == '__main__':
    main()
