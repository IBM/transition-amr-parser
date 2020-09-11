import os
import re
import argparse


# beam results dir name reges
beam_dir_re = re.compile(r'beam([0-9]+)')

# results file name regex
results_re = re.compile(r'(.+)_checkpoint_(.+?)\.(.+)')  # +? for non-greedy matching; only match the first "\."
# smatch_re = re.compile(r'valid_checkpoint([0-9]+)\.smatch')
# smatch_re_wiki = re.compile(r'valid_heckpoint([0-9]+)\.wiki\.smatch')

# model names to consider
models = ['last', 'wiki-smatch_best1', 'wiki-smatch_top3-avg', 'wiki-smatch_top5-avg']

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def parse_args():
    parser = argparse.ArgumentParser(description='Collect model results')
    parser.add_argument('checkpoints', type=str, default='/dccstor/jzhou1/work/EXP/exp_debug/models_ep120_seed42',
                        help='folder containing saved model checkpoints for a single training')
    parser.add_argument('--data_sets', type=str, nargs='*', default=['valid', 'test'],
                        help='data sets to collect scores')
    parser.add_argument('--models', type=str, nargs='*', default=models,
                        help='model checkpoint names to collect scores')
    parser.add_argument('--score_name', type=str, default='wiki.smatch',
                        help='postfix of the score files')
    parser.add_argument('--ndigits', type=int, default=2,
                        help='number of digits after the decimal point')
    # parser.add_argument('--save_name', type=str, default='collected_wiki-smatch_scores.txt',
    #                     help='save name for the collection of results')
    args = parser.parse_args()
    return args


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

    if results is not None:
        assert 'smatch' in score_name, 'currently only working for smatch where there is only one score'
        results = results[0]

    return results


def get_scores_from_beam_dir(beam_dir, data_sets, models, score_name):
    # get results from a beam directory with a single beam size
    results_dict = {}

    for dfile in os.listdir(beam_dir):

        if not results_re.match(dfile):
            continue

        data_set, model_name, sname = results_re.match(dfile).groups()

        if (data_set in data_sets) and (model_name in models) and (sname == score_name):
            score = get_score_from_log(os.path.join(beam_dir, dfile), score_name)
            results_dict.setdefault(model_name, {})[data_set] = score    # could be None

    return results_dict


def collect_final_scores(checkpoint_folder, data_sets, models, score_name):
    beam_sizes = []
    for name in os.listdir(checkpoint_folder):

        if not beam_dir_re.match(name):
            continue

        beam_size, = beam_dir_re.match(name).groups()
        beam_sizes.append(int(beam_size))

    beam_sizes = sorted(beam_sizes)
    beam_results = {}

    for bs in beam_sizes:

        beam_dir = os.path.join(checkpoint_folder, f'beam{bs}')

        results_dict = get_scores_from_beam_dir(beam_dir, data_sets, models, score_name)
        beam_results[bs] = results_dict

    # beam_results is a dictionary, with key levels beam_size -> model_name -> data_set -> score
    return beam_results


class SaveAndPrint:
    def __init__(self, save_path, print_to_console):
        self.save_path = save_path
        self.print_to_console = print_to_console
        self.f = open(save_path, 'w')

    def __call__(self, string):
        self.f.write(string)
        if self.print_to_console:
            print(string, end='')

    def close(self):
        self.f.close()


def save_beam_results(checkpoint_folder, beam_results, models, data_sets, score_name, ndigits=2, print_to_console=True):
    save_path = os.path.join(checkpoint_folder, f'collected_{score_name.replace(".", "-")}_scores.txt')
    beam_sizes = sorted(list(beam_results.keys()))
    model_name_len = max([len(m) for m in models])
    nspace = 10

    # save and print collected results
    save_and_print = SaveAndPrint(save_path, print_to_console)

    for data in data_sets:
        # record the best model scores
        best_score = (0, '', -1)    # (score, model_name, beam_size)
        save_and_print('-' * 50 + f' {data} ' + '-' * 50)
        save_and_print('\n')
        save_and_print('\n')
        save_and_print(' ' * (model_name_len + nspace) + (' ' * nspace).join(map(lambda x: f'beam{x}', beam_sizes)))
        save_and_print('\n')
        for model in models:
            # get the model scores from different beam sizes
            model_scores = [beam_results[bs].setdefault(model, {}).setdefault(data, None) or ('-' * (3 + ndigits))
                            for bs in beam_sizes]

            # get the best model scores
            model_scores_filtered = list(filter(lambda x: not isinstance(x[0], str), zip(model_scores, beam_sizes)))
            if model_scores_filtered:
                current_best = sorted(model_scores_filtered, key=lambda x: x[0], reverse=True)[0]
                if current_best[0] > best_score[0]:
                    best_score = (current_best[0], model, current_best[1])

            # print the row for this model with different beam sizes
            save_and_print(f'{model}' + ' ' * (model_name_len + nspace - len(model))
                           + (' ' * nspace).join(map(lambda x: f'{x * 100:.{ndigits}f}' if not isinstance(x, str) else x,
                                                     model_scores)))
            save_and_print('\n')
        save_and_print('\n')
        # print the best scores and corresponding setups
        save_and_print(f'{data} - {red("best score")}: {best_score[0] * 100:.{ndigits}f} '
                       f'(checkpoint_{best_score[1]}.pt, beam size {best_score[2]})')
        save_and_print('\n')
        save_and_print('\n')

    save_and_print.close()


if __name__ == '__main__':
    args = parse_args()

    beam_results = collect_final_scores(args.checkpoints, args.data_sets, args.models, args.score_name)
    save_beam_results(args.checkpoints, beam_results, args.models, args.data_sets, args.score_name,
                      ndigits=args.ndigits,
                      print_to_console=True)
