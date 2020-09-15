import os
import re
import argparse


# scores: model decoding setting
beam_size = 1
data_set = 'valid'

# results file name regex
results_re = re.compile(rf'{data_set}_checkpoint([0-9]+)\.(.+)')
# smatch_re = re.compile(r'dec-checkpoint([0-9]+)\.smatch')
# smatch_re_wiki = re.compile(r'dec-checkpoint([0-9]+)\.wiki\.smatch')

# results file content regex
smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')
las_results_re = re.compile(r'UAS: ([0-9\.]+) % LAS: ([0-9\.]+) %')


def parse_args():
    parser = argparse.ArgumentParser(description='Organize model results')
    parser.add_argument('--checkpoints', type=str, default='/dccstor/jzhou1/work/EXP/exp_debug/models_ep120_seed42',
                        help='folder containing saved model checkpoints for a single training')
    parser.add_argument('--link_best', type=int, default=5,
                        help='number of best smatch models to link (0 means do not link any)')
    parser.add_argument('--score_name', type=str, default='wiki.smatch',
                        help='postfix of the score files')
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

    return results


def get_scores_from_folder(score_folder, score_name):

    # Get results available in this folder
    scores = {}
    for dfile in os.listdir(score_folder):

        # if not a results file, skip
        if not results_re.match(dfile):
            continue

        epoch_number, sname = results_re.match(dfile).groups()

        if sname != score_name:
            continue

        # get score
        score = get_score_from_log(f'{score_folder}/{dfile}', score_name)
        if score is not None:
            scores[int(epoch_number)] = score

    return scores


def rank_scores(scores, score_name):

    if score_name == 'las':
        sort_idx = 1
    else:
        sort_idx = 0
    ranked_scores = sorted(scores.items(), key=lambda x: x[1][sort_idx], reverse=True)
    return ranked_scores


def collect_checkpoint_results(checkpoint_folder, score_name):
    """Collect the scores for all the checkpoints in the folder."""
    score_folder = os.path.join(checkpoint_folder, f'beam{beam_size}')
    scores = get_scores_from_folder(score_folder, score_name)
    ranked_scores = rank_scores(scores, score_name)
    # ranked_scores is a list of tuples, with each entry being (epoch, [score])
    return ranked_scores


def save_ranked_scores(checkpoint_folder, ranked_scores, score_name, print_to_console=True, print_best=3):
    save_path = os.path.join(checkpoint_folder, f'epoch_{score_name.replace(".", "-")}_ranks.txt')
    epochs, scores = zip(*ranked_scores)
    epoch_min, epoch_max = min(epochs), max(epochs)
    head_info = f'model ranking based on {score_name} (with beam size {beam_size} on {data_set} data); '
    if len(epochs) == epoch_max - epoch_min + 1:
        # epochs are continuous
        epoch_info = f'epochs: {epoch_min} - {epoch_max}'
    else:
        # epochs are not continuous
        epoch_info = 'epochs: ' + ' '.join(map(str, sorted(epochs, reverse=True)))
    # save and print ranked results
    with open(save_path, 'w') as f:
        f.write(head_info + epoch_info + '\n')
        if print_to_console:
            print('-' * 80)
            print(head_info + epoch_info)
        for rank, (epoch, score) in enumerate(ranked_scores):
            f.write(f'checkpoint{epoch}.pt {score}' + '\n')
            if print_to_console and rank < print_best:
                print(f'checkpoint{epoch}.pt {score}')
    if print_to_console:
        print('-' * 80)


def link_top_models(ranked_scores, checkpoint_folder, score_name, link_best=5):
    # works when 'link_best' is larger than 'len(ranked_scores)'
    for rank, epoch_score in enumerate(ranked_scores[:link_best], start=1):
        epoch, score = epoch_score
        target_best = os.path.join(os.path.realpath(checkpoint_folder),
                                   f'checkpoint_{score_name.replace(".", "-")}_best{rank}.pt')
        source_best = f'checkpoint{epoch}.pt'
        # We may have created a link before to a worse model --> remove it
        if (
            os.path.islink(target_best) and
            os.path.basename(os.path.realpath(target_best)) !=
                source_best
        ):
            os.remove(target_best)
        if not os.path.islink(target_best):
            os.symlink(source_best, target_best)

    return rank


if __name__ == '__main__':
    args = parse_args()

    ranked_scores = collect_checkpoint_results(args.checkpoints, args.score_name)

    if args.link_best > 0:
        rank = link_top_models(ranked_scores, args.checkpoints, args.score_name, args.link_best)

    save_ranked_scores(args.checkpoints, ranked_scores, args.score_name,
                       print_to_console=True, print_best=args.link_best)
