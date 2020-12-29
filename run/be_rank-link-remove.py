"""
Rank the model based on evaluation results so far,
and link the best models, save ranking results.

Remove all checkpoints that are with name "checkpoints[0-9]*",
but are not linked by the best models via model selection.
Also keep the last checkpoint.
"""
import argparse
import os
import glob
import re

from bb_rank_model import collect_checkpoint_results, link_top_models, save_ranked_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Organize model results')
    parser.add_argument('--checkpoints', type=str, default='/dccstor/jzhou1/work/EXP/exp_debug/models_ep120_seed42',
                        help='folder containing saved model checkpoints for a single training')
    parser.add_argument('--link_best', type=int, default=5,
                        help='number of best smatch models to link (0 means do not link any)')
    parser.add_argument('--score_name', type=str, default='wiki.smatch',
                        help='postfix of the score files')
    parser.add_argument('--remove', type=int, default=1,
                        help='whether to remove unlinked checkpoints (except the last one)')
    args = parser.parse_args()
    return args


def rm_checkpoints(checkpoint_folder, link_best, scored_epochs=None):
    being_linked = []
    # get the checkpoints that are linked as the best selected models
    for fname in glob.glob(f'{checkpoint_folder}/checkpoint_*'):
        if os.path.islink(fname):
            being_linked.append(os.readlink(fname))

    # do not go forward if we don't have best 5 models linked, or the flag for postprocessing is not raised
    # if len(being_linked) < 5 or not os.path.exists(os.path.join(checkpoint_folder, 'model-selection_stage3-done')):
    #     print('program abort due to model selection and best model link not finished')
    #     import sys
    #     sys.exit()

    # do not go forward if we haven't collected enough best checkpoints
    if len(being_linked) < link_best:
        print(f'Have not collected {link_best} best models --- not removing any checkpoints')
        return

    # keep the last checkpoint always (as a way to tell in other scripts how many epochs are trained)
    # get the last checkpoint epoch number saved
    checkpoint_re = re.compile(r'checkpoint([0-9]+).pt')
    epochs = []
    for fname in glob.glob(f'{checkpoint_folder}/checkpoint[0-9]*'):
        fname_base = os.path.basename(fname)
        epoch_num, = checkpoint_re.match(fname_base).groups()
        epochs.append(int(epoch_num))

    max_epoch = max(epochs)

    # remove checkpoints that are (unlinked && not the last one)
    # NOTE do not remove the checkpoints that have not beed decoded and scored
    #      as this would cause some issue during the separate evaluation process during training
    print('Removing unlinked and not-the-last checkpoints --- ')
    for fname in glob.glob(f'{checkpoint_folder}/checkpoint[0-9]*'):
        fname_base = os.path.basename(fname)
        epoch_num, = checkpoint_re.match(fname_base).groups()
        epoch_num = int(epoch_num)
        if epoch_num == max_epoch:
            continue
        if scored_epochs is not None and epoch_num not in scored_epochs:
            continue
        if fname_base not in being_linked:
            # print(fname_base)
            os.remove(fname)

    return


if __name__ == '__main__':
    args = parse_args()

    ranked_scores = collect_checkpoint_results(args.checkpoints, args.score_name)

    if args.link_best > 0:
        rank = link_top_models(ranked_scores, args.checkpoints, args.score_name, args.link_best)

    save_ranked_scores(args.checkpoints, ranked_scores, args.score_name,
                       print_to_console=True, print_best=args.link_best)

    # remove checkpoints
    if args.remove:
        # get the checkpoint epoch numbers that have been decoded and scored
        scored_epochs = [ep for ep, sc in ranked_scores]
        rm_checkpoints(args.checkpoints, args.link_best, scored_epochs)
