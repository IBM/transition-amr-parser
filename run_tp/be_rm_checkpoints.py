"""
remove all checkpoints that are with name "checkpoints[0-9]*",
but are not linked by the best models via model selection.
Also keep the last checkpoint.
"""
import os
import glob
import sys
import re


if __name__ == '__main__':
    checkpoint_folder = sys.argv[1]
    being_linked = []
    # get the checkpoints that are linked as the best selected models
    for fname in glob.glob(f'{checkpoint_folder}/checkpoint_*'):
        if os.path.islink(fname):
            being_linked.append(os.readlink(fname))

    # do not go forward if we don't have best 5 models linked, or the flag for postprocessing is not raised
    if len(being_linked) < 5 or not os.path.exists(os.path.join(checkpoint_folder, 'model-selection_stage3-done')):
        print('program abort due to model selection and best model link not finished')
        sys.exit()

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
    for fname in glob.glob(f'{checkpoint_folder}/checkpoint[0-9]*'):
        fname_base = os.path.basename(fname)
        epoch_num, = checkpoint_re.match(fname_base).groups()
        epoch_num = int(epoch_num)
        if epoch_num == max_epoch:
            continue
        if fname_base not in being_linked:
            # print(fname_base)
            os.remove(fname)
