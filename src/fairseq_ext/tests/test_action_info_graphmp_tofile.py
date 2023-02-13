import os
import sys

from fairseq.data import Dictionary

from fairseq_ext.amr_spec.action_info_binarize_graphmp import (binarize_actstates_tofile,
                                                               binarize_actstates_tofile_workers,
                                                               load_actstates_fromfile)

# import sys
# import importlib
# sys.path.insert(0, '..')
# importlib.import_module('fairseq_ext')
# sys.path.pop(0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 1

    # split = 'dev'
    split = 'train'

    en_file = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser/EXP/data/o5_act-states/oracle/{split}.en'
    actions_file = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser/EXP/data/o5_act-states/oracle/{split}.actions'
    actions_dict = Dictionary.load(
        '/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser/EXP/data/o5_act-states/processed/dict.actions_nopos.txt'
    )
    out_file_pref = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser/tmp/{split}.en-actions.actions'

    os.makedirs(os.path.dirname(out_file_pref), exist_ok=True)

    # res = binarize_actstates_tofile(en_file, actions_file, out_file_pref, actions_dict=actions_dict)
    res = binarize_actstates_tofile_workers(en_file, actions_file, out_file_pref, actions_dict=actions_dict,
                                            num_workers=num_workers)
    print(
        "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            'actions',
            actions_file,
            res['nseq'],
            res['ntok'],
            100 * res['nunk'] / res['ntok'],
            actions_dict.unk_word,
        )
    )

    os.system(f'ls -lh {os.path.dirname(out_file_pref)}')

    tgt_actstates = load_actstates_fromfile(out_file_pref, actions_dict)

    breakpoint()
