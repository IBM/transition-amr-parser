import os
import sys

from fairseq.data import Dictionary

from fairseq_ext.amr_spec.action_info_binarize import (binarize_actstates_tofile,
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

    split = 'dev'
    split = 'train'

    en_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.en'
    actions_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.actions'
    actions_dict = Dictionary.load(
        '/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/databin/dict.actions_nopos.txt'
    )
    out_file_pref = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/tmp/{split}.en-actions.actions'

    # en_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/data/o3align_roberta-base-last_act-noeos-states-2LAroot/oracle/{split}.en'
    # actions_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/data/o3align_roberta-base-last_act-noeos-states-2LAroot/oracle/{split}.actions'
    # actions_dict = Dictionary.load(
    #     '/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/data/o3align_roberta-base-last_act-noeos-states-2LAroot/processed/dict.actions_nopos.txt'
    #     )
    # out_file_pref = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/data/o3align_roberta-base-last_act-noeos-states-2LAroot/processed/{split}.en-actions.actions'

    os.makedirs(os.path.dirname(out_file_pref), exist_ok=True)

    # binarize_actstates_tofile(en_file, actions_file, out_file_pref, actions_dict=actions_dict)
    binarize_actstates_tofile_workers(en_file, actions_file, out_file_pref, actions_dict=actions_dict,
                                      num_workers=num_workers)

    os.system(f'ls -lh {os.path.dirname(out_file_pref)}')

    tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions = \
        load_actstates_fromfile(out_file_pref, actions_dict)

    import pdb; pdb.set_trace()
