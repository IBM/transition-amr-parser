from fairseq.data import Dictionary

import sys
# import importlib
# sys.path.insert(0, '..')
# importlib.import_module('fairseq_ext')
# sys.path.pop(0)
from neural_parser.fairseq_ext.amr_spec.action_info_binarize import binarize_actstates_tolist, binarize_actstates_tolist_workers


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 1

    split = 'dev'
    # split = 'train'

    en_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.en'
    actions_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.actions'
    actions_dict = Dictionary.load(
        '/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/databin/dict.actions_nopos.txt'
    )

    # tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors = binarize_actstates_tolist(en_file, actions_file,
    #                                                                                 actions_dict=actions_dict)
    # TODO not working for num_workers > 1
    tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions = \
        binarize_actstates_tolist_workers(en_file, actions_file, actions_dict=actions_dict, num_workers=num_workers)

    import pdb
    pdb.set_trace()
