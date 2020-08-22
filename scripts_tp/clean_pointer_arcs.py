import os
import sys

from fairseq_ext.utils import clean_pointer_arcs


if __name__ == '__main__':
    actions_prefix = sys.argv[1]
    print(f'Actions data prefix: {actions_prefix}')
    
    with open(f'{actions_prefix}.actions_nopos', 'r') as fa, \
         open(f'{actions_prefix}.actions_pos', 'r') as fb, \
         open(f'{actions_prefix}.actions', 'r') as fc, \
         open(f'{actions_prefix}.carc.actions_nopos', 'w') as ga, \
         open(f'{actions_prefix}.carc.actions_pos', 'w') as gb, \
         open(f'{actions_prefix}.carc.actions', 'w') as gc:
        cleaned_ids = []
        for i, (actions_nopos, actions_pos, actions) in enumerate(zip(fa, fb, fc)):
            if actions_nopos.strip():
                actions_nopos = actions_nopos.strip().split('\t')
                actions_pos = list(map(int, actions_pos.strip().split('\t')))
                actions = actions.strip().split('\t')
                actions_nopos_new, actions_pos_new, actions_new, invalid_idx = clean_pointer_arcs(actions_nopos,
                                                                                                  actions_pos,
                                                                                                  actions)
                ga.write('\t'.join(actions_nopos_new) + '\n')
                gb.write('\t'.join(map(str, actions_pos_new)) + '\n')
                gc.write('\t'.join(actions_new) + '\n')
                
                if invalid_idx:
                    cleaned_ids.append(i)
#                     import pdb; pdb.set_trace()
    print(f'cleaned {len(cleaned_ids)} actions sequences on the arc pointers:')
    print(cleaned_ids)