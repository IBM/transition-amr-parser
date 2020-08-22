import os
from tqdm import tqdm


def check_one_seq(seq):
    started = False
    arc_chunks = []    # continuous arc pointer value sequences
    arcs = []
    arc_chunks_starts = []
    for i, v in enumerate(seq):
        if v != -1:
            if not started:
                arcs.append(v)
                started = True
                arc_chunks_starts.append(i)
            else:
                arcs.append(v)
        else:
            if started:
                arc_chunks.append(arcs)
                arcs = []
                started = False
            else:
                pass
    assert len(arc_chunks) == len(arc_chunks_starts)
    
    # check: whether all the continuous arc pointer sequences are monotonically increasing
    # which means: no double arcs between two nodes
    pass_check1 = True
    for arcs in arc_chunks:
        arcs_diff = [b - a for a, b in zip(arcs, arcs[1:])]
        if len(arcs_diff) > 0 and min(arcs_diff) <= 0:
            pass_check1 = False
    
    # check: whether any of the arcs points to the most recent node itself
    # which means: no self-loops
    pass_check2 = True
    for nid, arcs in zip(arc_chunks_starts, arc_chunks):
        assert nid - 1 >= 0
        assert max(arcs) <= nid - 1 
        if max(arcs) == nid - 1:
            pass_check2 = False
    
    return arc_chunks, arc_chunks_starts, pass_check1, pass_check2

    
if __name__ == '__main__':
    # training data
    data_dir = '/dccstor/jzhou1/work/EXP/data/o3-prefix_act-states/oracle'
    data_dir = '/dccstor/jzhou1/work/EXP/data/o5-prefix_act-states/oracle'

    data_dir = '/dccstor/jzhou1/work/EXP/data/o3_act-states/oracle'
    data_dir = '/dccstor/jzhou1/work/EXP/data/o5_act-states/oracle'
    
    data_file = os.path.join(data_dir, 'dev.actions_pos')
    data_file = os.path.join(data_dir, 'test.actions_pos')
    data_file = os.path.join(data_dir, 'train.actions_pos')
    
    # model generated data
    beam = 1
    data_dir = ('/dccstor/jzhou1/work/EXP/'
                'exp_o5_roberta-large-top24_act-pos_vmask1_shiftpos1_ptr-layer456-head1_tis-embtop-comadd-bp1/'
                f'models_ep120_seed42/beam{beam}')
#     data_dir = ('/dccstor/jzhou1/work/EXP/'
#                 'exp_o5_roberta-large-top24_act-pos_vmask0_shiftpos1_ptr-layer456-head1_tis-embtop-comadd-bp1/'
#                 f'models_ep120_seed42/beam{beam}')
#     data_dir = ('/dccstor/jzhou1/work/EXP/'
#                 'exp_o5_roberta-large-top24_act-pos_vmask0_shiftpos1_ptr-layer6-head1_tis-embtop-comadd-bp1/'
#                 f'models_ep120_seed42/beam{beam}')
    data_dir = ('/dccstor/jzhou1/work/EXP/'
                'exp_o5_no-mw_roberta-large-top24_act-pos_vmask1_shiftpos1_ptr-layer6-head1_tis-embtop-comadd-bp1/'
                f'models_ep120_seed42/beam{beam}')
    
    data_file = os.path.join(data_dir, 'valid_checkpoint_last.carc.actions_pos')
#     data_file = os.path.join(data_dir, 'test_checkpoint_last.actions_pos')
    
    with open(data_file, 'r') as f:
        lines = [list(map(int, line.strip().split('\t'))) for line in f if line.strip()]
    
    for i, seq in tqdm(enumerate(lines)):
        arc_chunks, arc_chunks_starts, pass_check1, pass_check2 = check_one_seq(seq)
        # import pdb; pdb.set_trace()
        if not pass_check1 or not pass_check2:
#         if not pass_check2:
            print(f'Exception found: line {i}')
            import pdb; pdb.set_trace()
   