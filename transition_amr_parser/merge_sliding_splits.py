from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re
from glob import glob

arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')


def decrement_pointers_to_future(actions, pos, ignored):

    for (i,action) in enumerate(actions):
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if i > pos and int(idx)+ignored >= pos:
                idx = str(int(idx) - 1)
            actions[i] = action[:3]+"("+idx+","+lbl+")"

def increment_pointers_to_future(actions, pos, inserted, offset):

    for (i,action) in enumerate(actions):
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if i > pos and int(idx)-inserted >= pos+offset:
                idx = str(int(idx) + 1)
            actions[i] = action[:3]+"("+idx+","+lbl+")"
            
def merge_actions(actions, more_actions, overlap_start):

    token_idx = 0
    ret_actions = []
    overlap_action_start = None
    for action in actions:
        if token_idx == overlap_start and overlap_action_start is None:
            overlap_action_start = len(ret_actions)
        if action in ['SHIFT','CLOSE_SENTENCE']:
            token_idx += 1
        ret_actions.append(action)

    if token_idx == overlap_start and overlap_action_start is None:
        overlap_action_start = len(ret_actions)
        
    if token_idx < overlap_start:
        while token_idx < overlap_start:
            ret_actions.append('SHIFT')
            token_idx += 1
        overlap_action_start = len(ret_actions)
        
    for (i,action) in enumerate(more_actions):
        if arc_regex.match(action):
            (idx,lbl) = arc_regex.match(action).groups()
            idx = str(int(idx) + overlap_action_start)
            more_actions[i] = action[:4] + idx + "," + lbl + ")"

    j = overlap_action_start
    inserted = 0
    for (i,action) in enumerate(more_actions):
        while j < len(actions) and actions[j] != action:
            j += 1
            increment_pointers_to_future(more_actions, inserted, i, overlap_action_start)
            inserted += 1
        if j >= len(actions):
            ret_actions.extend(more_actions[i:])
            break
        j += 1

    return ret_actions
            
def main(args):

    fwindows = open(args.input_dir + "/" + args.data_split + ".windows")
    all_windows = [ eval(line.strip()) for line in fwindows ]

    fnames = glob(args.input_dir + "/" + args.data_split + "_[0-9]*.actions")
    fin_actions = [open(args.input_dir + "/" + args.data_split + "_" + str(i) + ".actions") for (i,_) in enumerate(fnames)]
    fnames = glob(args.input_dir + "/" + args.data_split + "_[0-9]*.en")
    fin_tokens = [open(args.input_dir + "/" + args.data_split + "_" + str(i) + ".en") for (i,_) in enumerate(fnames)]
    fout_actions = open(args.input_dir + "/" + args.data_split + "_merged.actions",'w')
    fout_tokens = open(args.input_dir + "/" + args.data_split + "_merged.en",'w')
    
    for windows in all_windows:
        actions = []
        tokens = []
        for (i,window) in enumerate(windows):
            if i == 0:
                actions = fin_actions[i].readline().strip().split()
                tokens = fin_tokens[i].readline().strip().split()
            else:
                actions = merge_actions(actions, fin_actions[i].readline().strip().split(), overlap_start=window[0])
                window_tokens = fin_tokens[i].readline().strip().split()
                tokens.extend( window_tokens[len(tokens)-window[0]:] )

        fout_actions.write("\t".join(actions)+"\n")
        fout_tokens.write("\t".join(tokens)+"\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="path to the oracle directory",  
        type=str
    )
    parser.add_argument(
        "--data-split",
        help="data split train/dev/test",  
        type=str
    )

    args = parser.parse_args()
    main(args)


