from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re

arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')


def decrement_pointers_to_future(action_lists, li, ai, ignored):

    pos = sum([len(alist) for alist in action_lists[:li]]) + ai + 1
    for (i,action_list) in enumerate(action_lists):
        for (j, action) in enumerate(action_list):
            if i > li or ( i==li and j>ai):
                if arc_regex.match(action):
                    (idx, lbl) = arc_regex.match(action).groups()
                    if int(idx)+ignored >= pos:
                        idx = str(int(idx) - 1)
                        action_lists[i][j] = action[:3]+"("+idx+","+lbl+")"

def sanity_check(actions):
    for action in actions:
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if arc_regex.match(actions[int(idx)]) or actions[int(idx)] in ['SHIFT','ROOT','CLOSE_SENTENCE']:
                import ipdb; ipdb.set_trace()
                print("*****bad pointer to from " + action + " to " + actions[int(idx)])                        


def force_overlap(actions, force_actions, start_idx):
    
    actions_per_token = []
    this_token_actions = []
    for action in actions:
        this_token_actions.append(action)
        if action in ['SHIFT','CLOSE_SENTENCE']:
            actions_per_token.append(this_token_actions)
            this_token_actions = []
    
    start_action_index = sum([len(acts) for acts in actions_per_token[:start_idx]]) if start_idx else 0
            
    out_actions = ""
    overlap_actions = []
    ignored = 0
    for ti in range(start_idx,len(actions_per_token)):
        useful_actions = []
        for (ai,action) in enumerate(actions_per_token[ti]):
            if arc_regex.match(action):
                (idx, lbl) = arc_regex.match(action).groups()
                idx = str(int(idx) - start_action_index)
                if int(idx) >= 0:
                    useful_actions.append(action[:3]+"("+idx+","+lbl+")")
                else:
                    decrement_pointers_to_future(actions_per_token,ti,ai,ignored)
                    ignored += 1
            else:
                useful_actions.append(action)
        overlap_actions.append(useful_actions)

    flat_actions = []
    for actions in overlap_actions:
        flat_actions.extend(actions)
    sanity_check(flat_actions)
                
    out_force_actions = overlap_actions
    
    if force_actions is not None:
        out_force_actions.extend(force_actions[len(overlap_actions):])
        
    #there can be a sanity check here
    return out_force_actions
    
                

def force_overlap_all(all_windows, all_actions, all_force_actions, in_widx):

    all_out_force_actions = []
    
    fidx = 0
    pidx = 0
    for (i, _) in enumerate(all_windows):
        if len(all_windows[i]) > in_widx:
            this_window = all_windows[i][in_widx]
            prev_window = all_windows[i][in_widx-1]
            actions = all_actions[pidx]
            force_actions = all_force_actions[fidx]
            
            start_idx = this_window[0] - prev_window[0]

            out_force_actions = force_overlap(actions, force_actions, start_idx)            
            
            all_out_force_actions.append(out_force_actions)
            
            fidx += 1
            
        if len(all_windows[i]) >= in_widx:
            pidx += 1

    return all_out_force_actions

def make_forced_overlap(in_pred, in_force, in_windows, in_widx, out_force):
    
    fpactions = open(in_pred)
    ffactions = open(in_force)
    fwindows = open(in_windows)

    all_windows = [eval(line.strip()) for line in fwindows]
    all_actions = [ line.strip().split() for line in fpactions ]
    all_force_actions = [ eval(line.strip()) for line in ffactions ]

    window_of_interest = in_widx

    ffactions.close()
    ffout = open(out_force, 'w')
    
    if window_of_interest == 0:
        return

    all_out_force_actions = force_overlap_all(all_windows, all_actions, all_force_actions, in_widx)

    for force_actions in all_out_force_actions:
        ffout.write(str(force_actions) + "\n")
        
def main(args):
    make_forced_overlap(args.in_pred, args.in_force, args.in_windows, args.in_widx, args.out_force)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--in-pred",
        help="input pred actions to be forced in next window",  
        type=str
    )
    parser.add_argument(
        "--in-force",
        help="input force actions to be updated",  
        type=str
    )
    parser.add_argument(
        "--in-windows",
        help="info about sliding window",
        type=str,
    )
    parser.add_argument(
        "--out-force",
        help="output force actions",
        type=str,
    )
    parser.add_argument(
        "--in-widx",
        help="index of the window to be updated",
        default=1,
        type=int,
    )
    
    args = parser.parse_args()
    main(args)


