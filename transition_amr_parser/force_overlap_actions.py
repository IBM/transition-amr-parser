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
                        
def main(args):

    fpactions = open(args.in_pred)
    ffactions = open(args.in_force)
    fwindows = open(args.in_windows)
    ffout = open(args.out_force, 'w')

    all_windows = [eval(line.strip()) for line in fwindows]
    all_actions = [ line.strip().split() for line in fpactions ]
    all_force_actions = [ eval(line.strip()) for line in ffactions ]

    window_of_interest = args.in_widx

    if window_of_interest == 0:
        return
    
    all_actions_per_token = []
    for actions in all_actions:
        actions_per_token = []
        this_token_actions = []
        for action in actions:
            this_token_actions.append(action)
            if action in ['SHIFT','CLOSE_SENTENCE']:
                actions_per_token.append(this_token_actions)
                this_token_actions = []
        all_actions_per_token.append(actions_per_token)

    fidx = 0
    pidx = 0
    for (i, _) in enumerate(all_windows):
        if len(all_windows[i]) > window_of_interest:
            this_window = all_windows[i][window_of_interest]
            prev_window = all_windows[i][window_of_interest-1]
            actions_per_token = all_actions_per_token[pidx]
            
            start_idx = this_window[0] - prev_window[0]
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
            in_force_actions = all_force_actions[fidx]

            out_force_actions.extend(in_force_actions[len(overlap_actions):])
            
            #there can be a sanity check here
            
            ffout.write(str(out_force_actions)+"\n")

            fidx += 1
            
        if len(all_windows[i]) >= window_of_interest:
            pidx += 1
            
        #print(str(windows)+"\t"+str(sum([len(actions) for actions in actions_per_token])) )

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


