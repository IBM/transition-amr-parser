from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re

arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')


def decrement_pointers_to_future(actions, pos, ignored):

    for (i,action) in enumerate(actions):
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if i > pos and int(idx)+ignored >= pos:
                idx = str(int(idx) - 1)
            actions[i] = action[:3]+"("+idx+","+lbl+")"

def main(args):

    window_size = args.window_size
    window_overlap = args.window_overlap
    print("window size ",window_size)
    print("window overlap ",window_overlap)

    factions = open(args.oracle_dir + "/" + args.data_split + ".actions")
    ffactions = open(args.oracle_dir + "/" + args.data_split + ".force_actions")
    ftokens = open(args.oracle_dir + "/" + args.data_split + ".en")

    all_actions = [ line.strip().split() for line in factions ]
    all_force_actions = [ eval(line.strip()) for line in ffactions ]
    all_tokens = [ line.strip().split() for line in ftokens ]

    assert len(all_actions) == len(all_force_actions)
    assert len(all_actions) == len(all_tokens)

    all_sentence_ends = []
    all_actions_per_token = []
    for actions in all_actions:
        sentence_ends = []
        actions_per_token = []
        this_token_actions = []
        for action in actions:
            this_token_actions.append(action)
            if action in ['SHIFT','CLOSE_SENTENCE']:
                if action == 'CLOSE_SENTENCE':
                    sentence_ends.append(len(actions_per_token))
                actions_per_token.append(this_token_actions)
                this_token_actions = []
        if len(sentence_ends) == 0:
            sentence_ends.append(len(this_token_actions)-1)

        new_sentence_ends = []
        for (i,send) in enumerate(sentence_ends):
            if i > 0:
                gap = send - new_sentence_ends[-1]
                while gap > window_size:
                    if gap < window_size * 2:
                        new_sentence_ends.append(new_sentence_ends[-1] + int(gap/2))
                    else:
                        new_sentence_ends.append(new_sentence_ends[-1] + window_size)
                    gap = send - new_sentence_ends[-1]
            new_sentence_ends.append(send)

        if len(sentence_ends) != len(new_sentence_ends):
            print("added extra sentence ends to break windows")
            print(sentence_ends)
            print(new_sentence_ends)
        
        all_sentence_ends.append(new_sentence_ends)
        all_actions_per_token.append(actions_per_token)

    max_num_windows = 0
    fout_tokens = []
    fout_actions = []
    fout_force_actions = []
    fout_windows = open(args.oracle_dir + "/" + args.data_split + ".windows", 'w')
    for (tokens, actions_per_token, force_actions, sentence_ends) in zip(all_tokens, all_actions_per_token, all_force_actions, all_sentence_ends):
        windows = []
        this_window = (0,0)
        start = 0
        while this_window[-1] < (len(tokens)-1):
            #find last sentence end within window size from start
            end = start
            for send in sentence_ends:
                if (send-start) < window_size:
                    end = send
                else:
                    break
            

            this_window = (start,end)
            windows.append(this_window)
            #find first start sentence within overlap in this_window
            for send in sentence_ends:
                if (this_window[-1] - send) < window_overlap:
                    start = send + 1
                    if start > this_window[0]:
                        break

        fout_windows.write(str(windows)+"\n")
                
        if len(windows) > max_num_windows:
            for n in range(max_num_windows,len(windows)):
                fout_tokens.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".en",'w'))
                fout_actions.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".actions", 'w'))
                fout_force_actions.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".force_actions", 'w'))
            max_num_windows = len(windows)

        for (widx,(start,end)) in enumerate(windows):
            start_action_index = sum([len(acts) for acts in actions_per_token[:start]]) if start else 0
            out_tokens = "\t".join(tokens[start:end+1])
            fout_tokens[widx].write(out_tokens+"\n")
            
            out_actions = ""
            window_actions = []
            for idx in range(start,end+1):
                for action in actions_per_token[idx]:
                    if arc_regex.match(action):
                        (idx, lbl) = arc_regex.match(action).groups()
                        idx = str(int(idx) - start_action_index)
                        action = action[:3]+"("+idx+","+lbl+")"
                    window_actions.append(action)

            good_actions = []
            ignored = 0
            for (i,action) in enumerate(window_actions):
                if arc_regex.match(action):
                    (idx, lbl) = arc_regex.match(action).groups()
                    if int(idx) < 0:
                        decrement_pointers_to_future(window_actions, i, ignored)
                        ignored += 1 
                        continue
                good_actions.append(action)

            #sanity check
            for action in good_actions:
                if arc_regex.match(action):
                    (idx, lbl) = arc_regex.match(action).groups()
                    if arc_regex.match(good_actions[int(idx)]) or good_actions[int(idx)] in ['SHIFT','ROOT','CLOSE_SENTENCE']:
                        import ipdb; ipdb.set_trace()
                        print("*****bad pointer to from " + action + " to " + good_actions[int(idx)])
                        
            out_actions += "\t".join(good_actions)
            fout_actions[widx].write(out_actions+"\n")
            out_force_actions = str(force_actions[start:end+1])
            fout_force_actions[widx].write(out_force_actions+"\n")
            
        #print(str(windows)+"\t"+str(sum([len(actions) for actions in actions_per_token])) )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--oracle-dir",
        help="path to the oracle directory",  
        type=str
    )
    parser.add_argument(
        "--data-split",
        help="data split train/dev/test",  
        type=str
    )

    parser.add_argument(
        "--window-size",
        help="size of sliding window",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--window-overlap",
        help="size of overlap between sliding windows",
        default=50,
        type=int,
    )
    args = parser.parse_args()
    main(args)


