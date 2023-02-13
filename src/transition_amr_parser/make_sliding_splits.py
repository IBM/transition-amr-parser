from argparse import ArgumentParser
from transition_amr_parser.io import read_blocks
import re
import warnings
arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')


def decrement_pointers_to_future(actions, pos, ignored):

    for (i,action) in enumerate(actions):
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if i > pos and int(idx)+ignored >= pos:
                idx = str(int(idx) - 1)
            actions[i] = action[:3]+"("+idx+","+lbl+")"


def check_pointers(actions):
    #sanity check                                                                                                                                                              
    for action in actions:
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if arc_regex.match(actions[int(idx)]) or actions[int(idx)] in ['SHIFT','ROOT','CLOSE_SENTENCE']:
                import ipdb; ipdb.set_trace()
                print("*****bad pointer from " + action + " to " + actions[int(idx)])
            
def get_windows(tokens, window_size=300, window_overlap=200, sentence_ends=None):

    windows = []
    this_window = (0,0)
    start = 0
    loop_count = 0
    while this_window[-1] < (len(tokens)-1):
        loop_count+=1
        if loop_count>len(tokens):
            raise Exception('Number of loops exceeded number of tokens, might lead to infinite loop')
        #find last sentence end within window size from start
        end = start
        for send in sentence_ends:
            if (send-start) < window_size:
                end = send
            else:
                break
            

        this_window = (start,end)
        if len(windows)>0 and windows[-1]==this_window:
            warnings.warn('This window is same as last window')

        windows.append(this_window)
        #find first start sentence within overlap in this_window
        for send in sentence_ends:
            if (this_window[-1] - send) < window_overlap:
                start = send + 1
                if start > this_window[0]:
                    break

    return windows

def adjust_sentence_ends(sentence_ends, window_size):
    
    new_sentence_ends = []
    
    for (i,send) in enumerate(sentence_ends):
        
        if i > 0:
            prev_send = new_sentence_ends[-1]
        else:
            prev_send = -1
            
        gap = send - prev_send
        while gap > window_size:
            if gap < window_size * 2:
                new_sentence_ends.append(prev_send + int(gap/2))
            else:
                new_sentence_ends.append(prev_send + window_size)
            prev_send = new_sentence_ends[-1]
            gap = send - prev_send
        
        new_sentence_ends.append(send)
        
    return new_sentence_ends
    
def get_good_window_actions(start, end, actions_per_token):

    start_action_index = sum([len(acts) for acts in actions_per_token[:start]]) if start else 0
    
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
    token_per_actions_window = []
    for (i,action) in enumerate(window_actions):
        if arc_regex.match(action):
            (idx, lbl) = arc_regex.match(action).groups()
            if int(idx) < 0:
                decrement_pointers_to_future(window_actions, i, ignored)
                ignored += 1
                continue
        token_per_actions_window.append(i)
        good_actions.append(action)

    #sanity check
    check_pointers(good_actions)

    return good_actions,token_per_actions_window


def main(args):

    window_size = args.window_size
    window_overlap = args.window_overlap
    print("window size ",args.window_size)
    print("window overlap ",args.window_overlap)
    num_actions_truncated = 0
    num_tokens_truncated = 0

    factions = open(args.oracle_dir + "/" + args.data_split + ".actions")
    ffactions = open(args.oracle_dir + "/" + args.data_split + ".force_actions")
    ftokens = open(args.oracle_dir + "/" + args.data_split + ".en")

    all_actions = [ line.strip().split('\t') for line in factions ]
    all_force_actions = [ eval(line.strip()) for line in ffactions ]
    all_tokens = [ line.strip().split() for line in ftokens ]

    assert len(all_actions) == len(all_force_actions)
    assert len(all_actions) == len(all_tokens)

    all_sentence_ends = []
    all_actions_per_token = []
    for actions in all_actions:
        check_pointers(actions)
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
        if (len(actions_per_token)-1) not in sentence_ends:
            sentence_ends.append(len(actions_per_token)-1)

        new_sentence_ends = adjust_sentence_ends(sentence_ends, args.window_size)

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
    count = 0
    for (tokens, actions_per_token, force_actions, sentence_ends) in zip(all_tokens, all_actions_per_token, all_force_actions, all_sentence_ends):
        count+=1
        windows = get_windows(tokens, args.window_size, args.window_overlap, sentence_ends)
        
        fout_windows.write(str(windows)+"\n")
                
        if len(windows) > max_num_windows:
            for n in range(max_num_windows,len(windows)):
                if args.train_sliding and args.data_split=='train' and len(fout_tokens)>0 :
                    fout_tokens.append(fout_tokens[-1])
                    fout_actions.append(fout_actions[-1])
                    fout_force_actions.append(fout_force_actions[-1])

                else:
                    fout_tokens.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".en",'w'))
                    fout_actions.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".actions", 'w'))
                    fout_force_actions.append(open(args.oracle_dir + "/" + args.data_split + "_" + str(n) + ".force_actions", 'w'))
            max_num_windows = len(windows)

        for (widx,(start,end)) in enumerate(windows):
            
            out_tok = tokens[start:end+1]
            out_tokens = "\t".join(out_tok)
            fout_tokens[widx].write(out_tokens+"\n")
            
            # token_per_action_window = []
            # for idx in range(start,end+1):
            #     for action in actions_per_token[idx]:
            #         token_per_action_window.append(idx)
            
            good_actions,token_per_action_window = get_good_window_actions(start, end, actions_per_token)
            assert len(good_actions)==len(token_per_action_window)

            
            # if len(out_tok)>900 and len(good_actions)<900:
            #     num_tokens_truncated +=1
            #     out_tok = out_tok[:900]
            #     while tok_idx
            #         good_actions = good_actions[0:out_tok[-1]+1]

            if len(good_actions)>=900:
                num_actions_truncated +=1

                #find the last CLOSE_SENTENCE
                # good_actions = good_actions[:899]
                # token_per_action = token_per_action[:899]
                # i = len(good_actions)-1
                # while good_actions[i] != 'CLOSE_SENTENCE' and i > 0:
                #     i -= 1
                # good_actions = good_actions[:i+1]
                # out_tok = out_tok[:token_per_action[i]+1]
                
                popped_toks = []
                last_tok = 0
                if good_actions[-1] == 'CLOSE_SENTENCE':
                    close_sent = True
                else:
                    close_sent = False
                while len(token_per_action_window)>=900 or last_tok in popped_toks:
                   pop_tok = token_per_action_window.pop()
                   popped_toks.append(pop_tok)
                   pop_act = good_actions.pop()
                   last_tok = token_per_action_window[-1]

                out_tok = out_tok[0:last_tok+1]
                if close_sent:
                    good_actions.append('CLOSE_SENTENCE')
                
                
                


            out_actions = "\t".join(good_actions)
            assert len(good_actions)>1,"empty actions"
            assert len(out_tok)>1,"empty out tokens "
            fout_actions[widx].write(out_actions+"\n")
            if force_actions is not None:
                out_force_actions = str(force_actions[start:end+1])
            else:
                out_force_actions = str(force_actions)

            fout_force_actions[widx].write(out_force_actions+"\n")
            
        #print(str(windows)+"\t"+str(sum([len(actions) for actions in actions_per_token])) )
    print("num of samples with actions truncated ",num_actions_truncated)
    print("num of samples with tokens truncated ",num_tokens_truncated)

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
        default=300,
        type=int,
    )
    parser.add_argument(
        "--window-overlap",
        help="size of overlap between sliding windows",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--train-sliding",
        help="if train data is split into sliding windows , it can be concatenated",
        action='store_true'
    )
    args = parser.parse_args()
    main(args)


