import os
import argparse
from transition_amr_parser.io import (
    read_amr,
    read_tokenized_sentences,
    read_action_scores,
    write_tokenized_sentences
)
from collections import Counter
from smatch import compute_f


def yellow_font(string):
    return "\033[93m%s\033[0m" % string

def argument_parser():

    parser = argparse.ArgumentParser(description='Tool to handle AMR')
    parser.add_argument(
        "--in-amr",
        help="input AMR files in pennman notation",
        type=str,
    )
    parser.add_argument(
        "--out-amr",
        help="output AMR files in pennman notation",
        type=str,
    )
    parser.add_argument(
        "--in-tokens",
        help="tab separated tokens one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--in-scored-actions",
        help="actions and action features pre-appended",
        type=str,
    )
    parser.add_argument(
        "--in-actions",
        help="tab separated actions one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-actions",
        help="tab separated actions one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--merge-mined",
        action='store_true',
        help="--out-actions will contain merge of --in-actions and --in-scored-actions",
    )
    parser.add_argument(
        "--fix-actions",
        action='store_true',
        help="fix actions split by whitespace arguments",
    )
    args = parser.parse_args()

    return args


def print_score_action_stats(scored_actions):
    scores = [0, 0, 0]
    action_count = Counter()
    for sa in scored_actions:
        action_count.update([sa[6]])
        for i in range(3):
            scores[i] += sa[i+1]
    smatch = compute_f(*scores)[2]
    display_str = 'Smatch {:.3f} scored mined {:d} length mined {:d}'.format(
        smatch,
        action_count['score'],
        action_count['length']
    )
    print(display_str)


def fix_actions_split_by_spaces(actions):
               
    # Fix actions split by spaces
    new_actions = []
    num_fixed = 0
    for sent_actions in actions:
        new_sent_actions = []
        index = 0
        while index < len(sent_actions):
            # There can be no actions with a single quote. If one found look
            # until we find another one and assume the sapn covered is a split
            # action
            if len(sent_actions[index].split('"')) == 2:
                start_index = index
                index += 1
                while len(sent_actions[index].split('"')) != 2:
                    index += 1
                new_sent_actions.append(
                    " ".join(sent_actions[start_index:index+1])
                )
                num_fixed += 1
            else:
                new_sent_actions.append(sent_actions[index])
            # increase index    
            index += 1
        new_actions.append(new_sent_actions)

    if num_fixed:
        message_str = (f'WARNING: {num_fixed} actions had to be fixed for '
                        'whitespace split')
        print(yellow_font(message_str))

    return new_actions


def merge_actions(actions, scored_actions):
    created_actions = []
    for index, actions in enumerate(actions):
        if scored_actions[index][6] is not None: 
            created_actions.append(scored_actions[index][7])
        else:
            created_actions.append(actions)
    return created_actions 


def main():

    # Argument handling
    args = argument_parser()

    # Read
    # Load AMR (replace some unicode characters)
    if args.in_amr:
        corpus = read_amr(args.in_amr, unicode_fixes=True)
        amrs = corpus.amrs
    # Load tokens    
    if args.in_tokens:
        sentences = read_tokenized_sentences(args.in_tokens, separator='\t')
    # Load actions i.e. oracle
    if args.in_actions:
        actions = read_tokenized_sentences(args.in_actions, separator='\t')
    # Load scored actions i.e. mined oracle     
    if args.in_scored_actions:
        scored_actions = read_action_scores(args.in_scored_actions)
        # measure performance
        print_score_action_stats(scored_actions)

    # Modify
    # merge --in-actions and --in-scored-actions and store in --out-actions
    if args.merge_mined:
        assert args.in_actions
        if args.in_actions:
            assert len(actions) == len(scored_actions)
        print(f'Merging {args.out_actions} and {args.in_scored_actions}')
        actions = merge_actions(actions, scored_actions)
    # fix actions split by whitespace arguments 
    if args.fix_actions:
        actions = fix_actions_split_by_spaces(actions)

    # Write
    # actions
    if args.out_actions:
        print(f'Wrote {args.out_actions}')
        dirname = os.path.dirname(args.out_actions)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        write_tokenized_sentences(
            args.out_actions,
            actions,
            separator='\t'
        )
    # AMR
    if args.out_amr:
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                fid.write(amr.toJAMRString())
