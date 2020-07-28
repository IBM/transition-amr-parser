import os
from tqdm import tqdm
import argparse
from transition_amr_parser.state_machine import (
    AMRStateMachine,
    get_spacy_lemmatizer
)
from transition_amr_parser.io import (
    read_amr,
    read_tokenized_sentences,
    read_action_scores,
    read_rule_stats,
    write_tokenized_sentences,
    write_rule_stats
)
from collections import Counter, defaultdict
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
    parser.add_argument(
        "--in-rule-stats",
        help="Input rule stats for statistics building",
        type=str,
    )
    parser.add_argument(
        "--out-rule-stats",
        help="Output rule stats from mined actions",
        type=str,
    )
    parser.add_argument(
        "--entity-rules",
        help="entity rules",
        type=str
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


def merge_rules(sentences, actions, rule_stats, entity_rules=None):

    # generate rules to restrict action space by stack content
    actions_by_stack_rules = rule_stats['possible_predicates']
    for token, counter in rule_stats['possible_predicates'].items():
        actions_by_stack_rules[token] = Counter(counter)

    spacy_lemmatizer = get_spacy_lemmatizer()

    possible_predicates = defaultdict(lambda: Counter())
    for index, sentence_actions in tqdm(enumerate(actions), desc='merge rules'):

        tokens = sentences[index]

        # Initialize machine
        state_machine = AMRStateMachine(
            tokens,
            actions_by_stack_rules=actions_by_stack_rules,
            spacy_lemmatizer=spacy_lemmatizer,
            entity_rules=entity_rules
        )

        for action in sentence_actions:
            # NOTE: At the oracle, possible predicates are collected before
            # PRED/COPY decision (tryConfirm action) we have to take all of
            # them into account
            position, mpositions = \
                state_machine.get_top_of_stack(positions=True)
            if action.startswith('PRED'):
                node = action[4:-1]
                possible_predicates[tokens[position]].update([node])
                if mpositions:
                    mtokens = ','.join([tokens[p] for p in mpositions])
                    possible_predicates[mtokens].update([node])

            elif action == 'COPY_LEMMA':
                lemma, _ = state_machine.get_top_of_stack(lemma=True)
                node = lemma
                possible_predicates[tokens[position]].update([node])
                if mpositions:
                    mtokens = ','.join([tokens[p] for p in mpositions])
                    possible_predicates[mtokens].update([node])

            elif action == 'COPY_SENSE01':
                lemma, _ = state_machine.get_top_of_stack(lemma=True)
                node = f'{lemma}-01'
                possible_predicates[tokens[position]].update([node])
                if mpositions:
                    mtokens = ','.join([tokens[p] for p in mpositions])
                    possible_predicates[mtokens].update([node])

            # execute action
            state_machine.applyAction(action)

    # TODO: compare with old stats
    out_rule_stats = rule_stats
    out_rule_stats['possible_predicates'] = {
        key: dict(value) for key, value in possible_predicates.items()
    }

    return out_rule_stats


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
    # Load rule stats
    if args.in_rule_stats:
        rule_stats = read_rule_stats(args.in_rule_stats)

    # Modify
    # merge --in-actions and --in-scored-actions and store in --out-actions
    if args.merge_mined:
        # sanity checks
        assert args.in_tokens, "--merge-mined requires --in-tokens"
        assert args.in_actions, "--merge-mined requires --in-actions"
        assert args.in_rule_stats, "--merge-mined requires --in-rule-stats"
        assert args.out_rule_stats, "--merge-mined requires --out-rule-stats"
        if args.in_actions:
            assert len(actions) == len(scored_actions)
        print(f'Merging {args.out_actions} and {args.in_scored_actions}')

        # actions
        actions = merge_actions(actions, scored_actions)

    # fix actions split by whitespace arguments 
    if args.fix_actions:
        actions = fix_actions_split_by_spaces(actions)

    # merge rules
    if args.merge_mined:
        out_rule_stats = merge_rules(sentences, actions, rule_stats, entity_rules=args.entity_rules)
        print(f'Merging {args.out_rule_stats} and {args.in_rule_stats}')

    # Write
    # actions
    if args.out_actions:
        dirname = os.path.dirname(args.out_actions)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        write_tokenized_sentences(
            args.out_actions,
            actions,
            separator='\t'
        )
        print(f'Wrote {args.out_actions}')

    # rule stats
    if args.out_rule_stats:
        write_rule_stats(args.out_rule_stats, out_rule_stats)
        print(f'Wrote {args.out_rule_stats}')

    # AMR
    if args.out_amr:
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                fid.write(amr.toJAMRString())
