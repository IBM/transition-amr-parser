# AMR parsing given a sentence and a model
import time
import os
import signal
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

from transition_amr_parser.action_pointer.o8_state_machine import (
    AMRStateMachine,
#     DepParsingStateMachine,
    get_spacy_lemmatizer
)
from transition_amr_parser.clbar import yellow_font
from transition_amr_parser.io import (
    writer,
    read_tokenized_sentences,
    read_rule_stats,
)

def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser')
    # Multiple input parameters
    parser.add_argument(
        "--in-sentences",
        help="file space with carriare return separated sentences",
        type=str
    )
    parser.add_argument(
        "--in-actions",
        help="file space with carriage return separated sentences",
        type=str
    )
    parser.add_argument(
        "--in-pred-entities",
        type=str,
        default="person,thing",
        help="comma separated list of entity types that can have pred"
    )
    parser.add_argument(
        "--out-amr",
        help="parsing model",
        type=str
    )
    # state machine rules
    parser.add_argument(
        "--action-rules-from-stats",
        help="Use oracle statistics to restrict possible actions",
        type=str
    )
    parser.add_argument(
        "--verbose",
        help="verbose mode",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--machine-type",
        choices=['AMR', 'dep-parsing'],
        default='AMR'
    )
    parser.add_argument(
        "--separator",
        default='\t'
    )

    parser.add_argument(
        "--step-by-step",
        help="pause after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--pause-time",
        help="time waited after each step, default is manual",
        type=int
    )
    parser.add_argument(
        "--clear-print",
        help="clear command line before each print",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--offset",
        help="start at given sentence number (starts at zero)",
        type=int
    )
    parser.add_argument(
        "--random-up-to",
        help="sample randomly from a max number",
        type=int
    )
    parser.add_argument(
        "--no-whitespace-in-actions",
        action='store_true',
        help="Assume whitespaces normalized to _ in PRED"
    )
    parser.add_argument(
        "--out-bio-tags",
        type=str,
        help="Output AMR info as BIO tags (PRED and ADDNODE actions)"
    )
    args = parser.parse_args()

    # Argument pre-processing
    if args.random_up_to:
        args.offset = np.random.randint(args.random_up_to)

    # force verbose
    if not args.verbose:
        args.verbose = bool(args.step_by_step)

    # Sanity checks
    assert args.in_sentences
    assert args.in_actions

    return args


def reduce_counter(counts, reducer):
    """
    Returns a new counter from an existing one where keys have been mapped
    to in  many-to-one fashion and counts added
    """
    new_counts = Counter()
    for key, count in counts.items():
        new_key = reducer(key)
        new_counts[new_key] += count
    return new_counts


def restrict_action(state_machine, raw_action, pred_counts, rule_violation):

    # Get valid actions
    valid_actions = state_machine.get_valid_actions()

    # Fallback for constrained PRED actions
    if 'PRED' in raw_action:
        if 'PRED' not in valid_actions:
            # apply restrictions to predict actions
            # get valid predict actions
            valid_pred_actions = [
                a for a in valid_actions if 'PRED' in a
            ]
            if valid_pred_actions == []:
                # no rule found for this token, try copy
                token, tokens = state_machine.get_top_of_stack()
                if tokens:
                    token = ",".join(tokens)
                # reasign raw action
                raw_action = f'PRED({token.lower()})'
                pred_counts.update(['token OOV'])
            elif raw_action not in valid_pred_actions:
                # not found, get most common match
                # reasign raw action
                raw_action = valid_pred_actions[0]
                pred_counts.update(['alignment OOV'])
            else:
                pred_counts.update(['matches'])
    elif (
        raw_action not in valid_actions and
        raw_action.split('(')[0] not in valid_actions
    ):

        # note-down rule violation
        token, _ = state_machine.get_top_of_stack()
        rule_violation.update([(token, raw_action)])

    return raw_action


def get_bio_from_machine(state_machine, raw_action):
    annotations = {}
    if (
        raw_action.startswith('PRED') or
        raw_action.startswith('ADDNODE') or
        raw_action in ['COPY_SENSE01', 'COPY_LEMMA']
    ):
        if raw_action == 'COPY_SENSE01':
            lemma, _ = state_machine.get_top_of_stack(lemma=True)
            raw_action = f'PRED({lemma}-01)'
        elif raw_action == 'COPY_LEMMA':
            lemma, _ = state_machine.get_top_of_stack(lemma=True)
            raw_action = f'PRED({lemma})'
        token, tokens = state_machine.get_top_of_stack(positions=True)
        tokens = tuple(tokens) if tokens else [token]
        for token in tokens:
            if token in annotations:
                raise Exception('Overlapping annotations')
            annotations[token] = raw_action
    return annotations


def get_bio_tags(state_machine, bio_alignments):
    bio_tags = []
    prev_label = None
    for idx, token in enumerate(state_machine.tokens[:-1]):
        if idx in bio_alignments:
            if prev_label == bio_alignments[idx]:
                tag = f'I-{bio_alignments[idx]}'
            else:
                tag = f'B-{bio_alignments[idx]}'
            prev_label = bio_alignments[idx]
        else:
            prev_label = None
            tag = 'O'
        bio_tags.append((token, tag))

    return bio_tags


class FakeAMRParser():
    """
    Fake parser that uses precomputed sequences of sentences and corresponding
    actions
    """

    def __init__(self, logger=None, machine_type='AMR',
                 from_sent_act_pairs=None, actions_by_stack_rules=None,
                 no_whitespace_in_actions=False, entities_with_preds=None):

        assert not no_whitespace_in_actions, \
            '--no-whitespace-in-actions deprected'

        # Dummy mode: simulate parser from pre-computed pairs of sentences
        # and actions
        self.actions_by_sentence = {
            " ".join(sent): actions for sent, actions in from_sent_act_pairs
        }
        self.logger = logger
        self.sent_idx = 0
        self.actions_by_stack_rules = actions_by_stack_rules
        self.no_whitespace_in_actions = no_whitespace_in_actions
        self.machine_type = machine_type
        self.entities_with_preds = entities_with_preds
        # initialize here for speed
        self.spacy_lemmatizer = get_spacy_lemmatizer()

        # counters
        self.pred_counts = Counter()
        self.rule_violation = Counter()

    def parse_sentence(self, sentence_str):
        """
        sentence_str is a string with whitespace separated tokens
        """

        # simulated actions given by a parsing model
        key =  " ".join(sentence_str)
        assert sentence_str in self.actions_by_sentence, \
            "Fake parser has no actions for sentence: %s" % sentence_str
        actions = self.actions_by_sentence[sentence_str]
        tokens = sentence_str.split()
        # Initialize state machine
        if self.machine_type == 'AMR':
            state_machine = AMRStateMachine(
                tokens,
                actions_by_stack_rules=self.actions_by_stack_rules,
                spacy_lemmatizer=self.spacy_lemmatizer,
                entities_with_preds=self.entities_with_preds
            )
        elif self.machine_type == 'dep-parsing':
            state_machine = DepParsingStateMachine(tokens)

        # this will store AMR parsing as BIO tag (PRED, ADDNODE)
        bio_alignments = {}

        # execute parsing model
#         while not state_machine.is_closed:

#             # Print state (pause if solicited)
#             self.logger.update(self.sent_idx, state_machine)

#             if len(actions) <= state_machine.time_step:
#                 # if machine is not propperly closed hard exit
#                 print(yellow_font(
#                     f'machine not closed at step {state_machine.time_step}'
#                 ))
#                 raw_action = 'CLOSE'
#             else:
#                 # get action from model
#                 raw_action = actions[state_machine.time_step]

#             # restrict action space according to machine restrictions and
#             # statistics
#             if self.machine_type == 'AMR':
#                 raw_action = restrict_action(
#                     state_machine,
#                     raw_action,
#                     self.pred_counts,
#                     self.rule_violation
#                 )

#                 # update bio tags from AMR
#                 bio_alignments.update(
#                     get_bio_from_machine(state_machine, raw_action)
#                 )

#             # Update state machine
#             state_machine.applyAction(raw_action)

        # CLOSE action is internally managed
        state_machine.apply_actions(actions if actions[-1] == 'CLOSE' else actions + ['CLOSE'])

        # build bio tags
#         bio_tags = get_bio_tags(state_machine, bio_alignments)
        bio_tags = []

        # count one sentence more
        self.sent_idx += 1

        return state_machine, bio_tags


class Logger():

    def __init__(self, step_by_step=None, clear_print=None, pause_time=None,
                 verbose=False):

        self.step_by_step = step_by_step
        self.clear_print = clear_print
        self.pause_time = pause_time
        self.verbose = verbose or self.step_by_step

        if step_by_step:

            # Set traps for system signals to die graceful when Ctrl-C used

            def ordered_exit(signum, frame):
                """Mesage user when killing by signal"""
                print("\nStopped by user\n")
                exit(0)

            signal.signal(signal.SIGINT, ordered_exit)
            signal.signal(signal.SIGTERM, ordered_exit)

    def update(self, sent_idx, state_machine):

        if self.verbose:
            if self.clear_print:
                # clean screen each time
                os.system('clear')
            print(f'sentence {sent_idx}\n')
            print(state_machine)
            # step by step mode
            if self.step_by_step:
                if self.pause_time:
                    time.sleep(self.pause_time)
                else:
                    input('Press any key to continue')


def main():

    # Argument handling
    args = argument_parser()

    # Get data
    sentences = read_tokenized_sentences(args.in_sentences, separator=args.separator)
    entities_with_preds = args.in_pred_entities.split(",")
    
    # Initialize logger/printer
    logger = Logger(
        step_by_step=args.step_by_step,
        clear_print=args.clear_print,
        pause_time=args.pause_time,
        verbose=args.verbose
    )

    # generate rules to restrict action space by stack content
    if args.action_rules_from_stats:
        rule_stats = read_rule_stats(args.action_rules_from_stats)
        actions_by_stack_rules = rule_stats['possible_predicates']
        for token, counter in rule_stats['possible_predicates'].items():
            actions_by_stack_rules[token] = Counter(counter)

    else:
        actions_by_stack_rules = None

    # Fake parser built from actions
    actions = read_tokenized_sentences(
        args.in_actions,
        separator=args.separator
    )
    assert len(sentences) == len(actions)
    parsing_model = FakeAMRParser(
        from_sent_act_pairs=zip(sentences, actions),
        machine_type=args.machine_type,
        logger=logger,
        actions_by_stack_rules=actions_by_stack_rules,
        no_whitespace_in_actions=args.no_whitespace_in_actions,
        entities_with_preds=entities_with_preds
    )

    # Get output AMR writer
    if args.out_amr:
        amr_write = writer(args.out_amr)
    if args.out_bio_tags:
        bio_write = writer(args.out_bio_tags)

    # Loop over sentences
    for sent_idx, tokens in tqdm(enumerate(sentences), desc='parsing'):

        # fast-forward until desired sentence number
        if args.offset and sent_idx < args.offset:
            continue

        # parse
        # NOTE: To simulate the real endpoint, input provided as a string of
        # whitespace separated tokens
        machine, bio_tags = parsing_model.parse_sentence(" ".join(tokens))

#         if sent_idx == 5:
#             import pdb; pdb.set_trace()

        # store output AMR
        if args.out_bio_tags:
            tag_str = '\n'.join([f'{to} {ta}' for to, ta in bio_tags])
            tag_str += '\n\n'
            bio_write(tag_str)
        if args.out_amr:
            amr_write(machine.amr.toJAMRString())

    if (
        getattr(parsing_model, "rule_violation") and
        parsing_model.rule_violation
    ):
        print(yellow_font("There were one or more action rule violations"))
        print(parsing_model.rule_violation)

    if args.action_rules_from_stats:
        print("Predict rules had following statistics")
        print(parsing_model.pred_counts)

    # close output writers
    if args.out_amr:
        amr_write()
    if args.out_bio_tags:
        bio_write()


if __name__ == '__main__':
    main()
