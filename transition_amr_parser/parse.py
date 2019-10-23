# AMR parsing given a sentence and a model 
from types import FunctionType
import json
import time
import os
import signal
import argparse
import re
from collections import Counter, defaultdict

from tqdm import tqdm

from transition_amr_parser.state_machine import AMRStateMachine
from transition_amr_parser.data_oracle import writer


is_url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


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
        "--out-amr",
        help="parsing model",
        type=str
    )
    parser.add_argument(
        "--in-model",
        help="parsing model",
        type=str
    )
    parser.add_argument(
        "--in-rule-stats",
        help="alignment statistics needed for the rule component",
        type=str
    )
    parser.add_argument(
        "--verbose",
        help="verbose mode",
        action='store_true',
        default=False
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

    args = parser.parse_args()

    # Argument pre-processing
    if args.random_up_to:
        import numpy as np
        args.offset = np.random.randint(args.random_up_to)

    # force verbose
    if not args.verbose:
        args.verbose = bool(args.step_by_step)

    # Sanity checks
    assert args.in_sentences or args.in_sentence_list
    assert args.in_actions or args.in_model
    # Not done yet
    if args.in_model:
        raise NotImplementedError()

    return args


def read_propbank(propbank_file):

    # Read frame argument description
    arguments_by_sense = {}
    with open(propbank_file) as fid:
        for line in fid:
            line = line.rstrip()
            sense = line.split()[0]
            arguments  = [
                re.match('^(ARG.+):$', x).groups()[0]
                for x in line.split()[1:] if re.match('^(ARG.+):$', x)
            ]
            arguments_by_sense[sense] = arguments

    return arguments_by_sense


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def token_reader(file_path):
    with open(file_path) as fid:
        for line in fid:
            yield line.rstrip().split()

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


def main():

    frames_path = '/dccstor/ramast1/_DATA/AMR/abstract_meaning_representation_amr_2.0/data/frames/propbank-frame-arg-descr.txt'
    arguments_by_sense = read_propbank(frames_path)

    # Argument handling
    args = argument_parser()

    # Get data generators
    sentences = token_reader(args.in_sentences)
    actions = token_reader(args.in_actions)

    # set orderd exit
    if args.step_by_step:
        signal.signal(signal.SIGINT, ordered_exit)
        signal.signal(signal.SIGTERM, ordered_exit)

    # Get copy stats if provided
    if args.in_rule_stats:
        with open(args.in_rule_stats) as fid:
            rule_stats = json.loads(fid.read())
            # Fix counter
            for rule in ['sense_by_token', 'lemma_by_token']:
                rule_stats[rule] = {
                    key: Counter(value)
                    for key, value in rule_stats[rule].items()
                }

    # Output AMR
    if args.out_amr:
        amr_write = writer(args.out_amr) 

    # Hard attention stats
    action_counts = Counter()
    action_tos_counts = Counter()

    sent_idx = -1
    for sent_tokens, sent_actions in tqdm(zip(sentences, actions)):

        # keep count of sentence index
        sent_idx += 1
        if args.offset and sent_idx < args.offset:
            continue

        # Initialize state machine
        amr_state_machine = AMRStateMachine(
            sent_tokens,
            rule_stats=rule_stats
        )
   
        # process each
        for raw_action in sent_actions:

            # Collect statistics
            if amr_state_machine.stack:
                stack0 = amr_state_machine.stack[-1]
                if stack0 in amr_state_machine.merged_tokens:
                    tos_token = " ".join(
                        amr_state_machine.amr.tokens[i -1] 
                        for i in amr_state_machine.merged_tokens[stack0]
                    )
                else:
                    tos_token = amr_state_machine.amr.tokens[stack0 - 1]
                action_tos_counts.update([(raw_action, tos_token)])
            action_counts.update([raw_action])
    
            # Print state
            if args.verbose:
                if args.clear_print:
                    # clean screen each time
                    os.system('clear')
                print(f'sentence {sent_idx}\n')
                print(amr_state_machine)
     
                # step by step mode
                if args.step_by_step:
                    if args.pause_time:
                        time.sleep(args.pause_time)
                    else:    
                        input('Press any key to continue')

            # Update machine
            amr_state_machine.applyAction(raw_action)

        # Output AMR
        if args.out_amr:
            amr_write(amr_state_machine.amr.toJAMRString())
 
    # DEBUG
#     cosa = reduce_counter(action_tos_counts, lambda x: x if x[0].startswith('PRED') else None)
#     senses = list(arguments_by_sense.keys())
#     cosa2 = reduce_counter(action_tos_counts, lambda x: (x[1], x[0]) if (x[0].startswith('PRED') and x[0][5:-1] not in senses) else None)
#     # Counts by surface token when not in senses
#     surface_counts = defaultdict(lambda: Counter())
#     for key, count in cosa2.items():
#         if key is not None:
#             surface_counts[key[0]][key[1][5:-1]] = count
#     cosa3 = reduce_counter(action_tos_counts, lambda x: (x[1], x[0][5:-1]) if (x[0].startswith('PRED') and x[0][5:-1] not in senses) else None)
    # DEBUG

    # Output AMR
    if args.out_amr:
        amr_write()
