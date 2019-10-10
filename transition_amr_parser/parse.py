# AMR parsing given a sentence and a model 
import os
import signal
import argparse
import re
from transition_amr_parser.state_machine import Transitions


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
        help="file space with carriare return separated sentences",
        type=str
    )
    parser.add_argument(
        "--in-model",
        help="parsing model",
        type=str
    )
    parser.add_argument(
        "--step-by-step",
        help="pause after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--offset",
        help="start at given sentence number (starts at zero)",
        type=int
    )

    args = parser.parse_args()

    # Sanity checks
    assert args.in_sentences or args.in_sentence_list
    assert args.in_actions or args.in_model
    # Not done yet
    if args.in_model:
        raise NotImplementedError()

    return args


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def token_reader(file_path):
    with open(file_path) as fid:
        for line in fid:
            yield line.rstrip().split()


def main():

    # Argument handling
    args = argument_parser()

    # Get data generators
    sentences = token_reader(args.in_sentences)
    actions = token_reader(args.in_actions)

    # set orderd exit
    if args.step_by_step:
        signal.signal(signal.SIGINT, ordered_exit)
        signal.signal(signal.SIGTERM, ordered_exit)

    sent_idx = -1
    for sent_tokens, sent_actions in zip(sentences, actions):


        sent_idx += 1
        if args.offset and sent_idx < args.offset:
            continue

        # Initialize state machine
        amr_state_machine = Transitions(sent_tokens)
    
        # process each
        for raw_action in sent_actions:
    
            # Print state
            if args.step_by_step:
                os.system('clear')
            print(amr_state_machine)

            # Update machine
            amr_state_machine.applyAction(raw_action)

            if args.step_by_step:
                input('Press any key to continue')
