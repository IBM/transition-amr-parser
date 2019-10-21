# AMR parsing given a sentence and a model 
import time
import os
import signal
import argparse
import re

from tqdm import tqdm

from transition_amr_parser.state_machine import AMRStateMachine
from transition_amr_parser.data_oracle import writer


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

    if args.random_up_to:
        import numpy as np
        args.offset = np.random.randint(args.random_up_to)

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

    # Output AMR
    if args.out_amr:
        amr_write = writer(args.out_amr) 

    sent_idx = -1
    for sent_tokens, sent_actions in tqdm(zip(sentences, actions)):

        # keep count of sentence index
        sent_idx += 1
        if args.offset and sent_idx < args.offset:
            continue

        # Initialize state machine
        amr_state_machine = AMRStateMachine(sent_tokens)

        # Output AMR
        if args.out_amr:
            amr_write(amr_state_machine.amr.toJAMRString())
    
        # process each
        for raw_action in sent_actions:
    
            # Print state
            if args.step_by_step or args.clear_print:
                os.system('clear')
            print(f'sentence {sent_idx}\n')
            print(amr_state_machine)

            # Update machine
            amr_state_machine.applyAction(raw_action)

            if args.step_by_step:
                if args.pause_time:
                    time.sleep(args.pause_time)
                else:    
                    input('Press any key to continue')

    # Output AMR
    if args.out_amr:
        amr_write()
