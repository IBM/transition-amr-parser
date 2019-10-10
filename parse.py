# AMR parsing given a sentence and a model 
import os
import argparse
import re
from state_machine import Transitions


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


def token_reader(file_path):
    with open(file_path) as fid:
        for line in fid:
            yield line.rstrip().split()


if __name__ == '__main__':

    args = argument_parser()

    # Get data generators
    sentences = token_reader(args.in_sentences)
    actions = token_reader(args.in_actions)

    sent_idx = 0
    for sent_tokens, sent_actions in zip(sentences, actions):

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
    
            # Rename PRED
            if raw_action.startswith('PRED'):
                raw_action = raw_action.replace('PRED', 'CONFIRM')
            elif raw_action == 'UNSHIFT':
                raw_action = 'SWAP'
    
            # FIXME: This action is wrong
            if raw_action == 'SHIFT' and amr_state_machine.buffer == []:
                raw_action = 'CLOSE'
    
            # Update state
            if raw_action in ['SHIFT', 'REDUCE', 'MERGE', 'SWAP', 'INTRODUCE', 
                              'CLOSE', 'UNSHIFT']:
                # argument-less action
                getattr(amr_state_machine, raw_action)()
            else:
                # action with arguments 
                fetch = re.match('([A-Z]+)\((.*)\)', raw_action)
                action, arguments = fetch.groups()
                arguments = arguments.split(',')
                getattr(amr_state_machine, action)(*arguments)

            sent_idx += 1
    
            if args.step_by_step:
                input('Press any key to continue')
