import argparse
import os
from tqdm import tqdm
from copy import deepcopy
import json
from ipdb import set_trace
from numpy.random import choice
# TAP
from transition_amr_parser.amr_machine import AMRStateMachine
from transition_amr_parser.gold_subgraph_align import (
    match_amrs,
    BadAlignModeSample
)
from transition_amr_parser.amr import AMR


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='sandbox to play actions of a state machine'
    )
    parser.add_argument(
        '--in-amr',
        type=str,
        help='Input AMR (will do random search)'
    )
    parser.add_argument(
        '-i', '--in-state-json',
        type=str,
        help='Output of AMRStateMAchine.save(state=True)'
    )
    parser.add_argument(
        '--context',
        type=int,
        default=1,
        help='set_trace(context) from ipdb'
    )
    parser.add_argument(
        '--trace',
        action='store_true',
        help='step by step while following history'
    )
    parser.add_argument(
        '--trace-random',
        action='store_true',
        help='step by step while random action choice'
    )
    parser.add_argument(
        '--trace-step',
        type=int,
        help='Stop at this time step'
    )
    parser.add_argument(
        '--max-step',
        type=int,
        help='After these many steps sample valid actions randomly'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='run machine multiple times'
    )
    args = parser.parse_args()

    assert bool(args.in_amr) ^ bool(args.in_state_json)
    if bool(args.in_amr):
        assert not bool(args.max_step)

    return args


def main(args):

    # get state and start machine
    if bool(args.in_state_json):

        # run recorded state sequence until asked to stop (--max-step)
        state = json.loads(open(args.in_state_json).read())['state']
        machine = AMRStateMachine.from_config(args.in_state_json)
        tokens = state['tokens']
        if 'gold_amr' in state:
            gold_amr = AMR.from_penman(state['gold_amr'])
        else:
            gold_amr = None
        action_history = state['action_history']

    else:

        # run random search on a gold AMR
        machine = AMRStateMachine()
        with open(args.in_amr) as fid:
            gold_amr = AMR.from_penman(fid.read())
        tokens = gold_amr.tokens
        action_history = []

    # run machine one or more times
    for i in tqdm(range(args.repeat)):

        # align mode <==> gold_amr is not None
        machine.reset(tokens, gold_amr=gold_amr)

        # play recorder actions fully or until time step args.max_step
        for action in action_history:

            # stop playing and switch to random sampling
            if (
                args.max_step is not None
                and len(machine.action_history) > args.max_step
            ):
                break

            # breakpoint
            if (
                (
                    args.trace_step is not None
                    and len(machine.action_history) >= args.trace_step
                ) or args.trace
            ):
                os.system('clear')
                print(machine)
                set_trace(context=args.context)
                machine.get_valid_actions()
                print()

            # machine valid actions inconsistent with recorded actions
            if action not in machine.get_valid_actions():
                os.system('clear')
                print(machine)
                set_trace(context=args.context)
                machine.get_valid_actions()
                print()

            try:
                machine.update(action)
            except BadAlignModeSample as exception:
                os.system('clear')
                print(machine)
                set_trace(context=args.context)

        # continue with random choice of actions
        while not machine.is_closed:
            action = choice(machine.get_valid_actions())
            if (
                (
                    args.trace_step is not None
                    and len(machine.action_history) >= args.trace_step
                ) or args.trace_random
            ):
                os.system('clear')
                print(machine)
                set_trace(context=args.context)
                machine.get_valid_actions()
                print()

            try:
                machine.update(action)
            except BadAlignModeSample as exception:
                os.system('clear')
                print(machine)
                set_trace(context=args.context)
                print()

        # align model sanity check
        if gold_amr is not None:

            missing_nodes, missing_edges, excess_edges = match_amrs(machine)

            if missing_nodes:
                os.system('clear')
                print(machine)
                set_trace(context=args.context)
                print()

            if bool(missing_edges):
                print(machine)
                set_trace(context=args.context)
                print()
            elif bool(excess_edges):
                print(machine)
                set_trace(context=args.context)
                print()

        # start a a new machine, but keep a copy for comparison
        old_machine = deepcopy(machine)
        if args.in_state_json:
            machine = AMRStateMachine.from_config(args.in_state_json)
        else:
            machine = AMRStateMachine.from_config()


if __name__ == '__main__':
    main(argument_parsing())
