import sys
import os
from transition_amr_parser.io import read_amr
from transition_amr_parser.amr_machine import AMRStateMachine, print_and_break
from numpy.random import choice


def main():
    amrs = read_amr(sys.argv[1], generate=True)
    machine = AMRStateMachine()

    for amr in amrs:
        machine.reset(amr.tokens, gold_amr=amr)

        while not machine.is_closed:
            os.system('clear')
            print()
            print_and_break(machine)
            possible_actions = machine.get_valid_actions()
            action = choice(possible_actions)
            machine.update(action)


if __name__ == '__main__':
    main()
