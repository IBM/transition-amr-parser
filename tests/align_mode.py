import sys
from tqdm import tqdm
from copy import deepcopy
import json
from transition_amr_parser.amr import AMR
from transition_amr_parser.io import read_amr
from transition_amr_parser.amr_machine import AMRStateMachine, print_and_break
from numpy.random import choice
from collections import defaultdict
from ipdb import set_trace
# from numpy.random import randint


surface_rules = True
if surface_rules:
    from transition_amr_parser.amr_aligner import surface_aligner


class RuleAlignments():

    def __init__(self):
        pass

    def reset(self, gold_amr, machine):

        # NOTE: This is a reference and thus will contain changes in the
        # machine
        self.machine = machine
        self.gold_amr = gold_amr

        tokens = list(gold_amr.tokens)
        id_nodes = list(gold_amr.nodes.items())
        nodeid2token = surface_aligner(
            tokens, id_nodes, cache_key=None
        )[0]

        # store rules activating in this position
        self.rules_by_position = defaultdict(list)
        for nid, items in nodeid2token.items():
            for position, _ in items:
                self.rules_by_position[position].append(gold_amr.nodes[nid])

        # store rules activating in future positions and thus can not be an
        # option at this position
        self.future_rules = defaultdict(set)
        for i in range(len(gold_amr.tokens)):
            for j in range(i + 1, len(gold_amr.tokens)):
                self.future_rules[i] |= set(self.rules_by_position[j])

    def get_valid_actions(self):
        possible_actions = self.machine.get_valid_actions()
        if self.machine.tok_cursor in self.rules_by_position:
            rule_actions = self.rules_by_position[self.machine.tok_cursor]
            valid_actions = list(set(possible_actions) & set(rule_actions))
        else:
            forbidden_rules = self.future_rules[self.machine.tok_cursor]
            valid_actions = list(set(possible_actions) - forbidden_rules)

        if valid_actions:
            return valid_actions
        else:
            return possible_actions


def main():

    trace = False
    trace_if_error = False

    # read AMR instantiate state machine and rules
    amrs = read_amr(sys.argv[1], generate=True)
    machine = AMRStateMachine()
    if surface_rules:
        rules = RuleAlignments()

    # stats to compute Smatch
    num_tries = 0
    num_hits = 0
    num_gold = 0

    # random_index = randint(1000)

    # loop over all AMRs, return basic alignment
    aligned_penman = []
    for index, amr in enumerate(amrs):

        # if index != random_index:
        #    continue

        # start the machine in align mode
        machine.reset(amr.tokens, gold_amr=amr)
        # optionally start the alignment rules
        if surface_rules:
            rules.reset(amr, machine)

        # runs machine until completion
        while not machine.is_closed:

            if trace:
                print_and_break(machine, 1)

            # valid actions
            if surface_rules:
                possible_actions = rules.get_valid_actions()
            else:
                possible_actions = machine.get_valid_actions()

            # random choice among those options
            action = choice(possible_actions)

            # update machine
            machine.update(action)

        # sanity check
        gold2dec = machine.align_tracker.get_flat_map(reverse=True)
        dec2gold = {v[0]: k for k, v in gold2dec.items()}

        # sanity check: all nodes and edges there
        missing_nodes = [
            n for n in machine.gold_amr.nodes if n not in gold2dec
        ]
        if missing_nodes and trace_if_error:
            print_and_break(machine)

        # sanity check: all nodes and edges match
        edges = [
            (dec2gold[e[0]], e[1], dec2gold[e[2]])
            for e in machine.edges if (e[0] in dec2gold and e[2] in dec2gold)
        ]
        missing = set(machine.gold_amr.edges) - set(edges)
        excess = set(edges) - set(machine.gold_amr.edges)
        if bool(missing) and trace_if_error:
            print_and_break(machine)
        elif bool(excess) and trace_if_error:
            print_and_break(machine)

        # edges
        num_tries += len(machine.edges)
        num_hits += len(machine.edges) - len(missing)
        num_gold += len(machine.gold_amr.edges)

        aligned_penman.append(machine.get_annotation())

    precision = num_hits / num_tries
    recall = num_hits / num_gold
    fscore = 2 * (precision * recall) / (precision + recall)
    print(precision, recall, fscore)
    set_trace(context=30)
    print()


def play():

    MAX_HISTORY = None
    STATE_FILE = 'debug.json'
    state = json.loads(open(STATE_FILE).read())['state']
    machine = AMRStateMachine.from_config(STATE_FILE)
    # repeat run to find stochastic errors
    repeat = 1

    for i in tqdm(range(repeat)):

        # read data
        machine.reset(
            state['tokens'],
            gold_amr=AMR.from_penman(state['gold_amr'])
        )

        # play recorder actions fully or until time step MAX_HISTORY
        for action in state['action_history']:

            # stop playing and switch to random sampling
            if (
                MAX_HISTORY is not None
                and len(machine.action_history) > MAX_HISTORY
            ):
                break

            # if len(machine.action_history) > 17:
            #    print_and_break(machine, 1)
            #    machine.get_valid_actions()

            # valid_actions = machine.get_valid_actions()
            # if action not in valid_actions:
            #   set_trace(context=30)

            # print_and_break(machine, 1)
            machine.update(action)

        # continue with random choice of actions
        while not machine.is_closed:
            action = choice(machine.get_valid_actions())
            # print_and_break(machine, 1)
            machine.update(action)

        # sanity check
        gold2dec = machine.align_tracker.get_flat_map(reverse=True)
        dec2gold = {v[0]: k for k, v in gold2dec.items()}

        # sanity check: all nodes and edges there
        missing_nodes = [
            n for n in machine.gold_amr.nodes if n not in gold2dec
        ]
        if missing_nodes:
            print_and_break(machine)

        # sanity check: all nodes and edges match
        edges = [(dec2gold[e[0]], e[1], dec2gold[e[2]]) for e in machine.edges]
        missing = set(machine.gold_amr.edges) - set(edges)
        excess = set(edges) - set(machine.gold_amr.edges)
        if bool(missing):
            print_and_break(machine)
        elif bool(excess):
            print_and_break(machine)

        # print_and_break(machine)
        old_machine = deepcopy(machine)


if __name__ == '__main__':
    #main()
    play()
