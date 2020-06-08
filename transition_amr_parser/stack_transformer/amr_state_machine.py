import argparse
from copy import deepcopy
import torch
from tqdm import tqdm
import json
import signal
import time
import os
from itertools import chain
from collections import defaultdict, Counter

import numpy as np

from transition_amr_parser.state_machine import AMRStateMachine, get_spacy_lemmatizer
from transition_amr_parser.data_oracle import writer
from transition_amr_parser.io import read_rule_stats

from fairseq.debug_tools import timeit, timeit_average, timeit_average


# FONT_COLORORS
FONT_COLOR = {
    'black': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34,
    'magenta': 35, 'cyan': 36, 'light gray': 37, 'dark gray': 90,
    'light red': 91, 'light green': 92, 'light yellow': 93,
    'light blue': 94, 'light magenta': 95, 'light cyan': 96, 'white': 97
}

# BG FONT_COLORORS
BACKGROUND_COLOR = {
    'black': 40,  'red': 41, 'green': 42, 'yellow': 43, 'blue': 44,
     'magenta': 45, 'cyan': 46, 'light gray': 47, 'dark gray': 100,
     'light red': 101, 'light green': 102, 'light yellow': 103,
     'light blue': 104, 'light magenta': 105, 'light cyan': 106,
     'white': 107
}


def white_background(string):
    return "\033[107m%s\033[0m" % string


def red_background(string):
    return "\033[101m%s\033[0m" % string


def black_font(string):
    return "\033[30m%s\033[0m" % string


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


def stack_style(string):
    return black_font(white_background(string))


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def get_word_states(amr_state_machine, sent_tokens, indices=[3, 4, 5]):

    # get legacy indexing of buffer and stack from function
    buffer, stack = amr_state_machine.get_buffer_stack_copy()
    # translate to sane indexing
    buffer = [
        i - 1 if i != -1 else len(sent_tokens) - 1 
        for i in reversed(buffer)
    ] 
    stack = [i - 1 for i in stack]

    # translate to word states
    memory = []
    memory_position = []
    for idx in range(len(sent_tokens)):
        if idx in buffer:
            memory.append(indices[0])
            memory_position.append(buffer.index(idx))
        elif idx in stack:
            memory.append(indices[1])
            memory_position.append(len(stack) - stack.index(idx) - 1)
        else:
            memory.append(indices[2])
            memory_position.append(0)

    return memory, memory_position


signal.signal(signal.SIGINT, ordered_exit)
signal.signal(signal.SIGTERM, ordered_exit)


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
        "--in-rule-stats", 
        help="rule statistics computed by the state machine",
        type=str
    )
    parser.add_argument(
        "--out-word-states", 
        help="stack-transformer word states",
        type=str
    )
    parser.add_argument(
        "--out-valid_actions", 
        help="stack-transformer word states",
        type=str
    )
    parser.add_argument(
        "--offset",
        help="start at given sentence number (starts at zero)",
        type=int
    )
    parser.add_argument(
        "--verbose",
        help="plot information",
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
        help="pause time after each action",
        default=0,
        type=float
    )
    args = parser.parse_args()

    if args.step_by_step or args.pause_time:
        # It is assumed that we want verbosity in this case
        args.verbose = True

    return args


def h5_writer(file_path):

    fid = h5py.File(file_path, 'w', libver='latest')
    sent_idx = 0

    def append_data(content=None):
        """writes to open file"""

        # close file
        if content is None:
            fid.close()
            return

        # add content for a new sentence
        nonlocal sent_idx
        sent_group = fid.create_group(f'sentence-{sent_idx}')
        for idx, arr in enumerate(content):
            sent_group.create_dataset(
                str(idx),
                data=arr,
                shape=arr.shape,
                chunks=arr.shape,
                compression='gzip',
                compression_opts=9
            )

        sent_idx = sent_idx + 1

    return append_data


def get_valid_actions(action_list, amr_state_machine, train_rule_stats,
                      action_by_basic, gold_action, stats): 
    # Get basic actions
    valid_basic_actions = amr_state_machine.get_valid_actions()

    # Expand non-pred actions
    valid_actions = [
        b 
        for a in valid_basic_actions if a != 'PRED' 
        for b in action_by_basic[a]
    ]

    # constrain PRED actions by train stats
    if 'PRED' in valid_basic_actions:

        # Get tokens at top of the stack
        token, merged_tokens = amr_state_machine.get_top_of_stack()
        if merged_tokens:
            # if merged tokens ket present, use it
            merged_tokens = ",".join(merged_tokens)
            if merged_tokens in train_rule_stats['possible_predicates']:
                token = merged_tokens                        

        # add rules from possible predicates
        if token in train_rule_stats['possible_predicates']:
            nodes = train_rule_stats['possible_predicates'][token]
            possible_predicates = [f'PRED({node})' for node in nodes]
            # add to the total number of actions
            valid_actions.extend(possible_predicates)

    # ensure gold action is among the choices if it is a PRED
    if (
        gold_action.startswith('PRED') and 
        gold_action not in train_rule_stats['possible_predicates']
    ):
        stats['missing_pred_count'].update([gold_action])
        valid_actions.append(gold_action)

    stats['fan_out_count'].update([len(valid_actions)])

    if len(valid_actions) == 0:
        stats['missing_action_count'].update([gold_action])
        valid_actions.append(gold_action)

    return np.array([action_list.index(act) for act in valid_actions])


def main():

    # Argument handling
    args = argument_parser()

    # read rules
    train_rule_stats = read_rule_stats(args.in_rule_stats)
    assert 'action_vocabulary' in train_rule_stats
    assert 'possible_predicates' in train_rule_stats
    action_list = list(sorted(train_rule_stats['action_vocabulary'].keys()))

    # get all actions indexec by action root
    action_by_basic = defaultdict(list)
    for action in train_rule_stats['action_vocabulary'].keys():
        key = action.split('(')[0]
        action_by_basic[key].append(action)

    # open file for reading if provided
    write_out_states = h5_writer(args.out_word_states)
    write_out_valid_actions = h5_writer(args.out_valid_actions)

    # Read content
    # TODO: Point to LDC data
    sentences = readlines(args.in_sentences)
    actions = readlines(args.in_actions)
    assert len(sentences) == len(actions)

    # initialize spacy lemmatizer out of the sentence loop for speed
    spacy_lemmatizer = get_spacy_lemmatizer()

    sent_idx = -1
    corpus_states = []
    stats = {
        'missing_pred_count': Counter(),
        'missing_action_count': Counter(),
        'fan_out_count': Counter()
    }
    for sent_tokens, sent_actions in tqdm(
        zip(sentences, actions),
        desc='extracting oracle masks',
        total=len(actions)
    ):

        # keep count of sentence index
        sent_idx += 1
        if args.offset and sent_idx < args.offset:
            continue

        # Initialize state machine
        amr_state_machine = AMRStateMachine(
            sent_tokens,
            spacy_lemmatizer=spacy_lemmatizer
        )
    
        # process each action
        word_states_sent = []
        valid_actions_sent = []
        for raw_action in sent_actions:

            # Store states BEFORE ACTION
            # state of each word (buffer B, stack S, reduced X)
            word_states = get_word_states(amr_state_machine, sent_tokens)

            # Get actions valid for this state 
            valid_actions = get_valid_actions(
                action_list,
                amr_state_machine,
                train_rule_stats,
                action_by_basic,
                raw_action, 
                stats
            )

            # update info
            word_states_sent.append(word_states)
            valid_actions_sent.append(valid_actions)

            # Update machine
            amr_state_machine.applyAction(raw_action)

        # Write states for this sentence
        write_out_states(word_states_sent)
        write_out_valid_actions(valid_actions_sent)

    # inform usre about missing predicates
    for miss in ['missing_pred_count', 'missing_action_count']:
        num_missing = len(stats[miss])
        if num_missing:
            alert_str = f'{num_missing} {miss} rule_stats'
            print(yellow_font(alert_str))

    # inform user about fan-out stats
    mode_fan_out = stats['fan_out_count'].most_common(1)[0][0]
    max_fan_out = max(stats['fan_out_count'].keys())
    alert_str = f'num_actions mode: {mode_fan_out} max: {max_fan_out}'
    print(alert_str)

    # Close file
    write_out_states()
    write_out_valid_actions()


def readlines(file_path):
    with open(file_path) as fid:
        # FIXME: using tab should be an option
        lines = [line.rstrip().split('\t') for line in fid]
    return lines


def print_state_machine(word_states, sent_tokens, sent_idx, amr_state_machine):

    # mask display strings  
    display_items = []
    display_pos = []
    for state, token in zip(word_states, sent_tokens):
        if state[1] == 'B':
            styled_token = " %s" % token
        elif state[1] == 'S':
            styled_token = white_background(" %s" % black_font(token))
        else:    
            styled_token = red_background(" %s" % black_font(token))
        display_items.append(styled_token)
        if state == (0, 'S'):
            # top of stack
            display_pos.append(' ' + '_' * len(token))
        elif state == (1, 'S'):
            # second of stack
            display_pos.append(' ' + '-' * len(token))
        else:
            display_pos.append(' ' + ' ' * len(token))
    
    os.system('clear')
    print("")
    print("sentence %d\n" % sent_idx)
    print("".join(display_items))
    print("".join(display_pos))
    print("")
    print(amr_state_machine)


def get_action_indexer(symbols):

    action_list_by_prefix = defaultdict(list)
    for action in symbols:
        prefix = action.split('(')[0]
        index = symbols.index(action)
        action_list_by_prefix[prefix].append(index)

    def action_indexer(actions):
        """
        Create a map of each actions prefix to each action in the dictionary
        """

        assert isinstance(actions, list)

        nonlocal symbols
        nonlocal action_list_by_prefix

        idx = set()
        for action in actions:
            if '(' in action:
                # specific action, just inde
                if action not in symbols:
                    # FIXME: Brittle
                    idx.add(symbols.index('<unk>'))
                else:
                    idx.add(symbols.index(action))
            else:    
                # base action, expand 
                idx |= set(action_list_by_prefix[action])
        return idx

    return action_indexer


class StackStateMachine():
    """
    Batch of state machines
    """


    def __init__(self, src_tokens, src_lengths, src_dict, tgt_dict, 
                 max_tgt_len, beam_size, rule_stats, machine_type=None,
                 prec_action_indexer=None, 
                 prec_get_new_state_machine=None):

        # Get all actions indexed by prefix
        if prec_action_indexer is None:
            self.action_indexer = get_action_indexer(tgt_dict.symbols)
        else:    
            self.action_indexer = prec_action_indexer

        # Initialize state machines
        batch_size, max_len = src_tokens.shape
        if rule_stats:
            actions_by_stack_rules = rule_stats['possible_predicates']
        else:
            actions_by_stack_rules = None

        # Returns function that generates initialized state machines given
        if prec_get_new_state_machine is None:
            get_new_state_machine = machine_generator(actions_by_stack_rules)
        else:
            get_new_state_machine = prec_get_new_state_machine

        # store some variables
        self.batch_size = batch_size
        self.tgt_dict = tgt_dict

        # keep count of time step for stopping and action history
        # self.max_tgt_len = max_tgt_len
        self.step_index = 0

        # Watch out, these two variables need to be reorderd in reorder_state!
        self.left_pad = []
        self.machines = []
        for batch_idx in range(batch_size):
            sent_len = src_lengths[batch_idx]
            word_idx = src_tokens[batch_idx, -sent_len:].cpu().numpy()
            tokens = [src_dict[x] for x in word_idx]

            # intialize state machine batch for size 1
            self.machines.append(get_new_state_machine(
                tokens,
                machine_type=machine_type
            ))

            # store left pad size to be used in mask creation
            self.left_pad.append(max_len - sent_len)

        # these have the same info as buffer and stack but in stack-transformer
        # form (batch_size * beam_size, src_len, tgt_len)
        dummy = src_tokens.unsqueeze(2).repeat(1, 1, max_tgt_len)
        self.memory = (torch.ones_like(dummy) * tgt_dict.pad()).float()
        self.memory_pos = (torch.ones_like(dummy) * tgt_dict.pad()).float()
        self.update_masks()

    def get_active_logits(self):

        # Collect active indices for the entire batch
        batch_active_logits = set()
        shape = (len(self.machines), 1, len(self.tgt_dict.symbols))
        logits_mask = torch.zeros(shape, dtype=torch.int16)
        for i in range(len(self.machines)):
            if self.machines[i].is_closed:
                # TODO: Change this to <pad> (will mess with decoder)
                expanded_valid_indices = set([self.tgt_dict.indices['</s>']])
            else:
                valid_actions = self.machines[i].get_valid_actions()
                expanded_valid_indices = self.action_indexer(valid_actions)
            batch_active_logits |= expanded_valid_indices
            logits_mask[i, 0, list(expanded_valid_indices)] = 1
        batch_active_logits = list(batch_active_logits)

        # FIXME: Entropic fix to avoid <unk>. Fix at oracle/fairseq level
        # needed
        batch_active_logits = [
            idx 
            for idx in batch_active_logits
            if idx != self.tgt_dict.indices['<unk>']
        ]

        # store as indices mapping    
        logits_indices = {
            key: idx 
            for idx, key in enumerate(batch_active_logits)
        }
        return logits_indices, logits_mask[:, :, batch_active_logits]

    def update(self, action_batch):

        # sanity check
        batch_size = len(action_batch)
        assert batch_size == len(self.machines)
        #batch_size, num_actions = log_probabilities.shape
        #assert batch_size == len(self.machines)
        # FIXME: Decode adds extra symbols? This seem to be appended but this
        # is a dangerous behaviour
        # if num_actions != len(self.tgt_dict.symbols):
        #    import ipdb; ipdb.set_trace(context=30)
        #    pass

        # execute most probable valid action and return masked probabilities
        # for each sentence in the batch
        for i in range(batch_size):
            self.machines[i].applyAction(action_batch[i])

        # increase action counter
        self.step_index += 1

        # update state expressed as masks
        self.update_masks()

    def update_masks(self, add_padding=0):

        # Get masks from states
        # buffer words    
        device = self.memory.device

        # basis is all padded
        if add_padding:
            raise NotImplementedError()
            # Need to concatenate extra space

        for sent_index, machine in enumerate(self.machines):

            # if machine is closed stop here
            if machine.is_closed:
                continue

            # Get machines buffer and stack compatible with AMR machine 
            # get legacy indexing of buffer and stack from function
            machine_buffer, machine_stack = machine.get_buffer_stack_copy()

            # Reset mask to all non pad elements as being in deleted state 
            pad = self.left_pad[sent_index].item()
            self.memory[sent_index, pad:, self.step_index] = 5
            self.memory_pos[sent_index, pad:, self.step_index] = 0

            # Set buffer elements taking into account padding
            if machine_buffer:

                indices = np.array(machine_buffer) - 1 + pad
                indices[indices == -1 - 1 + pad] = len(machine.tokens) - 1 + pad

                # update masks
                buffer_pos = np.arange(len(machine_buffer))
                positions = len(machine_buffer) - buffer_pos - 1
                self.memory[sent_index, indices, self.step_index] = 3
                self.memory_pos[sent_index, indices, self.step_index] = \
                    torch.tensor(positions, device=self.memory.device).float()

            # Set stack elements taking into account padding
            if machine_stack:

                indices = np.array(machine_stack) - 1 + pad
                # index of root in stack, if there is
                root_in_stack_idx = (indices == -1 - 1 + pad).nonzero()[0]
                indices[root_in_stack_idx] = len(machine.tokens) - 1 + pad

                # update masks
                stack_pos = np.arange(len(machine_stack))
                positions = len(machine_stack) - stack_pos - 1
                self.memory[sent_index, indices, self.step_index] = 4
                # FIXME: This is a BUG in preprocessing by which
                # shifted ROOT is considered deleted
                # update masks
                self.memory[sent_index, indices[root_in_stack_idx], self.step_index] = 5
                self.memory_pos[sent_index, indices, self.step_index] = \
                    torch.tensor(positions, device=self.memory.device).float()

    def reoder_machine(self, reorder_state):
        """Reorder/eliminate machines during decoding"""

        # DEBUG
        # self.encoder_padding_mask = self.encoder_padding_mask[reorder_state, :]
        
        new_machines = []
        new_left_pad = []
        used_indices = set()
        for i in reorder_state.cpu().tolist():
            if i in used_indices:
                # If a machine is duplicated we need to deep copy
                new_machines.append(deepcopy(self.machines[i]))
                new_left_pad.append(self.left_pad[i])
            else:
                new_machines.append(self.machines[i])
                new_left_pad.append(self.left_pad[i])
                used_indices.add(i)
        self.machines = new_machines
        self.left_pad = new_left_pad

        self.memory = self.memory[reorder_state, :, :]
        self.memory_pos = self.memory_pos[reorder_state, :, :]


def machine_generator(actions_by_stack_rules, spacy_lemmatizer=None):
    """Return function that itself returns initialized state machines"""

    # initialize spacy lemmatizer
    if spacy_lemmatizer is None:
        spacy_lemmatizer = get_spacy_lemmatizer()

    def get_new_state_machine(sent_tokens, machine_type=None):

        nonlocal actions_by_stack_rules
        nonlocal spacy_lemmatizer

        # automatic determination of machine if no flag provided
        if sent_tokens[0] in ['<NER>', '<AMR>', 'SRL']: 
            assert machine_type is None, \
                "specify --machine-type OR pre-append <machine-type token>"
            machine_type = sent_tokens[0][1:-1]
        elif machine_type is None:
            Exception(
                "needs either --machine-type or appending <machine-type token>"
            )

        # select machine
        if machine_type == 'AMR':
            return AMRStateMachine(
                sent_tokens,
                actions_by_stack_rules=actions_by_stack_rules,
                spacy_lemmatizer=spacy_lemmatizer
            )
        elif machine_type in ['NER', 'SRL']:
            from bio_tags.machine import BIOStateMachine
            return BIOStateMachine(sent_tokens)
        else:
            raise Exception(f'Unknown machine {machine_type}')

    return get_new_state_machine
