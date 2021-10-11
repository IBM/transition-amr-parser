from collections import defaultdict, Counter
import json
import random
import argparse
import os
from functools import partial
import re
from copy import deepcopy
from itertools import chain

from tqdm import tqdm
import numpy as np
from clbar import yellow_font, clbar
from transition_amr_parser.io import (
    AMR,
    read_amr2,
    read_tokenized_sentences,
    write_tokenized_sentences
)
from ipdb import set_trace


# la_regex = re.compile(r'LA\((.*);(.*)\)')
# ra_regex = re.compile(r'RA\((.*);(.*)\)')
# arc_regex = re.compile(r'[RL]A\((.*);(.*)\)')

# change the format of pointer string from LA(label;pos) -> LA(pos,label)
la_regex = re.compile(r'>LA\((.*),(.*)\)')
ra_regex = re.compile(r'>RA\((.*),(.*)\)')
arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')
la_nopointer_regex = re.compile(r'>LA\((.*)\)')
ra_nopointer_regex = re.compile(r'>RA\((.*)\)')
arc_nopointer_regex = re.compile(r'>[RL]A\((.*)\)')

# for repair mode
goto_regex = re.compile(r'GOTO\((.*)\)')


def red_background(string):
    return "\033[101m%s\033[0m" % string


def graph_alignments(unaligned_nodes, amr):
    """
    Shallow alignment fixer: Inherit the alignment of the last child or first
    parent. If none of these is aligned the node is left unaligned
    """

    fix_alignments = {}
    for (src, _, tgt) in amr.edges:
        if (
            src in unaligned_nodes
            and amr.alignments[tgt] is not None
            and max(amr.alignments[tgt])
                > fix_alignments.get(src, 0)
        ):
            # # debug: to justify to change 0 to -1e6 for a test data corner case; see if there are other cases affected
            # if max(amr.alignments[tgt]) <= fix_alignments.get(src, 0):
            #     breakpoint()
            fix_alignments[src] = max(amr.alignments[tgt])
        elif (
            tgt in unaligned_nodes
            and amr.alignments[src] is not None
            and min(amr.alignments[src])
                < fix_alignments.get(tgt, 1e6)
        ):
            fix_alignments[tgt] = max(amr.alignments[src])

    return fix_alignments


def fix_alignments(gold_amr):

    # Fix unaligned nodes by graph vicinity
    unaligned_nodes = set(gold_amr.nodes) - set(gold_amr.alignments)
    unaligned_nodes |= \
        set(nid for nid, pos in gold_amr.alignments.items() if pos is None)
    unaligned_nodes = list(unaligned_nodes)
    unaligned_nodes_original = list(unaligned_nodes)

    if not unaligned_nodes:
        # no need to do anything
        return gold_amr, []

    if len(unaligned_nodes) == 1 and len(gold_amr.tokens) == 1:
        # Degenerate case: single token
        node_id = list(unaligned_nodes)[0]
        gold_amr.alignments[node_id] = [0]
        return gold_amr, []

    # Align unaligned nodes by using graph vicinnity greedily (1 hop at a time)
    while unaligned_nodes:
        fix_alignments = graph_alignments(unaligned_nodes, gold_amr)
        for nid in unaligned_nodes:
            if nid in fix_alignments:
                gold_amr.alignments[nid] = [fix_alignments[nid]]
                unaligned_nodes.remove(nid)

        # debug: avoid infinite loop for AMR2.0 test data with bad alignments
        if not fix_alignments:
            # breakpoint()
            print(red_background('hard fix on 0th token for fix_alignments'))
            for k, v in list(gold_amr.alignments.items()):
                if v is None:
                    gold_amr.alignments[k] = [0]
            break

    return gold_amr, unaligned_nodes_original


def normalize(token):
    """
    Normalize token or node
    """
    if token == '"':
        return token
    else:
        return token.replace('"', '')


class AMROracle():

    def __init__(self, reduce_nodes=None, absolute_stack_pos=False):

        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to end of self.node_stack
        self.absolute_stack_pos = absolute_stack_pos

        self.dropout = 0.1

    def reset(self, gold_amr):

        # Force align missing nodes and store names for stats
        self.gold_amr, self.unaligned_nodes = fix_alignments(gold_amr)

        self.arcs_for_later = {}

        # will store alignments by token
        # TODO: This should store alignment probabilities
        align_by_token_pos = defaultdict(list)
        for node_id, token_pos in self.gold_amr.alignments.items():
            node = normalize(self.gold_amr.nodes[node_id])
            matched = False
            for pos in token_pos:
                if node == self.gold_amr.tokens[pos]:
                    align_by_token_pos[pos].append(node_id)
                    matched = True
            if not matched:
                align_by_token_pos[token_pos[0]].append(node_id)
        self.align_by_token_pos = align_by_token_pos

        node_id_2_node_number = {}
        for token_pos in sorted(self.align_by_token_pos.keys()):
            for node_id in self.align_by_token_pos[token_pos]:
                node_number = len(node_id_2_node_number)
                node_id_2_node_number[node_id] = node_number

        # will store edges not yet predicted indexed by node
        self.pend_edges_by_node = defaultdict(list)
        for (src, label, tgt) in self.gold_amr.edges:
            self.pend_edges_by_node[src].append((src, label, tgt))
            self.pend_edges_by_node[tgt].append((src, label, tgt))

        #sort edges in descending order of node2pos position
        for node_id in self.pend_edges_by_node :
            edges = []
            for (idx,e) in enumerate(self.pend_edges_by_node[node_id]):
                other_id = e[0]
                if other_id == node_id:
                    other_id = e[2]
                edges.append((node_id_2_node_number[other_id],idx))
            edges.sort(reverse=True)
            new_edges_for_node = []
            for (_,idx) in edges:
                new_edges_for_node.append(self.pend_edges_by_node[node_id][idx])
            self.pend_edges_by_node[node_id] = new_edges_for_node

        # Will store gold_amr.nodes.keys() and edges as we predict them
        self.node_map = {}
        self.node_reverse_map = {}
        self.predicted_edges = []

    def get_arc_action(self, machine):

        # Loop over edges not yet created
        top_node_id = machine.node_stack[-1]
        current_id = self.node_reverse_map[top_node_id]
        for (src, label, tgt) in list(self.pend_edges_by_node[current_id]):
            # skip if it involves nodes not yet created
            if src not in self.node_map or tgt not in self.node_map:
                continue

            arc_action = None

            if (
                self.node_map[src] == top_node_id
                and self.node_map[tgt] in machine.node_stack[:-1]
            ):
                # LA <--
                if self.absolute_stack_pos:
                    # node position is just position in action history
                    index = self.node_map[tgt]
                else:
                    # stack position 0 is closest to current node
                    index = machine.node_stack.index(self.node_map[tgt])
                    index = len(machine.node_stack) - index - 2
                # Remove this edge from for both involved nodes
                self.pend_edges_by_node[tgt].remove((src, label, tgt))
                self.pend_edges_by_node[current_id].remove((src, label, tgt))
                # return [f'LA({label[1:]};{index})'], [1.0]
                assert label[0] == ':'    # NOTE include the relation marker ':' in action names
                arc_action = [f'>LA({index},{label})'], [1.0]

            elif (
                self.node_map[tgt] == top_node_id
                and self.node_map[src] in machine.node_stack[:-1]
            ):
                # RA -->
                # note stack position 0 is closest to current node
                if self.absolute_stack_pos:
                    # node position is just position in action history
                    index = self.node_map[src]
                else:
                    # Relative node position
                    index = machine.node_stack.index(self.node_map[src])
                    index = len(machine.node_stack) - index - 2
                # Remove this edge from for both involved nodes
                self.pend_edges_by_node[src].remove((src, label, tgt))
                self.pend_edges_by_node[current_id].remove((src, label, tgt))
                # return [f'RA({label[1:]};{index})'], [1.0]
                assert label[0] == ':'    # NOTE include the relation marker ':' in action names
                arc_action = [f'>RA({index},{label})'], [1.0]

            #with dropout prob, save for later
            if random.random() < self.dropout:
                node_idx = machine.node_stack[-1]
                if node_idx not in self.arcs_for_later:
                    self.arcs_for_later[node_idx] = []
                self.arcs_for_later[node_idx].append( arc_action )
                continue
            else:
                return arc_action

    def get_reduce_action(self, machine, top=True):
        """
        If last action is an arc, check if any involved node (top or not top)
        has no pending edges
        """
        if machine.action_history == []:
            return False
        action = machine.action_history[-1]
        fetch = arc_regex.match(action)
        if fetch is None:
            return False
        if top:
            node_id = machine.node_stack[-1]
        else:
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                node_id = index
            else:
                # Relative node position
                index = len(machine.node_stack) - index - 2
                node_id = machine.node_stack[index]
        gold_node_id = self.node_reverse_map[node_id]
        return self.pend_edges_by_node[gold_node_id] == []

    def get_actions(self, machine):

        if machine.in_repair_mode:
            #repair mode
            for idx in sorted(self.arcs_for_later.keys()):
                goto_action = f'GOTO({idx})'
                if goto_action not in machine.action_history:
                    return [goto_action], [1.0]
                arcs = self.arcs_for_later[idx]
                num_arcs = len(arcs)
                for ai in range(num_arcs):
                    arc_action = arcs[ai]
                    del self.arcs_for_later[idx][ai]
                    return arc_action

            self.arcs_for_later = {}
            return ['DONE'], [1.0]


        # Label node as root
        if (
            machine.node_stack
            and machine.root is None
            and self.node_reverse_map[machine.node_stack[-1]] ==
                self.gold_amr.root
        ):
            return ['ROOT'], [1.0]

        # REDUCE in stack after are LA/RA that completes all edges for an node
        if self.reduce_nodes == 'all':
            arc_reduce_no_top = self.get_reduce_action(machine, top=False)
            arc_reduce_top = self.get_reduce_action(machine, top=True)
            if arc_reduce_no_top and arc_reduce_top:
                # both nodes invoved
                return ['REDUCE3'], [1.0]
            elif arc_reduce_top:
                # top of the stack node
                return ['REDUCE'], [1.0]
            elif arc_reduce_no_top:
                # the other node
                return ['REDUCE2'], [1.0]

        # Return action creating next pending edge last node in stack
        if len(machine.node_stack) > 1:
            arc_action = self.get_arc_action(machine)
            if arc_action:
                return arc_action

        # Return action creating next node aligned to current cursor
        for nid in self.align_by_token_pos[machine.tok_cursor]:
            if nid in self.node_map:
                continue

            # NOTE: For PRED action we also include the gold id for
            # tracking and scoring of graph
            target_node = normalize(self.gold_amr.nodes[nid])

            if normalize(machine.tokens[machine.tok_cursor]) == target_node:
                # COPY
                return [('COPY', nid)], [1.0]
            else:
                # Generate
                return [(target_node, nid)], [1.0]

        # Move monotonic attention
        if machine.tok_cursor < len(machine.tokens)-1 or (machine.tok_cursor == len(machine.tokens)-1 and len(self.arcs_for_later) == 0):
            return ['SHIFT'], [1.0]

        if len(self.arcs_for_later) > 0:
            return ['REPAIR'], [1.0]
        else:
            return ['CLOSE'], [1.0]

class AMRStateMachine():

    def __init__(self, reduce_nodes=None, absolute_stack_pos=False):

        # Here non state variables (do not change across sentences) as well as
        # slow initializations
        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to stack top
        self.absolute_stack_pos = absolute_stack_pos

        # base actions allowed
        self.base_action_vocabulary = [
            'SHIFT',   # Move cursor
            'COPY',    # Copy word under cursor to node (add node to stack)
            'ROOT',    # Label node as root
            '>LA',      # Arc from node under cursor (<label>, <to position>) (to be different from LA the city)
            '>RA',      # Arc to node under cursor (<label>, <from position>)
            'REPAIR',   # start repair mode
            'GOTO',     # go to a node in repair mode (<to index>)
            'DONE',   # done with repair mode
            'CLOSE',   # Close machine
            # ...      # create node with ... as name (add node to stack)
            'NODE'     # other node names
        ]
        if self.reduce_nodes:
            self.base_action_vocabulary.append([
                'REDUCE',   # Remove node at top of the stack
                'REDUCE2',  # Remove node at las LA/RA pointed position
                'REDUCE3'   # Do both above
            ])

    def canonical_action_to_dict(self, vocab):
        """Map the canonical actions to ids in a vocabulary, each canonical action corresponds to a set of ids.

        CLOSE is mapped to eos </s> token.
        """
        canonical_act_ids = dict()
        vocab_act_count = 0
        assert vocab.eos_word == '</s>'
        for i in range(len(vocab)):
            # NOTE can not directly use "for act in vocab" -> this will never stop since no stopping iter implemented
            act = vocab[i]
            if act in ['<s>', '<pad>', '<unk>', '<mask>'] or act.startswith('madeupword'):
                continue
            cano_act = self.get_base_action(act) if i != vocab.eos() else 'CLOSE'
            if cano_act in self.base_action_vocabulary:
                vocab_act_count += 1
                canonical_act_ids.setdefault(cano_act, []).append(i)
        # print for debugging
        # print(f'{vocab_act_count} / {len(vocab)} tokens in action vocabulary mapped to canonical actions.')
        return canonical_act_ids

    # def canonical_action_to_dict_bpe(self, vocab):
    #     """Map the canonical actions to ids in a vocabulary, each canonical action corresponds to a set of ids.
    #     Here the mapping is specially dealing with shared bpe vocabulary, with possible node splits.

    #     CLOSE is mapped to eos </s> token.
    #     """
    #     canonical_act_ids = dict()
    #     vocab_act_count = 0
    #     assert vocab.eos_word == '</s>'
    #     for i in range(len(vocab)):
    #         # NOTE can not directly use "for act in vocab" -> this will never stop since no stopping iter implemented
    #         act = vocab[i]
    #         if act in ['<s>', '<pad>', '<unk>', '<mask>'] or act.startswith('madeupword'):
    #             continue
    #         # NOTE remove the start of token space in bpe symbols
    #         if act[0].startswith(vocab.bpe.INIT):
    #             act = act[1:]
    #         # NOTE the subtokens are currently included in the class of "NODE" base action, which
    #         # is allowed at every step except after the last SHIFT

    #         # NOTE in the bart bpe vocabulary both "CLOSE" and "ĠCLOSE" exist -> they should be mapped to eos
    #         if act == 'CLOSE':
    #             # skip these conflicting CLOSE tokens
    #             continue

    #         cano_act = self.get_base_action(act) if i != vocab.eos() else 'CLOSE'
    #         if cano_act in self.base_action_vocabulary:
    #             vocab_act_count += 1
    #             canonical_act_ids.setdefault(cano_act, []).append(i)
    #     # print for debugging
    #     # print(f'{vocab_act_count} / {len(vocab)} tokens in action vocabulary mapped to canonical actions.')
    #     return canonical_act_ids

    def reset(self, tokens):
        '''
        Reset state variables and set a new sentence
        '''
        # state
        self.tokens = list(tokens)
        self.tok_cursor = 0
        self.node_stack = []
        self.action_history = []
        # AMR as we construct it
        # NOTE: We will use position of node generating action in action
        # history as node_id
        self.nodes = {}
        self.edges = []
        self.root = None
        self.alignments = defaultdict(list)
        # set to true when machine finishes
        self.is_closed = False
        self.in_repair_mode = False
        # state info useful in the model
        self.actions_tokcursor = []

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as fid:
            config = json.loads(fid.read())
        return cls(**config)

    def save(self, config_path):
        with open(config_path, 'w') as fid:
            # NOTE: Add here all *non state* variables in __init__()
            fid.write(json.dumps(dict(
                reduce_nodes=self.reduce_nodes,
                absolute_stack_pos=self.absolute_stack_pos
            )))

    def __deepcopy__(self, memo):
        """
        Manual deep copy of the machine

        avoid deep copying spacy lemmatizer
        """
        cls = self.__class__
        result = cls.__new__(cls)
        # DEBUG: usew this to detect very heavy constants that can be referred
        # import time
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # start = time.time()
            # if k in ['spacy_lemmatizer', 'actions_by_stack_rules']:
            #     setattr(result, k, v)
            # else:
            #     setattr(result, k, deepcopy(v, memo))
            setattr(result, k, deepcopy(v, memo))
            # print(k, time.time() - start)
        # import ipdb; ipdb.set_trace(context=30)
        return result

    def get_current_token(self):
        return self.tokens[self.tok_cursor]

    def get_base_action(self, action):
        """Get the base action form, by stripping the labels, etc."""
        if action in self.base_action_vocabulary:
            return action
        # remaining ones are ['>LA', '>RA', 'NODE']
        # NOTE need to deal with both '>LA(pos,label)' and '>LA(label)', as in the vocabulary the pointers are peeled off
        if arc_regex.match(action) or arc_nopointer_regex.match(action):
            return action[:3]

        if goto_regex.match(action):
            return 'GOTO'

        return 'NODE'

    def get_valid_actions(self, max_1root=True):

        valid_base_actions = []

        if self.tok_cursor < len(self.tokens) and \
           not self.in_repair_mode: #we might want to change this to allow new nodes during repair
            valid_base_actions.append('SHIFT')
            valid_base_actions.extend(['COPY', 'NODE'])

        if self.action_history and \
                self.get_base_action(self.action_history[-1]) in ['COPY', 'NODE', 'ROOT', '>LA', '>RA', 'GOTO']:
            valid_base_actions.extend(['>LA', '>RA'])

        if self.action_history and \
                self.get_base_action(self.action_history[-1]) in ['COPY', 'NODE']:
            if max_1root:
                # force to have at most 1 root (but it can always be with no root)
                if not self.root:
                    valid_base_actions.append('ROOT')
            else:
                valid_base_actions.append('ROOT')

        if self.tok_cursor == len(self.tokens):
            assert not valid_base_actions and self.action_history[-1] == 'SHIFT'
            valid_base_actions.append('CLOSE')

        if self.tok_cursor == len(self.tokens)-1:
            valid_base_actions.append('REPAIR')

        if self.in_repair_mode and self.get_base_action(self.action_history[-1]) != 'GOTO':
            valid_base_actions.append('GOTO')
            valid_base_actions.append('DONE')

        if self.reduce_nodes:
            raise NotImplementedError

            if len(self.node_stack) > 0:
                valid_base_actions.append('REDUCE')
            if len(self.node_stack) > 1:
                valid_base_actions.append('REDUCE2')
                valid_base_actions.append('REDUCE3')

        return valid_base_actions

    def get_actions_nodemask(self):
        """Get the binary mask of node actions"""
        actions_nodemask = [0] * len(self.action_history)
        for i in self.node_stack:
            actions_nodemask[i] = 1
        return actions_nodemask

    def update(self, action):

        #print(action)

        assert not self.is_closed

        self.actions_tokcursor.append(self.tok_cursor)

        if re.match(r'CLOSE', action):
            self.is_closed = True

        elif re.match(r'REPAIR', action):
            self.in_repair_mode = True

        elif re.match(r'DONE', action):
            self.in_repair_mode = False
            self.tok_cursor = len(self.tokens) - 1

        elif re.match(r'ROOT', action):
            self.root = self.node_stack[-1]

        elif action in ['SHIFT']:
            # Move source pointer
            self.tok_cursor += 1

        elif goto_regex.match(action):
            # goto a previously generted node index
            node_index = int(goto_regex.match(action).groups()[0])
            self.node_stack.append(node_index)
            # not sure if source cursor should also move or not
            self.tok_cursor = self.actions_tokcursor[node_index]

        elif action in ['REDUCE']:
            # eliminate top of the stack
            assert self.reduce_nodes
            assert self.action_history[-1]
            self.node_stack.pop()

        elif action in ['REDUCE2']:
            # eliminate the other node involved in last arc not on top
            assert self.reduce_nodes
            assert self.action_history[-1]
            fetch = arc_regex.match(self.action_history[-1])
            assert fetch
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                # Absolute position and also node_id
                self.node_stack.remove(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                self.node_stack.pop(index)

        elif action in ['REDUCE3']:
            # eliminate both nodes involved in arc
            assert self.reduce_nodes
            assert self.action_history[-1]
            fetch = arc_regex.match(self.action_history[-1])
            assert fetch
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                # Absolute position and also node_id
                self.node_stack.remove(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                self.node_stack.pop(index)
            self.node_stack.pop()

        # Edge generation
        elif la_regex.match(action):
            # Left Arc <--
            # label, index = la_regex.match(action).groups()
            index, label = la_regex.match(action).groups()
            if self.absolute_stack_pos:
                tgt = int(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                tgt = self.node_stack[index]
            src = self.node_stack[-1]
            self.edges.append((src, f'{label}', tgt))
            #if goto_regex.match(self.action_history[-1]):
            #    self.node_stack.pop()

        elif ra_regex.match(action):
            # Right Arc -->
            # label, index = ra_regex.match(action).groups()
            index, label = ra_regex.match(action).groups()
            if self.absolute_stack_pos:
                src = int(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                src = self.node_stack[index]
            tgt = self.node_stack[-1]
            self.edges.append((src, f'{label}', tgt))

        # Node generation
        elif action == 'COPY':
            # copy surface symbol under cursor to node-name
            node_id = len(self.action_history)
            self.nodes[node_id] = normalize(self.tokens[self.tok_cursor])
            self.node_stack.append(node_id)
            self.alignments[node_id].append(self.tok_cursor)

        else:

            # Interpret action as a node name
            # Note that the node_id is the position of the action that
            # generated it
            node_id = len(self.action_history)
            self.nodes[node_id] = action
            self.node_stack.append(node_id)
            self.alignments[node_id].append(self.tok_cursor)

        # Action for each time-step
        self.action_history.append(action)

    def get_annotation(self):
        amr = AMR(self.tokens, self.nodes, self.edges, self.root,
                  alignments=self.alignments)
        return amr.__str__()


def get_ngram(sequence, order):
    ngrams = []
    for n in range(len(sequence) - order + 1):
        ngrams.append(tuple(sequence[n:n+order]))
    return ngrams


class Stats():

    def __init__(self, ignore_indices, ngram_stats=False, breakpoint=False):
        self.index = 0
        self.ignore_indices = ignore_indices
        # arc generation stats
        self.stack_size_count = Counter()
        self.pointer_positions_count = Counter()
        # alignment stats
        self.unaligned_node_count = Counter()
        self.node_count = 0
        # token/action stats
        self.tokens = []
        self.action_sequences = []
        self.action_count = Counter()

        self.ngram_stats = ngram_stats
        self.breakpoint = breakpoint

        # Stats for action n-grams
        if self.ngram_stats:
            self.bigram_count = Counter()
            self.trigram_count = Counter()
            self.fourgram_count = Counter()

    def update_machine_stats(self, machine):

        if self.breakpoint:
            os.system('clear')
            print(" ".join(machine.tokens))
            print(" ".join(machine.action_history))
            print(" ".join([machine.action_history[i]
                            for i in machine.node_stack]))
            set_trace()

            # if len(machine.node_stack) > 8 and stats.index not in [12]:
            #    set_trace(context=30)

        action = machine.action_history[-1]
        fetch = arc_regex.match(action)
        if fetch:
            # stack_pos = int(fetch.groups()[1])
            stack_pos = int(fetch.groups()[0])
            self.stack_size_count.update([len(machine.node_stack)])
            self.pointer_positions_count.update([stack_pos])

    def update_sentence_stats(self, oracle, machine):

        # Note that we do not ignore this one either way
        self.tokens.append(machine.tokens)
        self.action_sequences.append(machine.action_history)
        base_actions = [x.split('(')[0] for x in machine.action_history]
        self.action_count.update(base_actions)

        # alignment fixing stats
        unodes = [oracle.gold_amr.nodes[n] for n in oracle.unaligned_nodes]
        self.unaligned_node_count.update(unodes)
        self.node_count += len(oracle.gold_amr.nodes)

        if self.index in self.ignore_indices:
            self.index += 1
            return

        if self.ngram_stats:
            actions = machine.action_history
            self.bigram_count.update(get_ngram(actions, 2))
            self.trigram_count.update(get_ngram(actions, 3))
            self.fourgram_count.update(get_ngram(actions, 4))

        # breakpoint if AMR does not match
        self.stop_if_error(oracle, machine)

        # update counter
        self.index += 1

    def stop_if_error(self, oracle, machine):

        # Check node name match
        for nid, node_name in oracle.gold_amr.nodes.items():
            node_name_machine = machine.nodes[oracle.node_map[nid]]
            if normalize(node_name_machine) != normalize(node_name):
                set_trace(context=30)
                print()

        # Check mapped edges match
        mapped_gold_edges = []
        for (s, label, t) in oracle.gold_amr.edges:
            if s not in oracle.node_map or t not in oracle.node_map:
                set_trace(context=30)
                continue
            mapped_gold_edges.append(
                (oracle.node_map[s], label, oracle.node_map[t])
            )
        if sorted(machine.edges) != sorted(mapped_gold_edges):
            set_trace(context=30)
            print()

        # Check root matches
        mapped_root = oracle.node_map[oracle.gold_amr.root]
        if machine.root != mapped_root:
            set_trace(context=30)
            print()

    def display(self):

        num_processed = self.index - len(self.ignore_indices)
        perc = num_processed * 100. / self.index
        print(
            f'{num_processed}/{self.index} ({perc:.1f} %)'
            f' exact match of AMR graphs (non printed)'
        )
        print(yellow_font(
            f'{len(self.ignore_indices)} sentences ignored for stats'
        ))

        num_copy = self.action_count['COPY']
        perc = num_copy * 100. / self.node_count
        print(
            f'{num_copy}/{self.node_count} ({perc:.1f} %) of nodes generated'
            ' by COPY'
        )

        if self.unaligned_node_count:
            num_unaligned = sum(self.unaligned_node_count.values())
            print(yellow_font(
                f'{num_unaligned}/{self.node_count} unaligned nodes aligned'
                ' by graph vicinity'
            ))

        # Other stats
        return

        # format viewer
        clbar2 = partial(clbar, ylim=(0, None), norm=True,
                         yform=lambda y: f'{100*y:.1f}', ncol=80)

        print('Stack size')
        clbar2(xy=self.stack_size_count, botx=20)

        print('Positions')
        clbar2(xy=self.pointer_positions_count, botx=20)

        if self.ngram_stats:
            print('tri-grams')
            clbar(xy=self.trigram_count, topy=20)
            set_trace()
            print()


def peel_pointer(action, pad=-1):
    """Peel off the pointer value from arc actions"""
    if arc_regex.match(action):
        # LA(pos,label) or RA(pos,label)
        action, properties = action.split('(')
        properties = properties[:-1]    # remove the ')' at last position
        properties = properties.split(',')    # split to pointer value and label
        pos = int(properties[0].strip())
        label = properties[1].strip()    # remove any leading and trailing white spaces
        action_label = action + '(' + label + ')'
        return (action_label, pos)
    else:
        return (action, pad)


class StatsForVocab:
    """Collate stats for predicate node names with their frequency, and list of all the other action symbols.
    For arc actions, pointers values are stripped.
    The results stats (from training data) are going to decide which node names (the frequent ones) to be added to
    the vocabulary used in the model.
    """
    def __init__(self, no_close=False):
        # DO NOT include CLOSE action (as this is internally managed by the eos token in model)
        # NOTE we still add CLOSE into vocabulary, just to be complete although it is not used
        self.no_close = no_close

        self.nodes = Counter()
        self.left_arcs = Counter()
        self.right_arcs = Counter()
        self.control = Counter()

    def update(self, action, machine):
        if self.no_close:
            if action in ['CLOSE', '_CLOSE_']:
                return

        if la_regex.match(action) or la_nopointer_regex.match(action):
            # LA(pos,label) or LA(label)
            action, pos = peel_pointer(action)
            # NOTE should be an iterable instead of a string; otherwise it'll be character based
            self.left_arcs.update([action])
        elif ra_regex.match(action) or ra_nopointer_regex.match(action):
            # RA(pos,label) or RA(label)
            action, pos = peel_pointer(action)
            self.right_arcs.update([action])
        elif machine.get_base_action(action) == 'GOTO':
            self.control.update(['GOTO'])
        elif action in machine.base_action_vocabulary:
            self.control.update([action])
        else:
            # node names
            self.nodes.update([action])

    def display(self):
        print('Total number of different node names:')
        print(len(list(self.nodes.keys())))
        print('Most frequent node names:')
        print(self.nodes.most_common(20))
        print('Most frequent left arc actions:')
        print(self.left_arcs.most_common(20))
        print('Most frequent right arc actions:')
        print(self.right_arcs.most_common(20))
        print('Other control actions:')
        print(self.control)

    def write(self, path_prefix):
        """Write the stats into file. Two files will be written: one for nodes, one for others."""
        path_nodes = path_prefix + '.nodes'
        path_others = path_prefix + '.others'
        with open(path_nodes, 'w') as f:
            for k, v in self.nodes.most_common():
                print(f'{k}\t{v}', file=f)
        with open(path_others, 'w') as f:
            for k, v in chain(self.control.most_common(), self.left_arcs.most_common(), self.right_arcs.most_common()):
                print(f'{k}\t{v}', file=f)


def oracle(args):

    # Read AMR
    amrs = read_amr2(args.in_aligned_amr, ibm_format=True)

    # NOTE fix the unicode issue
    # breakpoint()
    unicode_fixes = True
    if unicode_fixes:

        # Replacement rules for unicode chartacters
        replacement_rules = {
            'ˈtʃærɪti': 'charity',
            '\x96': '_',
            '⊙': 'O'
        }

        # FIXME: normalization shold be more robust. Right now use the tokens
        # of the amr inside the oracle. This is why we need to normalize them.
        for idx, amr in enumerate(amrs):
            new_tokens = []
            for token in amr.tokens:
                forbidden = [x for x in replacement_rules.keys() if x in token]
                if forbidden:
                    token = token.replace(
                        forbidden[0],
                        replacement_rules[forbidden[0]]
                    )
                new_tokens.append(token)
            amr.tokens = new_tokens

    # broken annotations that we ignore in stats
    # 'DATA/AMR2.0/aligned/cofill/train.txt'
    ignore_indices = [
        8372,   # (49, ':time', 49), (49, ':condition', 49)
        17055,  # (3, ':mod', 7), (3, ':mod', 7)
        27076,  # '0.0.2.1.0.0' is on ::edges but not ::nodes
    ]

    # Initialize machine
    machine = AMRStateMachine(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions
    )
    # Save machine config
    machine.save(args.out_machine_config)

    # initialize oracle
    oracle = AMROracle(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions
    )

    # will store statistics and check AMR is recovered
    stats = Stats(ignore_indices, ngram_stats=False)
    stats_vocab = StatsForVocab(no_close=False)
    for idx, amr in tqdm(enumerate(amrs), desc='Oracle'):

        # debug
        # print(idx)    # 96 for AMR2.0 test data infinit loop
        # if idx == 96:
        #     breakpoint()

        # spawn new machine for this sentence
        machine.reset(amr.tokens)

        # initialize new oracle for this AMR
        oracle.reset(amr)

        random.seed(0)

        # proceed left to right throught the sentence generating nodes
        while not machine.is_closed:

            # get valid actions
            _ = machine.get_valid_actions()

            # oracle
            actions, scores = oracle.get_actions(machine)
            # actions = [a for a in actions if a in valid_actions]
            # most probable
            action = actions[np.argmax(scores)]

            # if it is node generation, keep track of original id in gold amr
            if isinstance(action, tuple):
                action, gold_node_id = action
                node_id = len(machine.action_history)
                oracle.node_map[gold_node_id] = node_id
                oracle.node_reverse_map[node_id] = gold_node_id

            # update machine,
            machine.update(action)

            # update machine stats
            stats.update_machine_stats(machine)

            # update vocabulary
            stats_vocab.update(action, machine)

        # Sanity check: We recovered the full AMR
        stats.update_sentence_stats(oracle, machine)

        # do not write 'CLOSE' in the action sequences
        # this might change the machine.action_history in place, but it is the end of this machine already
        close_action = stats.action_sequences[-1].pop()
        assert close_action == 'CLOSE'

    # display statistics
    stats.display()

    # save action sequences and tokens
    write_tokenized_sentences(
        args.out_actions,
        stats.action_sequences,
        '\t'
    )
    write_tokenized_sentences(
        args.out_tokens,
        stats.tokens,
        '\t'
    )

    # save action vocabulary stats
    # debug

    stats_vocab.display()
    if getattr(args, 'out_stats_vocab', None) is not None:
        stats_vocab.write(args.out_stats_vocab)
        print(f'Action vocabulary stats written in {args.out_stats_vocab}.*')


def play(args):

    sentences = read_tokenized_sentences(args.in_tokens, '\t')
    action_sequences = read_tokenized_sentences(args.in_actions, '\t')
    assert len(sentences) == len(action_sequences)

    # This will store the annotations to write
    annotations = []

    # Initialize machine
    machine = AMRStateMachine.from_config(args.in_machine_config)
    for index in tqdm(range(len(action_sequences)), desc='Machine'):

        # New machine for this sentence
        machine.reset(sentences[index])

        # add back the 'CLOSE' action if it is not written in file
        if action_sequences[index][-1] != 'CLOSE':
            action_sequences[index].append('CLOSE')

        for action in action_sequences[index]:
            machine.update(action)

        assert machine.is_closed

        # print AMR
        annotations.append(machine.get_annotation())

    with open(args.out_amr, 'w') as fid:
        for annotation in annotations:
            fid.write(annotation)


def main(args):

    # TODO: Use two separate entry points with own argparse and requires

    if args.in_actions:
        # play actions and return AMR
        assert args.in_machine_config
        assert args.in_tokens
        assert args.in_actions
        assert args.out_amr
        assert not args.out_actions
        assert not args.in_aligned_amr
        play(args)

    elif args.in_aligned_amr:
        # Run oracle and determine actions from AMR
        assert args.in_aligned_amr
        assert args.out_actions
        assert args.out_tokens
        assert args.out_machine_config
        assert not args.in_tokens
        assert not args.in_actions
        oracle(args)


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    # Single input parameters
    parser.add_argument(
        "--in-aligned-amr",
        help="In file containing AMR in penman format AND IBM graph notation "
             "(::node, etc). Graph read from the latter and not penman",
        type=str
    )
    # ORACLE
    parser.add_argument(
        "--reduce-nodes",
        choices=['all'],
        help="Rules to delete completed nodes during parsing"
             "all: delete every complete node",
        type=str,
    )
    parser.add_argument(
        "--absolute-stack-positions",
        help="e.g. LA(<label>, <pos>) <pos> is absolute position in sentence",
        action='store_true'
    )
    parser.add_argument(
        "--out-actions",
        help="tab separated actions, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-tokens",
        help="space separated tokens, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-machine-config",
        help="configuration for state machine in config format",
        type=str,
    )
    parser.add_argument(
        "--out-stats-vocab",
        type=str,
        help="action vocabulary frequencies"
    )
    # MACHINE
    parser.add_argument(
        "--in-tokens",
        help="space separated tokens, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--in-actions",
        help="tab separated actions, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-amr",
        help="In file containing AMR in penman format",
        type=str,
    )
    parser.add_argument(
        "--in-machine-config",
        help="configuration for state machine in config format",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
