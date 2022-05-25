import json
import argparse
import os
from functools import partial
import re
from copy import deepcopy
from itertools import chain
from collections import defaultdict, Counter
import random
from tqdm import tqdm
import numpy as np
from transition_amr_parser.io import (
    read_blocks,
    read_tokenized_sentences,
    write_tokenized_sentences,
    read_neural_alignments
)
from transition_amr_parser.docamr_io import AMR_doc,process_corefs,read_amr, read_amr_penman
# from transition_amr_parser.amr_aligner import get_ner_ids
from transition_amr_parser.gold_subgraph_align import (
    AlignModeTracker, check_gold_alignment
)
from transition_amr_parser.amr import normalize, create_valid_amr
# TODO: Remove this dependency
import penman
from transition_amr_parser.clbar import (
    yellow_font, green_font, red_background, clbar
)
from ipdb import set_trace
from operator import itemgetter

# change the format of pointer string from LA(label;pos) -> LA(pos,label)
la_regex = re.compile(r'>LA\((.*),(.*)\)')
ra_regex = re.compile(r'>RA\((.*),(.*)\)')
arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')
la_nopointer_regex = re.compile(r'>LA\((.*)\)')
ra_nopointer_regex = re.compile(r'>RA\((.*)\)')
arc_nopointer_regex = re.compile(r'>[RL]A\((.*)\)')


def graph_alignments(unaligned_nodes, amr):
    """
    Shallow alignment fixer: Inherit the alignment of the FIRST child or first
    parent. If none of these is aligned the node is left unaligned
    """

    # scan first for all children-based alignments
    fix_alignments = {}
    for (src, _, tgt) in amr.edges:
        if (
            src in unaligned_nodes
            and amr.alignments.get(tgt, None) is not None
            and max(amr.alignments[tgt])
                > fix_alignments.get(src, -1)
        ):
            # # debug: to justify to change 0 to -1e6 for a test data corner
            # case; see if there are other cases affected
            # if max(amr.alignments[tgt]) <= fix_alignments.get(src, 0):
            #     breakpoint()
            fix_alignments[src] = max(amr.alignments[tgt])

    # exit if any fix (to try again recursively to fix another one)
    if len(fix_alignments):
        return fix_alignments

    # then parent-based ones
    for (src, _, tgt) in amr.edges:
        if (
            tgt in unaligned_nodes
            and amr.alignments.get(src, None) is not None
            and min(amr.alignments[src])
                < fix_alignments.get(tgt, 1e6)
        ):
            fix_alignments[tgt] = max(amr.alignments[src])

    return fix_alignments


def graph_vicinity_align(gold_amr):
    '''
    Fix unaligned nodes by graph vicinity
    '''

    # easy fix, where no alignments is admissible
    if (
        gold_amr.alignments is None
        and len(gold_amr.nodes) == 1
        and len(gold_amr.tokens) == 1
    ):
        single_nid = list(gold_amr.nodes.keys())[0]
        gold_amr.alignments = {single_nid: [0]}
        return gold_amr, set([single_nid])

    elif gold_amr.alignments is None:
        raise Exception("Expected alignments in AMR")

    unaligned_nodes = set(gold_amr.nodes) - set(gold_amr.alignments)
    unaligned_nodes |= \
        set(nid for nid, pos in gold_amr.alignments.items() if pos is None)
    unaligned_nodes = sorted(list(unaligned_nodes))
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
    count = 0
    while unaligned_nodes:
        fix_alignments = graph_alignments(unaligned_nodes, gold_amr)
        for nid in unaligned_nodes:
            if nid in fix_alignments:
                gold_amr.alignments[nid] = [fix_alignments[nid]]
                unaligned_nodes.remove(nid)

        # debug: avoid infinite loop for AMR2.0 test data with bad alignments
        count += 1
        if count == 1000:
            msg = 'hard fix on 0th token for fix_alignments'
            print(f'\n{red_background(msg)}\n')
            for nid in gold_amr.nodes:
                if (
                    nid not in gold_amr.alignments
                    or gold_amr.alignments[nid] == []
                ):
                    gold_amr.alignments[nid] = [0]
            break

    return gold_amr, unaligned_nodes_original


def sample_alignments(gold_amr, alignment_probs, temperature=1.0):
    # this contains p(node, token_pos | tokens)
    node_token_joint = alignment_probs['p_node_and_token']
    # summing we can get p(node | tokens)
    node_marginal = node_token_joint.sum(1, keepdims=True)
    # p(token_pos | nodes, tokens)
    #    = p(node, token_pos | tokens) / p(node | tokens)
    token_posterior = node_token_joint / node_marginal

    # sharpen / flatten by temperature
    if temperature > 0:
        phi = token_posterior**(1. / temperature)
        token_posterior2 = phi / phi.sum(1, keepdims=True)
    else:
        num_tokens, num_nodes = token_posterior.shape
        token_posterior2 = np.zeros((num_tokens, num_nodes))
        token_posterior2[np.arange(num_tokens), token_posterior.argmax(1)] = 1
        # FIXME: this sanity check shows node order is dict order and not that
        # on node_short_id
        # for counts aligner fails in two (27503)
#         assert [y for x in gold_amr.alignments.values() for y in x] \
#             == list(token_posterior2.argmax(1)), \
#             "--in-aligned-amr and --in-aligned-probs have different argmax"

    if gold_amr.alignments is None:
        gold_amr.alignments = {}

    # FIXME: See above
    # for idx, node_id in enumerate(alignment_probs['node_short_id']):
    align_info = dict(node_idx=[], token_idx=[], p=[])

    # This is because numpy casts as float64 anyway. See:
    # https://github.com/numpy/numpy/issues/8317
    token_posterior2 = token_posterior2.astype(np.float64) \
        / token_posterior2.astype(np.float64).sum(-1, keepdims=True)
    for idx, node_id in enumerate(gold_amr.alignments.keys()):
        alignment = np.random.multinomial(1, token_posterior2[idx, :]).argmax()
        gold_amr.alignments[node_id] = [alignment]

        align_info['node_idx'].append(idx)
        align_info['token_idx'].append(alignment)
        align_info['p'].append(token_posterior2[idx, alignment])

    assert set(gold_amr.alignments.keys()) <= set(gold_amr.nodes.keys()), \
        'node ids from graph and alignment probabilities do not match' \
        ', maybe read JAMR format when alignments are PENMAN?'

    assert (
        set([y for x in gold_amr.alignments.values() for y in x])
        <= set(range(len(gold_amr.tokens)))
    ), 'Alignment token positions out of bounds with respect to given tokens'

    return gold_amr, align_info


class AMROracle():

    def __init__(
        self,
        reduce_nodes=None,                   # garbage collection mode
        absolute_stack_pos=False,            # index nodes relative or absolute
        use_copy=True,                       # copy mechanism toggle
        alignment_sampling_temperature=1.0,  # temperature if sampling
        force_align_ner=False                # align NER parents to first child
    ):

        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to end of self.node_stack
        self.absolute_stack_pos = absolute_stack_pos
        self.alignment_sampling_temperature = alignment_sampling_temperature
        self.use_copy = use_copy
        self.force_align_ner = force_align_ner

    def reset(self, gold_amr, alignment_probs=None,
              alignment_sampling_temp=1.0):

        # if probabilties provided sample alignments from them
        self.align_info = None
        if alignment_probs:
            gold_amr, align_info = sample_alignments(
                gold_amr, alignment_probs, alignment_sampling_temp)
            self.align_info = align_info

        # Force align unaligned nodes and store names for stats
        self.gold_amr, self.unaligned_nodes = graph_vicinity_align(gold_amr)

        if self.force_align_ner:
            raise NotImplementedError()
            # align parents in NERs to first child

        # will store node-id by token they align to
        # TODO: no order of nodes enforced (?), see TODO below
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

        # get gold node-id to decoded node-id map ()
        # TODO: Previous order expected to be the generation order
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

        # sort edges in descending order of node2pos position
        for node_id in self.pend_edges_by_node:
            edges = []
            for (idx, e) in enumerate(self.pend_edges_by_node[node_id]):
                other_id = e[0]
                if other_id == node_id:
                    other_id = e[2]
                edges.append((node_id_2_node_number[other_id], idx))
            edges.sort(reverse=True)
            new_edges_for_node = []
            for (_, idx) in edges:
                new_edges_for_node.append(
                    self.pend_edges_by_node[node_id][idx]
                )
            self.pend_edges_by_node[node_id] = new_edges_for_node

        # Will store gold_amr.nodes.keys() and edges as we predict them
        self.node_map = {}
        self.node_reverse_map = {}
        self.predicted_edges = []

    def get_arc_action(self, machine):

        # Loop over edges not yet created
        top_node_id = machine.node_stack[-1]
        current_id = self.node_reverse_map[top_node_id]
        for (src, label, tgt) in self.pend_edges_by_node[current_id]:
            # skip if it involves nodes not yet created
            if src not in self.node_map or tgt not in self.node_map:
                continue
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
                # NOTE include the relation marker ':' in action names
                #FIXME temp fix of adding ':' to action names
                #assert label[0] == ':'
                if label[0]!=':':
                    print("No semi colon in label ",label)
                    label = ':'+label
                return [f'>LA({index},{label})'], [1.0]

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
                # NOTE include the relation marker ':' in action names
                #FIXME temp fix of adding ':' to action names
                # assert label[0] == ':'
                if label[0]!=':':
                    print("No semi colon in label ",label)
                    label = ':'+label

                return [f'>RA({index},{label})'], [1.0]

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
        #NEW_ACTION
        # # new sentence in document
        # if machine.tok_cursor < len(machine.tokens):
        #     if machine.tokens[machine.tok_cursor] == '<next_sent>':
        #         return ['NEXT_SENTENCE'], [1.0]
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

            if (
                self.use_copy and
                normalize(machine.tokens[machine.tok_cursor]) == target_node
            ):
                # COPY
                return [('COPY', nid)], [1.0]
            else:
                # Generate
                return [(target_node, nid)], [1.0]

        # Move monotonic attention
        if machine.tok_cursor < len(machine.tokens):
            return ['SHIFT'], [1.0]

        return ['CLOSE'], [1.0]


class AMRStateMachine():

    def __init__(self, reduce_nodes=None, absolute_stack_pos=True,
                 use_copy=True, debug=False,norm='v0',ignore_coref=False):

        # debug flag
        self.debug = debug

        # Here non state variables (do not change across sentences) as well as
        # slow initializations
        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to stack top
        self.absolute_stack_pos = absolute_stack_pos
        if not absolute_stack_pos:
            # to support align-mode. Also it may be unnecesarily confusing to
            # have to modes
            raise NotImplementedError('Deprecated relative stack indexing')

        # use copy action
        self.use_copy = use_copy
        self.ignore_coref = ignore_coref
        self.norm = norm


        # base actions allowed
        self.base_action_vocabulary = [
            'SHIFT',   # Move cursor
            'COPY',    # Copy word under cursor to node (add node to stack)
            'ROOT',    # Label node as root
            # Arc from node under cursor (<label>, <to position>) (to be
            # different from LA the city)
            '>LA',
            '>RA',      # Arc to node under cursor (<label>, <from position>)
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

        if not self.use_copy:
            self.base_action_vocabulary.remove('COPY')

    def canonical_action_to_dict(self, vocab):
        """
        Map the canonical actions to ids in a vocabulary, each canonical action
        corresponds to a set of ids.

        CLOSE is mapped to eos </s> token.
        """
        canonical_act_ids = dict()
        vocab_act_count = 0
        assert vocab.eos_word == '</s>'
        for i in range(len(vocab)):
            # NOTE can not directly use "for act in vocab" -> this will never
            # stop since no stopping iter implemented
            act = vocab[i]
            if (
                act in ['<s>', '<pad>', '<unk>', '<mask>']
                or act.startswith('madeupword')
            ):
                continue
            cano_act = self.get_base_action(
                act) if i != vocab.eos() else 'CLOSE'
            if cano_act in self.base_action_vocabulary:
                vocab_act_count += 1
                canonical_act_ids.setdefault(cano_act, []).append(i)
        return canonical_act_ids

    def reset(self, tokens, gold_amr=None, reject_align_samples=False):
        '''
        Reset state variables and set a new sentence

        Use gold_amr for align mode

        reject_align_samples = True raises BadAlignModeSample if sample does
        not satisfy contraints
        '''

        assert tokens is not None, \
            "State machine requires sentence to be tokenized"

        # state
        self.tokens = list(tokens)
        self.tok_cursor = 0
        self.node_stack = []
        self.action_history = []
        self.node_action_vars = []

        # AMR as we construct it
        # NOTE: We will use position of node generating action in action
        # history as node_id
        self.nodes = {}
        self.edges = []
        self.roots = []
        self.root = None
        self.alignments = defaultdict(list)
        # set to true when machine finishes
        self.is_closed = False
        # for multi senetence action sequences
        #NEW_ACTION
        # self.sentence_reset()

        # def sentence_reset(self):
        #     self.root = None
        #     self.sentence_nodes = []


        # state info useful in the model
        self.actions_tokcursor = []

        # align mode
        self.gold_amr = gold_amr
        if gold_amr:
            # this will track the node alignments between
            self.align_tracker = AlignModeTracker(
                gold_amr,
                reject_samples=reject_align_samples
            )
            self.align_tracker.update(self)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as fid:
            config = json.loads(fid.read())
        # remove state
        if 'state' in config:
            del config['state']
        return cls(**config)

    def save(self, config_path, state=False):
        with open(config_path, 'w') as fid:
            # NOTE: Add here all *non state* variables in __init__()
            data = dict(
                reduce_nodes=self.reduce_nodes,
                absolute_stack_pos=self.absolute_stack_pos,
                use_copy=self.use_copy
            )
            if state:
                data['state'] = dict(
                    tokens=self.tokens,
                    action_history=self.action_history
                )
                if self.gold_amr:
                    data['state']['gold_amr'] = \
                        penman.encode(self.gold_amr.penman)
                    data['state']['gold_id_map'] = \
                        dict(self.align_tracker.gold_id_map)
            fid.write(json.dumps(data))

    def __deepcopy__(self, memo):
        """
        Manual deep copy of the machine

        avoid deep copying heavy files
        """
        cls = self.__class__
        result = cls.__new__(cls)
        # DEBUG: usew this to detect very heavy constants that can be referred
        # import time
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # start = time.time()
            # if k in ['actions_by_stack_rules']:
            #     setattr(result, k, v)
            # else:
            #     setattr(result, k, deepcopy(v, memo))
            setattr(result, k, deepcopy(v, memo))
            # print(k, time.time() - start)
        return result

    def state_str(self, node_map=None):
        '''
        Return string representing machine state
        '''
        string = ' '.join(self.tokens[:self.tok_cursor])
        if self.tok_cursor < len(self.tokens):
            string += f' \033[7m{self.tokens[self.tok_cursor]}\033[0m '
            string += ' '.join(self.tokens[self.tok_cursor+1:]) + '\n\n'
        else:
            string += '\n\n'

        # string += ' '.join(self.action_history) + '\n\n'
        for action in self.action_history:
            if action in ['SHIFT', 'ROOT', 'CLOSE'] or action.startswith('>'):
                string += f'{action} '
            else:
                string += f'\033[7m{action}\033[0m '
        string += '\n\n'

        if self.edges:
            # This can die with cicles saying its a disconnected graph
            amr_str = self.get_amr(node_map=node_map).to_penman()
        else:
            # invalid AMR
            amr_str = '\n'.join(
                f'({nid} / {nname})' for nid, nname in self.nodes.items()
            )
        amr_str = '\n'.join(
            x for x in amr_str.split('\n') if x and x[0] != '#'
        )
        string += f'{amr_str}\n\n'

        return string

    def __str__(self):

        if self.gold_amr:
            # align mode
            # create a node map relating decoded and gold ids with color code
            dec2gold = self.align_tracker.get_flat_map()
            node_map = {
                k: green_font(f'{k}-{dec2gold[k][0]}')
                if k in dec2gold else yellow_font(k)
                for k in self.nodes
            }
            tracker = self.align_tracker
            return f'{self.state_str(node_map)}\n{tracker.__str__()}'
        else:
            return self.state_str()

    def get_current_token(self):
        if self.tok_cursor >= len(self.tokens):
            return None
        else:
            return self.tokens[self.tok_cursor]

    def get_base_action(self, action):
        """Get the base action form, by stripping the labels, etc."""
        if action in self.base_action_vocabulary:
            return action
        # remaining ones are ['>LA', '>RA', 'NODE']
        # NOTE need to deal with both '>LA(pos,label)' and '>LA(label)', as in
        # the vocabulary the pointers are peeled off
        if arc_regex.match(action) or arc_nopointer_regex.match(action):
            return action[:3]
        return 'NODE'

    def _get_valid_align_actions(self):
        '''Get actions that generate given gold AMR'''

        # return arc actions if any
        # corresponding possible decoded edges
        arc_actions = []
        for (s, gold_e_label, t) in self.align_tracker.get_missing_edges(self):
            if s in self.node_stack[:-1] and t == self.node_stack[-1]:
                # right arc stack --> top
                action = f'>RA({s},{gold_e_label})'
            else:
                # left arc stack <-- top
                action = f'>LA({t},{gold_e_label})'
            if action not in arc_actions:
                arc_actions.append(action)

        if arc_actions:
            # TODO: Pointer and label can only be enforced independently, which
            # means that if we hae two diffrent arcs to choose from, we could
            # make a mistake. We need to enforce an arc order.
            return arc_actions

        # otherwise choose between producing a gold node and shifting (if
        # possible)
        valid_base_actions = []
        for nname in self.align_tracker.get_missing_nnames():
            if normalize(nname) == self.get_current_token():
                valid_base_actions.append('COPY')
            else:
                valid_base_actions.append(normalize(nname))
        if self.tok_cursor < len(self.tokens):
            valid_base_actions.append('SHIFT')

        if valid_base_actions == []:
            # if no possible option, just close
            return ['CLOSE']
        else:
            return valid_base_actions

    def get_valid_actions(self, max_1root=True):

        # debug
        if self.debug:
            os.system('clear')
            print(self)
            set_trace()
            print()

        if self.gold_amr:

            # align mode (we know the AMR)
            return self._get_valid_align_actions()

        valid_base_actions = []
        gen_node_actions = ['COPY', 'NODE'] if self.use_copy else ['NODE']

        if self.tok_cursor < len(self.tokens):
            valid_base_actions.append('SHIFT')
            valid_base_actions.extend(gen_node_actions)
        #NEW_ACTION
        # if self.tok_cursor < len(self.tokens) and self.tokens[self.tok_cursor] == '<next_sent>':
        #         return ['NEXT_SENTENCE']

        if (
            self.action_history
            and self.get_base_action(self.action_history[-1]) in (
                gen_node_actions + ['ROOT', '>LA', '>RA']
            )
        ):
            valid_base_actions.extend(['>LA', '>RA'])

        if (
            self.action_history
            and self.get_base_action(self.action_history[-1])
                in gen_node_actions
        ):
            if max_1root:
                # force to have at most 1 root (but it can always be with no
                # root)
                if not self.root:
                    valid_base_actions.append('ROOT')
            else:
                valid_base_actions.append('ROOT')
        
        # if (
        #     self.action_history 
        #     and self.get_base_action(self.action_history[-1]) != 'ROOT' 
        #     and len(self.sentence_nodes) >= 1 and len(self.node_stack) >= 2 
        #     and 'NEXT_SENTENCE' in self.action_history 
        #     and '<unk>' not in valid_base_actions
        #     ):
        #     valid_base_actions.extend(coref_actions)

        if self.tok_cursor == len(self.tokens):
            assert not valid_base_actions \
                and self.action_history[-1] == 'SHIFT'
            valid_base_actions.append('CLOSE')

        if self.reduce_nodes:
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

        assert not self.is_closed

        # FIXME: Align mode can not allow '<unk>' node names but we need a
        # handling of '<unk>' that works with other NN vocabularies
        if self.gold_amr and action == '<unk>':
            valid_actions = ' '.join(self.get_valid_actions())
            raise Exception(
                f'{valid_actions} is an <unk> action: you can not use align '
                'mode enforcing actions not in the vocabulary'
            )

        self.actions_tokcursor.append(self.tok_cursor)

        if re.match(r'CLOSE', action):

            if self.gold_amr:
                # sanity check, we got the exact same AMR
                check_gold_alignment(
                    self,
                    trace=False,
                    reject_samples=self.align_tracker.reject_samples
                )

            self.is_closed = True

        elif re.match(r'ROOT', action):
            self.root = self.node_stack[-1]

        elif action in ['SHIFT']:
            # Move source pointer
            self.tok_cursor += 1
        #NEW_ACTION
        # elif action in ['NEXT_SENTENCE']:
        #     while self.tok_cursor < len(self.tokens) and self.tokens[self.tok_cursor] != '<next_sent>':
        #         self.cursor += 1
        #     # Move source pointer                                         
        #     self.tok_cursor += 1
        #     # save current sentence nodes and root
        #     self.connect_sentence_graph()
        #     self.sentence_reset()

        # TODO: Separate REDUCE actions into its own method
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
            #NEW_ACTION
            # for keeping each sentence's nodes separate
            # self.sentence_nodes.append(node_id)

        else:

            # Interpret action as a node name
            # Note that the node_id is the position of the action that
            # generated it
            node_id = len(self.action_history)
            self.nodes[node_id] = action
            self.node_stack.append(node_id)
            self.alignments[node_id].append(self.tok_cursor)
            #NEW_ACTION
            # for keeping each sentence's nodes separate
            # self.sentence_nodes.append(node_id)

        # Update align mode tracker after machine state has been updated
        if self.gold_amr:
            self.align_tracker.update(self)

        # in align mode we can not predict ROOT in situ, but its determined
        # from alignments when available
        if self.gold_amr and self.root is None:

            # map from decoded nodes to gold nodes
            gold2dec = self.align_tracker.get_flat_map(reverse=True)

            if self.gold_amr.root in gold2dec:
                # this will not work for partial AMRs
                self.root = gold2dec[self.gold_amr.root]

        # Action for each time-step
        self.action_history.append(action)

    def connect_sentence_graph(self):

        #import ipdb ; ipdb.set_trace()
        if not self.sentence_nodes:
            return

        sen_nodes = self.sentence_nodes
        sen_root  = self.root
        
        descendents = {n: {n} for n in sen_nodes}
        potential_roots = [n for n in sen_nodes]
        for x, r, y in self.edges:
            if y in potential_roots and (x not in potential_roots or x not in descendents[y]):
                potential_roots.remove(y)
            if x in sen_nodes and y in sen_nodes:
                descendents[x].update(descendents[y])
            for n in descendents:
                if x in descendents[n]:
                    descendents[n].update(descendents[x])

        for node in potential_roots.copy():
            for other_node in sen_nodes:
                if node != other_node and node in descendents[other_node]:
                    potential_roots.remove(node)
                    break
                    
        disconnected = potential_roots.copy()
        for n in potential_roots.copy():
            if len([e for e in self.edges if e[0] == n]) == 0:
                potential_roots.remove(n)

        #if len(disconnected) > 1:
        #    import ipdb; ipdb.set_trace()
                
        # assign root
        if potential_roots:
            if sen_root is None or sen_root not in potential_roots:
                sen_root = potential_roots[0]
            disconnected.remove(sen_root)
        else:
            if sen_root is None and len(sen_nodes):
                sen_root = max(sen_nodes,
                            key=lambda x: len([e for e in self.edges if e[0] == x])
                            - len([e for e in self.edges if e[2] == x]))
        # connect graph
        if len(disconnected) > 0 and sen_root is not None:
            for n in disconnected:
                if n != sen_root:
                    self.edges.append((sen_root, ":rel", n))

        self.roots.append(sen_root)
        self.sidx2nodes[sen_root] = sen_nodes[:]
        for node in sen_nodes:
            self.nodes2sidx[node] = sen_root
            
   
    def get_amr(self, node_map=None):

        # ensure AMR is valid
        tokens, nodes, edges, root, alignments = create_valid_amr(
            self.tokens, self.nodes, self.edges, self.root, self.alignments
        )

        # create an AMR class
        amr = AMR_doc(tokens, nodes, edges, root, alignments=alignments)

        # use valid node names
        if node_map is None:
            node_map = amr.get_node_id_map()
        amr.remap_ids(node_map)

        return amr

    def get_annotation(self, node_map=None, jamr=False):

        if self.gold_amr:
            assert self.gold_amr.penman, "Align mode requires AMR.from_penman"
            assert not jamr, "Align dows not support --jamr write"

            # just add alignments to existing penman
            return self.align_tracker.add_alignments_to_penman(self)

        else:
            return self.get_amr(node_map=node_map).to_penman(jamr=jamr)


def get_ngram(sequence, order):
    ngrams = []
    for n in range(len(sequence) - order + 1):
        ngrams.append(tuple(sequence[n:n+order]))
    return ngrams


class Stats():

    def __init__(self, ignore_indices, ngram_stats=False, breakpoint=False,
                 stop_if_error=False):
        self.index = 0
        self.ignore_indices = ignore_indices
        self.stop_if_error = stop_if_error
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
        if self.stop_if_error:
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
        # remove the ')' at last position
        properties = properties[:-1]
        # split to pointer value and label
        properties = properties.split(',')
        pos = int(properties[0].strip())
        # remove any leading and trailing white spaces
        label = properties[1].strip()
        action_label = action + '(' + label + ')'
        return (action_label, pos)
    else:
        return (action, pad)


class StatsForVocab:
    """
    Collate stats for predicate node names with their frequency, and list of
    all the other action symbols. For arc actions, pointers values are
    stripped. The results stats (from training data) are going to decide which
    node names (the frequent ones) to be added to the vocabulary used in the
    model.
    """

    def __init__(self, no_close=False):
        # DO NOT include CLOSE action (as this is internally managed by the eos
        # token in model)
        # NOTE we still add CLOSE into vocabulary, just to be complete although
        # it is not used
        self.no_close = no_close

        self.nodes = Counter()
        self.left_arcs = Counter()
        self.right_arcs = Counter()
        self.control = Counter()

        # node stack stats (candidate pool for the pointer)
        self.node_stack_corpus = []

    def update(self, action, machine):

        # new sentence
        if len(machine.action_history) == 1:
            self.node_stack_corpus.append([])
        # update stats for node stack size
        # if we have an arc action, store pool size
        if action.startswith('>LA') or action.startswith('>RA'):
            # store for current sentence
            pool_size = len(machine.node_stack) - 1
            self.node_stack_corpus[-1].append(pool_size)

        if self.no_close:
            if action in ['CLOSE', '_CLOSE_']:
                return

        if la_regex.match(action) or la_nopointer_regex.match(action):
            # LA(pos,label) or LA(label)
            action, pos = peel_pointer(action)
            # NOTE should be an iterable instead of a string; otherwise it'll
            # be character based
            self.left_arcs.update([action])
        elif ra_regex.match(action) or ra_nopointer_regex.match(action):
            # RA(pos,label) or RA(label)
            action, pos = peel_pointer(action)
            self.right_arcs.update([action])
        elif action in machine.base_action_vocabulary:
            self.control.update([action])
        else:
            # node names
            self.nodes.update([action])

    def display(self):

        # Uniform Pointer Perplexity
        node_stack_corpus = [x for x in self.node_stack_corpus if x]
        UPP_sents = list(map(np.mean, node_stack_corpus))
        print(
            f'Average Node Memory Size: {np.mean(UPP_sents):.2f}'
            f' (max {np.max(UPP_sents):.2f} at sent {np.argmax(UPP_sents)})'
        )
        print('Total number of different node names: ', end='')
        print(len(list(self.nodes.keys())))
        # print('Most frequent node names:')
        # print(self.nodes.most_common(20))
        # print('Most frequent left arc actions:')
        # print(self.left_arcs.most_common(20))
        # print('Most frequent right arc actions:')
        # print(self.right_arcs.most_common(20))
        # print('Other control actions:')
        # print(self.control)

    def write(self, path_prefix):
        """
        Write the stats into file. Two files will be written: one for nodes,
        one for others.
        """
        path_nodes = path_prefix + '.nodes'
        path_others = path_prefix + '.others'
        with open(path_nodes, 'w') as f:
            for k, v in self.nodes.most_common():
                print(f'{k}\t{v}', file=f)
        with open(path_others, 'w') as f:
            for k, v in chain(
                self.control.most_common(),
                self.left_arcs.most_common(),
                self.right_arcs.most_common()
            ):
                print(f'{k}\t{v}', file=f)

def make_doc_amrs(corefs, amrs, coref=True,chains=True):
    doc_amrs = {}

    desc = "making doc-level AMRs"
    if not coref:
        desc += " (without corefs)"
    for doc_id in tqdm(corefs, desc=desc):
        (doc_corefs,doc_sids,fname) = corefs[doc_id]
        if doc_sids[0] not in amrs:
            import ipdb; ipdb.set_trace()
        doc_amr = deepcopy(amrs[doc_sids[0]])
        for sid in doc_sids[1:]:
            if sid not in amrs:
                import ipdb; ipdb.set_trace()
            if amrs[sid].root is None:
                continue
            doc_amr = doc_amr + amrs[sid]
        doc_amr.amr_id = doc_id
        doc_amr.doc_file = fname
        for e in doc_amr.edges:
            if ':' not in e[1]:
                print("BEFORE COREF MISSING COLON ",e)
        if coref:
            if chains:
                doc_amr.add_corefs(doc_corefs)
            else:
                doc_amr.add_edges(doc_corefs)
       
        for e in doc_amr.edges:
            if ':' not in e[1]:
                print("AFTER COREF MISSING COLON ",e)
        doc_amrs[doc_id] = doc_amr

    return doc_amrs


def recent_member_by_sent(chain,sid,doc_id):
    def get_sid_fromstring(string):
         
        sid = [int(s) for s in re.findall(r'\d+', string)]
        assert len(sid)==1
        return sid[0]

    sid = get_sid_fromstring(sid)    
    diff = lambda chain : abs(get_sid_fromstring(chain[0].split('.')[0]) - sid)
    ent = min(chain, key=diff)
    if get_sid_fromstring(ent[0].split('.')[0]) > sid:
        print(doc_id," closest sent is higher than connecting node ",ent[0],sid)
    return ent[0]

    

def recent_member_by_align(chain,src_align,doc_id,rel=None):
 
    diff = lambda chain : abs(chain[1]-src_align)
    ent = min(chain, key=diff)
    if ent[1]>= src_align:
        print(doc_id," coref edge missing ",ent[1],src_align,rel)      
    return ent[0]

#convert v0 coref edge to connect to most recent sibling in the chain
def make_pairwise_edges(damr):
    
    ents_chain = defaultdict(list)
    edges_to_delete = []
    nodes_to_delete = []
    doc_id = damr.amr_id
    # damr.edges.sort(key = lambda x: x[0])
    for idx,e in enumerate(damr.edges):
        if e[1] == ':coref-of':
            # if len(ents[e[2]])==0:
                #damr.edges[idx] = (e[0],':coref-edge',ents[e[2]][-1])
            # else:
            edges_to_delete.append(e)

            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ents_chain[e[2]].append((e[0],damr.alignments[e[0]][0]))
            else:
                #FIXME adding the src node of a coref edge with no alignments member of chain with closest sid
                # print(doc_id + '  ',e[0],' alignments is None  src node in coref edge, not adding it ')
                sid = e[0].split('.')[0]
                if len(ents_chain[e[2]]) >0 :
                    ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
                    damr.edges[idx] = (e[0],e[1],ent)
                #FIXME
                else:
                    print("coref edge missing, empty chain, edge not added")
                
            assert e[2].startswith('rel')
       

    
    #adding coref edges between most recent sibling in chain    
    for cents in ents_chain.values():
        cents.sort(key=lambda x:x[1])
        for idx in range(0,len(cents)-1):
            damr.edges.append((cents[idx+1][0],':coref-edge',cents[idx][0]))

    for e in edges_to_delete:
        while e in damr.edges:
            damr.edges.remove(e)

    #connecting all other edges involving chain to most recent member in the chain
    for idx,e in enumerate(damr.edges):
        #Both src and target are coref nodes
        if e[0] in ents_chain and e[2] in ents_chain:
            damr.edges[idx] = (ents_chain[e[0]][-1][0],e[1],ents_chain[e[2]][-1][0])
        
        elif e[2] in ents_chain.keys():
            #src node is a normal amr node
            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ent = recent_member_by_align(ents_chain[e[2]],damr.alignments[e[0]][0],doc_id,e[1])
                
            else:
                #FIXME assigning src node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[0].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
            damr.edges[idx] = (e[0],e[1],ent)

        elif e[0] in ents_chain.keys():
            if e[2] in damr.alignments and damr.alignments[e[2]] is not None:
                ent = recent_member_by_align(ents_chain[e[0]],damr.alignments[e[2]][0],doc_id,e[1])
            else:
                #FIXME assigning tgt node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[2].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[0]],sid,doc_id)
        
            damr.edges[idx] = (ent,e[1],e[2])

       
    for n in ents_chain.keys():
        while n in damr.nodes:
            del damr.nodes[n]
    
        
    
    return damr

def connect_sen_amrs(amr):

    if len(amr.roots) <= 1:
        return

    node_id = amr.add_node("document")
    amr.root = str(node_id)
    for (i,root) in enumerate(amr.roots):
        amr.edges.append((amr.root, ":snt"+str(i+1), root))

def pack_amrs(amrs):

    packed_amrs = []
    keys = list(amrs.keys())
    random.shuffle(keys)
    packed_amr = None
    for _ in range(5):
        for key in keys:
            amr = amrs[key]
            if packed_amr:
                potential_src_length = len(packed_amr.tokens) + len(amr.tokens) + 1
                tgt_length = len(packed_amr.nodes) + len(packed_amr.edges) + len(packed_amr.roots) + len(amr.nodes) + len(amr.edges) + 1 
                if potential_src_length > 800 or tgt_length > 1000 :
                    packed_amrs.append(packed_amr)
                    packed_amr = deepcopy(amr)
                else:
                    packed_amr += amr
            else:
                packed_amr = deepcopy(amr)
        random.shuffle(keys)
        
    return packed_amrs
def oracle(args):

    if args.jamr:
        raise Exception('--jamr format is deprecated')


    if args.coref_fof:
        
        coref_files = [args.fof_path+line.strip() for line in open(args.coref_fof)]
        corefs = process_corefs(coref_files)
        
    # Read AMR as a generator with tqdm progress bar
    amr_file = args.in_amr if args.in_amr else args.in_aligned_amr
    # tqdm_amrs = read_blocks(amr_file)
    # tqdm_amrs.set_description(f'Computing oracle')

    if args.coref_fof:

        
        tqdm_amrs_str = read_blocks(amr_file)
        # amrs = read_amr_penman(tqdm_amrs_str)
        amrs = read_amr_penman(tqdm_amrs_str)
        
        plain_doc_amrs = make_doc_amrs(corefs,amrs,coref=False).values()
        damrs = []
        if args.out_amr is None:
            args.out_amr = args.out_actions.rstrip('.actions')+'_'+args.norm+'.docamr'
        with open(args.out_amr+".nocoref", 'w') as fid:
            for amr in plain_doc_amrs:
                damr = deepcopy(amr)
                connect_sen_amrs(damr)
                damr.normalize(args.norm)
                fid.write(damr.__str__())        
        # use corefs to merge sentence level AMRs into Documentr level AMRs
        amrs = make_doc_amrs(corefs,amrs).values()
       
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                damr = deepcopy(amr)
                connect_sen_amrs(damr)
                damr.normalize(args.norm)
                
                damr = make_pairwise_edges(damr)
                damr.alignments['d'] = [len(damr.tokens)-1]
                damrs.append(damr)
                fid.write(damr.__str__())
  

    # read AMR alignments if provided
    if args.in_alignment_probs:
        corpus_align_probs = read_neural_alignments(args.in_alignment_probs)
        assert len(corpus_align_probs) == len(amrs)
    else:
        corpus_align_probs = None

    # broken annotations that we ignore in stats
    # 'DATA/AMR2.0/aligned/cofill/train.txt'
    ignore_indices = [
        8372,   # (49, ':time', 49), (49, ':condition', 49)
        17055,  # (3, ':mod', 7), (3, ':mod', 7)
        27076,  # '0.0.2.1.0.0' is on ::edges but not ::nodes
        # for AMR 3.0 data: DATA/AMR3.0/aligned/cofill/train.txt
        # self-loop:
        # "# ::edge vote-01 condition vote-01 0.0.2 0.0.2",
        # "# ::edge vote-01 time vote-01 0.0.2 0.0.2"
        9296,
    ]
    # NOTE we add indices to ignore for both amr2.0 and amr3.0 in the same list
    # and used for both oracles, since: this would NOT change the oracle
    # actions, but only ignore sanity checks and displayed stats after oracle
    # run

    # Initialize machine
    machine = AMRStateMachine(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions,
        use_copy=args.use_copy
    )
    # Save machine config
    if args.out_machine_config:
        machine.save(args.out_machine_config)

    # initialize oracle
    oracle = AMROracle(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions,
        use_copy=args.use_copy,
        force_align_ner=args.force_align_ner
    )

    # will store statistics and check AMR is recovered
    stats = Stats(ignore_indices, ngram_stats=False,
                  stop_if_error=args.stop_if_error)
    stats_vocab = StatsForVocab(no_close=False)
    for idx,amr in enumerate(tqdm(damrs, desc='Oracle')):

        

        # spawn new machine for this sentence
        machine.reset(amr.tokens)

        # initialize new oracle for this AMR
        if corpus_align_probs:
            # sampling of alignments
            oracle.reset(amr, alignment_probs=corpus_align_probs[idx],
                         alignment_sampling_temp=args.alignment_sampling_temp)
        else:
            oracle.reset(amr)

        # proceed left to right throught the sentence generating nodes
        while not machine.is_closed:

            # get valid actions
            _ = machine.get_valid_actions()

            # oracle
            actions, scores = oracle.get_actions(machine)
            # actions = [a for a in actions if a in valid_actions]
            # most probable
            action = actions[np.argmax(scores)]
            node_var = "_"

            # if it is node generation, keep track of original id in gold amr
            if isinstance(action, tuple):
                action, gold_node_id = action
                node_id = len(machine.action_history)
                oracle.node_map[gold_node_id] = node_id
                oracle.node_reverse_map[node_id] = gold_node_id
                node_var = "None"
                if gold_node_id in oracle.gold_amr.nvars:
                    node_var = oracle.gold_amr.nvars[gold_node_id]

            # update machine,
            machine.update(action)
            machine.node_action_vars.append(node_var)

            # update machine stats
            stats.update_machine_stats(machine)

            # update vocabulary
            stats_vocab.update(action, machine)

        # Sanity check: We recovered the full AMR
        stats.update_sentence_stats(oracle, machine)

        # do not write 'CLOSE' in the action sequences
        # this might change the machine.action_history in place, but it is the
        # end of this machine already
        close_action = stats.action_sequences[-1].pop()
        assert close_action == 'CLOSE'

    # display statistics
    stats.display()

    # save action sequences and tokens
    if args.out_actions:
        write_tokenized_sentences(
            args.out_actions,
            stats.action_sequences,
            '\t'
        )
    if args.out_tokens:
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
            fid.write(f'{annotation}\n')


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
        assert not args.force_align_ner
        play(args)

    elif args.in_aligned_amr or args.in_amr:
        # Run oracle and determine actions from AMR
        assert args.in_aligned_amr or (args.in_amr and args.in_alignment_probs)
        if args.out_actions:
            # if we write the actions we must aslo save the config
            assert args.out_machine_config
            assert args.out_tokens
        assert not args.in_tokens
        assert not args.in_actions
        oracle(args)

    else:
        raise Exception('Needs --in-actions or --in-*amr')


def argument_parser():

    parser = argparse.ArgumentParser(
        description='Produces oracle sequences given AMR alignerd to sentence'
    )
    # Single input parameters
    parser.add_argument(
        "--in-aligned-amr",
        help="In file containing AMR in penman format AND isi alignments ",
        type=str
    )

    parser.add_argument(
        "--coref-fof",
        help="xml files containing AMR coreference information ",
        type=str
    )
    parser.add_argument(
        "--norm",
        help="norm of DocAMR ",
        type=str,
        default='docAMR'
    )

    parser.add_argument(
        "--in-amr",
        help="In file containing AMR in penman format, requires "
             "--in-alignment-probs",
        type=str
    )

    parser.add_argument(
        "--jamr",
        help="Read AMR and alignments from JAMR and not PENMAN",
        action='store_true'
    )
    parser.add_argument(
        "--in-alignment-probs",
        help="Alignment probabilities produced by align_cfg/main.py",
        type=str
    )
    parser.add_argument(
        "--alignment-sampling-temp",
        help="Temperature for sampling alignments",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--force-align-ner",
        help="(<tag> :name name :opx <token>) allways aligned to token",
        action='store_true'
    )

    parser.add_argument(
        '--ignore-coref',
        action='store_true',
        help='produce outputs without cross sentential corefs'
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
        "--use-copy",
        help='Use COPY action to copy words at source token cursor',
        type=int,
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
    parser.add_argument(
        "--stop-if-error",
        help="set_trace if a reconstructed AMR is not perfect",
        type=str,
    )
    parser.add_argument(
        "--fof-path",
        help="path to coref fof files",
        type=str


    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
