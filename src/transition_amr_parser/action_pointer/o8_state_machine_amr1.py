import json
import os
import re
from collections import Counter
from copy import deepcopy

from transition_amr_parser.amr import AMR

"""
AMRStateMachine applies operations in a transition-based AMR parser, but combined with a pointer for arcs.
It maintains a cursor on the token sequence and moves from left to right and apply actions to generate an AMR graph.

Actions are
    SHIFT : move cursor to next position in the token sequence
    REDUCE : delete current token
    MERGE : merge two tokens (for MWEs)
    ENTITY(type) : form a named entity, or a subgraph
    PRED(label) : form a new node with label
    COPY_LEMMA : form a new node by copying lemma
    COPY_SENSE01 : form a new node by copying lemma and add '01'
    DEPENDENT(edge,node) : Add a node which is a dependent of the current node
    LA(pos,label) : form a left arc from the current node to the previous node at location pos
    RA(pos,label) : form a left arc to the current node from the previous node at location pos
    CLOSE : complete AMR, run post-processing

Note:
    - Do not put actions inside actions, which may cause issue with state management.
"""

entity_rules_json = None
NUM_RE = re.compile(r'^([0-9]|,)+(st|nd|rd|th)?$')
entity_rule_stats = Counter()
entity_rule_totals = Counter()
entity_rule_fails = Counter()

# get path of provided entity_rules
repo_root = os.path.realpath(f'{os.path.dirname(__file__)}')
#entities_path = f'{repo_root}/o8_entity_rules.json'
entities_path = f'{repo_root}/entity_rules.json'

default_rel = ':rel'


def white_background(string):
    return "\033[107m%s\033[0m" % string


def red_background(string):
    return "\033[101m%s\033[0m" % string


def green_background(string):
    return "\033[102m%s\033[0m" % string


def black_font(string):
    return "\033[30m%s\033[0m" % string


def blue_font(string):
    return "\033[94m%s\033[0m" % string


def green_font(string):
    return "\033[92m%s\033[0m" % string


def stack_style(string, confirmed=False):
    if confirmed:
        return black_font(green_background(string))
    else:
        return black_font(white_background(string))


def reduced_style(string):
    return black_font(red_background(string))


class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)


class AMRStateMachine:
    """AMR state machine. For a token sequence, run a series of actions and build an AMR graph as a result.

    Args:
        tokens (List[str]): a sequence of tokens. Default: None
        tokseq_len (int): token sequence length; only used under `canonical_mode`. Default: None
        canonical_mode (bool): whether to 1) run the state machine with canonical actions, and 2) without generating
            the AMR graph along the way, and 3) under this mode we can only provide the token sequence length
            (including the ending "<ROOT>" token) instead of feeding the actual token sequence.
            This will run the machine with minimum operations, by ignoring the detailed labels
            associated with each action and only keep the internal token cursor and canonical action history states.
            This is used during action sequence generation (e.g. beam search from a model) to do sanity check for next
            actions to restrict the action space. Default: False

    Note:
        - currently, token sequence should always have "<ROOT>" as the last token; although we add it if not existing,
          we recommend providing it from the input to be better orgainized, and
        - under the `canonical_mode`, when inputting the token sequence length, it should be the length including the
          ending "<ROOT>" token.
    """
    # canonical actions without the detailed node/edge labels and action properties (e.g. arc pointer value)
    canonical_actions = ['REDUCE',
                         'MERGE',
                         'ENTITY', 'PRED', 'COPY_LEMMA', 'COPY_SENSE01',    # add new node
                         'DEPENDENT',
                         'LA', 'RA',
                         'SHIFT',
                         'LA(root)',    # specific on the "<ROOT>" node
                         'CLOSE']

    def __init__(self, tokens=None, tokseq_len=None, verbose=False, add_unaligned=0,
                 actions_by_stack_rules=None, amr_graph=True,
                 canonical_mode=False,
                 spacy_lemmatizer=None,
                 entities_with_preds=None):
        # TODO verbose not used, actions_by_stack_rules not used

        # check for canonical mode
        assert tokens is not None or (canonical_mode and tokseq_len is not None)
        if canonical_mode:
            amr_graph = False
            # spacy_lemmatizer = None

        self.entities_with_preds = entities_with_preds

        self.canonical_mode = canonical_mode
        self.actions_canonical = []    # used in canonical mode
        self.actions_nodemask = []     # used in canonical mode
        self.actions_tokcursor = []    # used in canonical mode, as well as normal mode

        # graph structure information to be used in the model; used in canonical mode
        self.actions_edge_mask = []
        self.actions_latest_node = None
        self.actions_edge_cur_node = []
        self.actions_edge_pre_node = []
        self.actions_edge_direction = []

        if tokens is not None:
            # word tokens of sentence
            self.tokens = tokens.copy()

            # spacy lemmatizer
            # TODO change this to be created inside, by having spacy_lemmatizer as a bool flag.
            # ASK Ramon if there is a special reason
            self.spacy_lemmatizer = spacy_lemmatizer
            self.lemmas = None

            # add unaligned to the token sequence
            if add_unaligned and '<unaligned>' not in self.tokens:
                for i in range(add_unaligned):
                    self.tokens.append('<unaligned>')

            # add root to the token sequence
            if '<ROOT>' not in self.tokens:
                self.tokens.append("<ROOT>")

            self.tokseq_len = len(self.tokens)    # this includes the '<ROOT>', which is not treated specially
        else:
            self.tokens = None
            self.tokseq_len = tokseq_len

        # machine is active
        self.time_step = 0
        self.is_closed = False    # when True, no action can be applied except CLOSE
        self.is_postprocessed = False    # when True, no action can be applied

        # init current processing position in the token sequence
        self.tok_cursor = 0
        self.current_node_id = None    # frontier of the node id
        self.new_node_id = 1    # counting from 1 (root is -1); avoid 0 to be more different from None in if statements

        # build and store amr graph (needed e.g. for oracle)
        self.amr_graph = amr_graph

        # init amr: empty graph with only the root node added
        self.tokid_to_nodeid = {}    # one token could generate multiple nodes

        self.root_id = -1    # or `self.tokseq_len - 1` for consistent positive values with other nodes
        # TODO 'root' should be tied with -1 currently <-- since -1 is a must for self.connect_graph() processing
        if self.amr_graph:
            self.amr = AMR(tokens=self.tokens)
            for i, tok in enumerate(self.tokens):
                if tok != "<ROOT>":
                    # note that the node id is NOT shifted by 1, compared with the AMR alignments
                    self.tokid_to_nodeid[i] = []    # one token can generate multiple nodes
                else:
                    self.amr.nodes[self.root_id] = tok
                    self.tokid_to_nodeid[i] = [self.root_id]

        # action sequence and parser AMR target output
        self.actions = []
        self.actions_to_nodes = []    # node ids generated by each action; None for no node by this action
        self.actions_to_nlabels = []
        self.actions_to_elabels = []
        self.alignments = {}

        # information for oracle
        self.merged_tokens = {}                    # keys are token ids of the last merged token
        self.entities = []                         # node ids
        self.entity_tokens = {}                    # named entities, key: node ids, value: surface tokens
        self.entity_tokenids = []                  # token ids on which ENTITY is done

    def __str__(self):
        """Command line styling"""
        display_str = ""
        # Tokens
        tokens_str = ""
        for position, token in enumerate(self.tokens):
            pad = ' ' if position > 0 else ''
            if position == self.tok_cursor:
                tokens_str += f'{pad}{stack_style(token)}'
            else:
                tokens_str += f'{pad}{token}'
        # Actions
        action_str = ' '.join([a for a in self.actions])
        return tokens_str + '\n\n' + action_str

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
            if k in ['spacy_lemmatizer', 'actions_by_stack_rules']:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
            # print(k, time.time() - start)
        # import ipdb; ipdb.set_trace(context=30)
        return result

    @property
    def nodeid_to_tokid(self):
        self._nodeid_to_tokid = {}
        for t, ns in self.tokid_to_nodeid.items():
            for n in ns:
                self._nodeid_to_tokid.setdefault(n, set()).add(n)
        return self._nodeid_to_tokid

    def get_current_token(self, lemma=False):
        """Get the token at current cursor position."""
        if lemma:
            # Compute lemmas for this sentence and cache it
            if self.lemmas is None:
                assert self.spacy_lemmatizer, "No spacy_lemmatizer provided"
                toks = [self.normalize_token(x) if self.normalize_token(x) != "" else x for x in self.tokens]
                for tok in toks:
                    if tok == "":
                        import ipdb; ipdb.set_trace()
                self.lemmas = [
                    x.lemma_ for x in self.spacy_lemmatizer(toks[:-1])
                ] + ['ROOT']
            token = self.lemmas[self.tok_cursor]
        else:
            token = self.tokens[self.tok_cursor]
        return token

    @classmethod
    def read_action(cls, action):
        """Read action string and parse it."""
        if '(' not in action:
            return action, None
        elif action.startswith('LA') or action.startswith('RA'):
            # note: should use `startswith` otherwise might be incorrect cornercases, e.g.
            # "http://www.cms.gov/ActuarialStudies/Downloads/2010TRAlternativeScenario.pdf" is in the node label, which
            # contains 'RA'

            # arcs have format 'LA(pos,label)' and 'RA(pos,label)'
            # for pointer peeled format, we have 'LA(label)' and 'RA(label)', where we return the pointer as -1
            # root arc is 'LA(pos,root)' if '<ROOT>' token at the end

            items = action.split('(')
            arc_name = items[0]
            arc_args = items[1][:-1].split(',')
            if len(arc_args) == 1:
                # no pos provided
                return arc_name, (-1, arc_args[0])
            arc_pos = int(arc_args[0])
            arc_label = arc_args[1]
            return arc_name, (arc_pos, arc_label)
        else:
            items = action.split('(')
            action_label = items[0]
            arg_string = items[1][:-1]
            if action_label not in ['PRED', 'CONFIRM']:    # TODO 'CONFIRM' deprecated
                # split by comma respecting quotes
                props = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', arg_string)
            else:
                props = [arg_string]

            # TODO check if closing this (for functionality consistency) would cause any problem
            # # To keep original name to keep learner happy
            # if action_label == 'DEPENDENT':
            #     action_label = action

            return action_label, props

    @classmethod
    def canonical_action_form(cls, action):
        """Get the canonical form of an action with labels/properties."""
        if action in cls.canonical_actions:
            return action
        action, properties = cls.read_action(action)
        if action.startswith('LA'):
            if properties[1] == 'root':
                action = 'LA(root)'
        # assert action in cls.canonical_actions
        return action

    @classmethod
    def canonical_action_form_ptr(cls, action):
        """Get the canonical form of an action with labels/properties, and return the pointer value for arcs."""
        if action in cls.canonical_actions:
            return action, None
        action, properties = cls.read_action(action)
        if action.startswith('LA'):
            if properties[1] == 'root':
                action = 'LA(root)'
        if action.startswith('LA') or action.startswith('RA'):
            arc_pos = properties[0]
        else:
            arc_pos = None
        # assert action in cls.canonical_actions
        return action, arc_pos

    @classmethod
    def canonical_action_to_dict(cls, vocab):
        """Map the canonical actions to ids in a vocabulary, each canonical action corresponds to a set of ids.

        CLOSE is mapped to eos </s> token.
        """
        canonical_act_ids = dict()
        vocab_act_count = 0
        for i in range(len(vocab)):
            # NOTE can not directly use "for act in vocab" -> this will never stop since no stopping iter implemented
            act = vocab[i]
            cano_act = cls.canonical_action_form(act) if i != vocab.eos() else 'CLOSE'
            if cano_act in cls.canonical_actions:
                vocab_act_count += 1
                canonical_act_ids.setdefault(cano_act, []).append(i)
        # print for debugging
        # print(f'{vocab_act_count} / {len(vocab)} tokens in action vocabulary mapped to canonical actions.')
        return canonical_act_ids

    # TODO need to update the rules here
    def get_valid_canonical_actions(self):
        """Get valid actions at the current step, based on input tokens and the action history up to now.

        We only return the prefix of the action (action labels, or canonical actions), which could be mapped to detailed
        actions.
        """
        #raise NotImplementedError

        if self.canonical_mode:
            past_actions = self.actions_canonical
            actions_nodemask = self.actions_nodemask
        else:
            past_actions = self.actions
            actions_nodemask = map(lambda x: 0 if x is None else 1, self.actions_to_nodes)

        gen_node_actions = ['ENTITY', 'PRED', 'COPY_LEMMA', 'COPY_SENSE01']
        gen_arc_actions = ['LA', 'RA']
        pre_node_actions = ['REDUCE'] + gen_node_actions + ['MERGE']    # dependent on the remaining number of tokens
        post_node_actions = ['SHIFT', 'LA', 'RA'] #, 'DEPENDENT']
        post_merge_actions = gen_node_actions + ['MERGE']
        post_arc_actions = gen_arc_actions + ['SHIFT']
        cursor_moving_actions = ['REDUCE', 'MERGE', 'SHIFT']
        close_actions = ['CLOSE']
        root_token_actions = ['LA(root)', 'SHIFT']

        inside_entity = False
        for i in range(1,len(past_actions)+1):
            if past_actions[-i] in ['REDUCE', 'SHIFT']:
                break
            if past_actions[-i] in gen_node_actions: #== 'ENTITY':
                inside_entity = True

        #import ipdb; ipdb.set_trace()

        if self.tok_cursor == 0:
            # at the beginning
            if past_actions == []:
                # the first action
                if self.tokseq_len == 1:
                    raise ValueError('<ROOT> is always included, thus the token sequence length is at least 2.')
                    # cano_actions = ['REDUCE'] + new_node_actions
                else:
                    cano_actions = pre_node_actions
            else:
                # has previous action
                prev_action = self.canonical_action_form(past_actions[-1])
                assert prev_action not in cursor_moving_actions, 'impossible at the first tokens position'
                if self.tokseq_len == 1:
                    raise ValueError('<ROOT> is always included, thus the token sequence length is at least 2.')
                else:
                    if prev_action in gen_node_actions + ['DEPENDENT']:
                        cano_actions = post_node_actions
                    elif prev_action in gen_arc_actions:
                        cano_actions = post_arc_actions
                    elif prev_action in ['MERGE']:
                        cano_actions = post_merge_actions
                    elif prev_action in ['REDUCE', 'SHIFT']:
                        cano_actions = pre_node_actions
                    else:
                        raise ValueError('unallowed previous action sequence.')
        elif self.tok_cursor == self.tokseq_len - 1:    # at least 1, since self.tokseq_len is at least 2
            assert past_actions, 'impossible to move to the last token position with empty action sequence'
            # currently pointing to the '<ROOT>' token
            prev_action = self.canonical_action_form(past_actions[-1])
            if prev_action == 'LA(root)':
                cano_actions = ['SHIFT']  # , 'LA(root)']  # TODO this is only for training; change it back for decoding
            elif prev_action == 'SHIFT':
                # need to know if it's SHIFT at the last position or SHIFT from previous position
                # the last SHIFT could only have preceding actions being LA(root), SHIFT, REDUCE, which
                # are all not possible on previous SHIFT, since if not on the last <ROOT> node, there must be
                # some action to add a node before a SHIFT action (between SHIFT SHIFT or REDUCE SHIFT).
                prev_prev_action = self.canonical_action_form(past_actions[-2])
                if prev_prev_action in ['SHIFT', 'REDUCE']:
                    shift_on_last = True
                elif prev_prev_action == 'LA(root)':
                    shift_on_last = True
                else:
                    shift_on_last = False
                if shift_on_last:
                    cano_actions = close_actions
                else:
                    # just reached the last <ROOT> token via SHIFT
                    cano_actions = root_token_actions
            else:
                # just reached the last <ROOT> token via all the other actions
                cano_actions = root_token_actions
        else:
            # not the first token, not the last root token
            # the token sequence length is at least 3 here, and 0 < self.tok_cursor < self.tokseq_len - 1
            prev_action = self.canonical_action_form(past_actions[-1])
            if prev_action in gen_node_actions + ['DEPENDENT']:
                cano_actions = post_node_actions
            elif prev_action in gen_arc_actions:
                cano_actions = post_arc_actions
            elif prev_action in ['MERGE']:
                cano_actions = post_merge_actions
            elif prev_action in ['REDUCE', 'SHIFT']:
                cano_actions = pre_node_actions
            else:
                raise ValueError('unallowed previous action sequence.')

        # modify for special cases for MERGE
        if self.tok_cursor + 1 == self.tokseq_len - 1:
            # next token is the '<ROOT>' token
            if 'MERGE' in cano_actions:
                # NOTE the cano_actions list should not have duplicated entries
                cano_actions.remove('MERGE')

        # modify for arc actions based on the number of previous generated nodes
        num_prev_nodes = sum(actions_nodemask)
        # if num_prev_nodes < 2:
        #     # for LA and RA there must have been at least 2 nodes generated (current one included)
        #     # NOTE comment this for AMR1.0 data oracle, as there is a special of self-loop at the first arc
        #     cano_actions = list(filter(lambda x: x not in gen_arc_actions, cano_actions))
        if num_prev_nodes < 1:
            # for LA(root) there must have been at least 1 node generated (root node is by default there)
            if 'LA(root)' in cano_actions:
                # NOTE the cano_actions list should not have duplicated entries
                cano_actions.remove('LA(root)')
                # cano_actions = list(filter(lambda x: x != 'LA(root)', cano_actions))

        if inside_entity:
            for act in gen_node_actions:
                if act not in cano_actions:
                    cano_actions = cano_actions + [act]

        return cano_actions

    def apply_canonical_action(self, action, arc_pos=None):
        assert self.canonical_mode
        assert action in self.canonical_actions

        # check ending
        if self.is_postprocessed:
            assert self.is_closed, '"is_closed" flag must be raised before "is_postprocessed" flag'
            print('AMR state machine: completed --- no more actions can be applied.')
            return
        else:
            if self.is_closed:
                assert action == 'CLOSE', 'AMR state machine: token sequence finished --- only CLOSE action ' \
                                          'can be applied for AMR postprocessing'

        self.actions_tokcursor.append(self.tok_cursor)

        # apply action: only move token cursor, and record the executed action
        if action in ['SHIFT', 'REDUCE', 'MERGE']:
            self._shift()
            self.actions_nodemask.append(0)
        elif action in ['PRED', 'COPY_LEMMA', 'COPY_SENSE01', 'ENTITY']:
            self.actions_nodemask.append(1)
            self.actions_latest_node = len(self.actions_nodemask) - 1
        elif action in ['DEPENDENT']:
            self.actions_nodemask.append(0)    # TODO arc to dependent node is disallowed now. discuss
        elif action in ['LA', 'RA', 'LA(root)']:
            self.actions_nodemask.append(0)
        elif action == 'CLOSE':
            self._close()
            self.is_postprocessed = True    # do nothing for postprocessing in canonical mode
            self.actions_nodemask.append(0)
        else:
            raise Exception(f'Unrecognized canonical action: {action}')

        self.actions_canonical.append(action)

        # graph structure: edge information
        if action in ['LA', 'RA']:
            self.actions_edge_mask.append(1)
            self.actions_edge_cur_node.append(self.actions_latest_node)
            self.actions_edge_pre_node.append(arc_pos)
            if action == 'RA':
                self.actions_edge_direction.append(1)
            else:
                self.actions_edge_direction.append(-1)
        elif action == 'LA(root)':
            self.actions_edge_mask.append(1)
            self.actions_edge_cur_node.append(-2)    # NOTE root node is not added by any action
            self.actions_edge_pre_node.append(arc_pos)
            self.actions_edge_direction.append(-1)
        else:
            self.actions_edge_mask.append(0)
            self.actions_edge_cur_node.append(-1)
            self.actions_edge_pre_node.append(-1)
            self.actions_edge_direction.append(0)

        # Increase time step
        self.time_step += 1

        return

    def apply_action(self, action, **kwargs):
        assert not self.canonical_mode, 'Should not be in the canonical mode to apply detailed actions with labels'

        # read in action and properties
        action_label, properties = self.read_action(action)

        # check ending
        if self.is_postprocessed:
            assert self.is_closed, '"is_closed" flag must be raised before "is_postprocessed" flag'
            print('AMR state machine: completed --- no more actions can be applied.')
            return
        else:
            if self.is_closed:
                assert action_label == 'CLOSE', 'AMR state machine: token sequence finished --- only CLOSE action ' \
                                                'can be applied for AMR postprocessing'

        self.actions_tokcursor.append(self.tok_cursor)

        # apply action
        if action_label == 'SHIFT':
            self.SHIFT(properties[0] if properties else None)
        elif action_label == 'REDUCE':
            self.REDUCE()
        # the flowing 3 actions are for node generation
        elif action_label == 'PRED':
            assert len(properties) == 1
            self.PRED(properties[0])
        elif action_label == 'COPY_LEMMA':
            self.COPY_LEMMA()
        elif action_label in ['COPY_SENSE01']:
            self.COPY_SENSE01()
        # for multiple alignments and other cases
        elif action_label.startswith('DEPENDENT'):
            self.DEPENDENT(*properties)
        elif action_label in ['ADDNODE', 'ENTITY']:    # TODO 'ADDNODE' currently not used
            # preprocessing
            self.ENTITY(",".join(properties))
        elif action_label in ['MERGE']:
            self.MERGE()
        # add an arc
        elif action_label == 'LA':
            self.LA(*properties)
        elif action_label == 'RA':
            self.RA(*properties)
        # close and postprocessing
        elif action_label == 'CLOSE':
            self.CLOSE(**kwargs)
        else:
            raise Exception(f'Unrecognized action: {action}')

        # Increase time step
        self.time_step += 1

        return

    def apply_actions(self, actions, **kwargs):
        # no special extra actions such as CLOSE, thus `apply_actions` can be applied multiple times sequentially
        for action in actions:
            self.apply_action(action, **kwargs)

    def _close(self):
        if not self.is_closed:
            self.is_closed = True
        return

    def _shift(self):
        if self.tok_cursor == self.tokseq_len - 1:
            # the only condition to close the machine: token cursor at last token & shift is called
            self._close()
            return
        if not self.is_closed:
            self.tok_cursor += 1
            if self.tok_cursor == self.tokseq_len - 1:
                self.current_node_id = self.root_id
            else:
                self.current_node_id = None
        return

    def _postprocessing(self, training=False, gold_amr=None):
        # TODO this part of postprocessing code untouched and unorganized; minimally modified previous code
        if self.is_postprocessed:
            return
        if self.amr_graph:
            if training:
                self.postprocessing_training(gold_amr)
            else:
                self.postprocessing(gold_amr)
            self.clean_amr()
            # do not do multiple close, cuz of this
            self.convert_state_machine_alignments_to_amr_alignments()
            self.connect_graph()
        self.is_postprocessed = True
        return

    def REDUCE(self):
        """REDUCE : delete token when there is no alignment"""
        self._shift()    # shift to next position in the token sequence

        # record action info
        self.actions.append('REDUCE')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(None)
        return

    def SHIFT(self, shift_label=None):
        """SHIFT : Move the current pointer to the next word"""
        self._shift()

        # record action info
        if shift_label is not None:
            self.actions.append(f'SHIFT({shift_label})')
        else:
            self.actions.append(f'SHIFT')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(None)
        return

    # TODO **kwargs to be compatible compared with previous code taking other arguments
    # TODO modify postprocessing inside close
    def CLOSE(self, training=False, gold_amr=None, **kwargs):
        """CLOSE : finish parsing + postprocessing"""
        self._close()
        self._postprocessing(training, gold_amr)

        # record action info
        self.actions.append('CLOSE')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(None)
        return

    def PRED(self, node_label):
        """PRED : assign a propbank label"""
        node_id = self.new_node_id
        self.new_node_id += 1

        if self.amr_graph:
            self.amr.nodes[node_id] = node_label
        self.current_node_id = node_id
        self.tokid_to_nodeid[self.tok_cursor].append(node_id)

        # update machine state

        # keep node to token alignment
        self.alignments[node_id] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)

        # record action info
        self.actions.append(f'PRED({node_label})')
        self.actions_to_nodes.append(node_id)
        self.actions_to_nlabels.append(node_label)
        self.actions_to_elabels.append(None)

        return

    def COPY_LEMMA(self):
        """COPY_LEMMA: same as PRED but use lowercased lemma"""
        node_id = self.new_node_id
        self.new_node_id += 1

        node_label = self.get_current_token(lemma=True)
        if self.amr_graph:
            self.amr.nodes[node_id] = node_label
        self.current_node_id = node_id
        self.tokid_to_nodeid[self.tok_cursor].append(node_id)

        # update machine state

        # keep node to token alignment
        self.alignments[node_id] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)

        # record action info
        self.actions.append('COPY_LEMMA')
        self.actions_to_nodes.append(node_id)
        self.actions_to_nlabels.append(node_label)
        self.actions_to_elabels.append(None)

        return

    def COPY_SENSE01(self):
        """COPY_SENSE01: same as PRED but use lowercased lemma + '-01'"""
        node_id = self.new_node_id
        self.new_node_id += 1

        node_label = self.get_current_token(lemma=True)
        node_label = node_label + '-01'
        if self.amr_graph:
            self.amr.nodes[node_id] = node_label
        self.current_node_id = node_id
        self.tokid_to_nodeid[self.tok_cursor].append(node_id)

        # update machine state

        # keep node to token alignment
        self.alignments[node_id] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)

        # record action info
        self.actions.append('COPY_SENSE01')
        self.actions_to_nodes.append(node_id)
        self.actions_to_nlabels.append(node_label)
        self.actions_to_elabels.append(None)

        return

    def ENTITY(self, entity_type):
        """ENTITY : create a named entity"""
        # get the surface tokens
        surface_tokens = ','.join(self.tokens[x].replace(',', '-COMMA-') for x in self.merged_tokens[self.tok_cursor]) \
            if self.tok_cursor in self.merged_tokens \
            else self.tokens[self.tok_cursor]

        head_id = self.new_node_id
        self.new_node_id += 1

        if self.entities_with_preds and entity_type in self.entities_with_preds:
            if self.amr_graph:
                self.amr.nodes[head_id] = entity_type
            self.entity_tokens[head_id] = surface_tokens
            self.entity_tokenids.append(self.tok_cursor)    # could have duplicates -> indicating how many ENTITY on it
            self.current_node_id = head_id
            self.tokid_to_nodeid[self.tok_cursor].append(head_id)
            self.alignments[head_id] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)
            self.actions.append(f'ENTITY({entity_type})')
            self.actions_to_nodes.append(head_id)
            self.actions_to_nlabels.append(entity_type)
            return

        child_id = self.new_node_id
        self.new_node_id += 1

        if self.amr_graph:
            # self.amr.nodes[child_id] = self.amr.nodes[head]
            self.amr.nodes[child_id] = surface_tokens
            self.amr.nodes[head_id] = entity_type
            self.amr.edges.append((head_id, 'entity', child_id))
        self.entities.append(head_id)

        self.entity_tokens[head_id] = surface_tokens
        self.entity_tokenids.append(self.tok_cursor)    # could have duplicates -> indicating how many ENTITY on it

        # child id (the surface token nodes) is completely hidden -> left for postprocessing
        self.current_node_id = head_id
        self.tokid_to_nodeid[self.tok_cursor].append(head_id)

        # update machine state

        # keep node to token alignment
        self.alignments[head_id] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)

        # record action info
        self.actions.append(f'ENTITY({entity_type})')
        self.actions_to_nodes.append(head_id)
        self.actions_to_nlabels.append(entity_type)
        self.actions_to_elabels.append('entity')

        return

    def DEPENDENT(self, node_label, edge_label, node_id=None):
        """DEPENDENT : add a single edge and node"""
        edge_label = edge_label if edge_label.startswith(':') else ':' + edge_label
        if self.amr_graph:
            if node_id is not None:
                # existing node id
                assert node_id in self.amr.nodes, '"node_id", when provided, should already exist'
            else:
                # add a new node
                node_id = self.new_node_id
                self.new_node_id += 1
                self.amr.nodes[node_id] = node_label
            self.amr.edges.append((self.current_node_id, edge_label, node_id))

        # NOTE do not update self.current_node_id here

        # record action info
        self.actions.append(f'DEPENDENT({node_label},{edge_label.replace(":","")})')
        # TODO or use this action to link to node and remove previous link (consider)
        self.actions_to_nodes.append(None)
        # TODO in previous code, below were all set to None
        self.actions_to_nlabels.append(node_label)
        self.actions_to_elabels.append(edge_label)

        return

    def MERGE(self):
        """MERGE : merge two tokens to be the same node"""
        # merge the current token and the next one
        assert self.tok_cursor < self.tokseq_len - 1, 'no next token to merge'
        lead = self.tok_cursor + 1
        sec = self.tok_cursor

        # maintain merged tokens dict
        if lead not in self.merged_tokens:
            self.merged_tokens[lead] = [lead]
        if sec in self.merged_tokens:
            self.merged_tokens[lead] = self.merged_tokens[sec] + self.merged_tokens[lead]
        else:
            self.merged_tokens[lead].insert(0, sec)
        merged = ','.join(self.tokens[x].replace(',', '-COMMA-') for x in self.merged_tokens[lead])

        # change the token at next position to be merged for PRED?
        # NOTE currently MERGE only works for ENTITY

        # move the cursor to the next token position
        self._shift()

        # record action info
        self.actions.append('MERGE')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(None)
        return

    def LA(self, pos, label):
        """LA : add an arc from current node to a previous node (linked with a previous action)"""
        edge_label = label if label.startswith(':') else (':' + label if label != 'root' else 'root')
        if self.amr_graph:
            if edge_label == 'root':
                assert self.current_node_id == self.root_id
            self.amr.edges.append((self.current_node_id, edge_label, self.actions_to_nodes[pos]))

        # record action info
        self.actions.append(f'LA({pos},{edge_label})')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(edge_label)
        return

    def RA(self, pos, label):
        """RA : add an arc from a previous node (linked with a previous action) to the current node"""
        edge_label = label if label.startswith(':') else (':' + label if label != 'root' else 'root')
        if self.amr_graph:
            if edge_label == 'root':
                # note: in principle, '<ROOT>' token can be at any position
                assert self.current_node_id == self.root_id
            self.amr.edges.append((self.actions_to_nodes[pos], edge_label, self.current_node_id))

        # record action info
        self.actions.append(f'RA({pos},{edge_label})')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(edge_label)
        return

    def postprocessing_training(self, gold_amr):

        #         import pdb; pdb.set_trace()

        for entity_id in self.entities:

            entity_edges = [e for e in self.amr.edges if e[0] == entity_id and e[1] == 'entity']

            for e in entity_edges:
                self.amr.edges.remove(e)

            child_id = [t for s, r, t in entity_edges][0]
            del self.amr.nodes[child_id]

            new_node_ids = []

            entity_alignment = gold_amr.alignmentsToken2Node(entity_id + 1)    # TODO here need to +1 for id
            gold_entity_subgraph = gold_amr.findSubGraph(entity_alignment)

            for i, n in enumerate(entity_alignment):
                if i == 0:
                    self.amr.nodes[entity_id] = gold_amr.nodes[n]
                    new_node_ids.append(entity_id)
                else:
                    self.amr.nodes[self.new_node_id] = gold_amr.nodes[n]
                    new_node_ids.append(self.new_node_id)
                    self.new_node_id += 1

            for s, r, t in gold_entity_subgraph.edges:
                new_s = new_node_ids[entity_alignment.index(s)]
                new_t = new_node_ids[entity_alignment.index(t)]
                self.amr.edges.append((new_s, r, new_t))

    def postprocessing(self, gold_amr):
        global entity_rules_json, entity_rule_stats, entity_rule_totals, entity_rule_fails

        if not entity_rules_json:
            with open(entities_path, 'r', encoding='utf8') as f:
                entity_rules_json = json.load(f)

        for entity_id in self.entities:

            if entity_id not in self.amr.nodes:
                continue
            # Test postprocessing ----------------------------
            gold_concepts = []
            if gold_amr:
                entity_alignment = gold_amr.alignmentsToken2Node(entity_id + 1)
                gold_entity_subgraph = gold_amr.findSubGraph(entity_alignment)
                for n in gold_entity_subgraph.nodes:
                    node = gold_entity_subgraph.nodes[n]
                    if n == gold_entity_subgraph.root:
                        gold_concepts.append(node)
                    for s, r, t in gold_entity_subgraph.edges:
                        if t == n:
                            edge = r
                            gold_concepts.append(edge + ' ' + node)
            # -------------------------------------------

            new_concepts = []

            entity_type = self.amr.nodes[entity_id]
            if self.entities_with_preds and entity_type in self.entities_with_preds:
                continue

            model_entity_alignments = None
            if entity_id in self.alignments:
                model_entity_alignments = self.alignments[entity_id]
            if entity_type.startswith('('):
                entity_type = entity_type[1:-1]
            entity_edges = [e for e in self.amr.edges if e[0] == entity_id and e[1] == 'entity']
            if not entity_edges:
                continue

            child_id = [t for s, r, t in entity_edges][0]
            entity_tokens = self.amr.nodes[child_id].split(',')

            for e in entity_edges:
                self.amr.edges.remove(e)
            del self.amr.nodes[child_id]

            # date-entity special rules
            if entity_type == 'date-entity':
                date_entity_rules = entity_rules_json['date-entity']
                assigned_edges = ['' for _ in entity_tokens]
                if len(entity_tokens) == 1:
                    date = entity_tokens[0]
                    if date.isdigit() and len(date) == 8:
                        # format yyyymmdd
                        entity_tokens = [date[:4], date[4:6], date[6:]]
                        assigned_edges = [':year', ':month', ':day']
                    elif date.isdigit() and len(date) == 6:
                        # format yymmdd
                        entity_tokens = [date[:2], date[2:4], date[4:]]
                        assigned_edges = [':year', ':month', ':day']
                    elif '/' in date and date.replace('/', '').isdigit():
                        # format mm-dd-yyyy
                        entity_tokens = date.split('/')
                        assigned_edges = ['' for _ in entity_tokens]
                    elif '-' in date and date.replace('-', '').isdigit():
                        # format mm-dd-yyyy
                        entity_tokens = date.split('-')
                        assigned_edges = ['' for _ in entity_tokens]
                    elif date.lower() == 'tonight':
                        entity_tokens = ['night', 'today']
                        assigned_edges = [':dayperiod', ':mod']
                    elif date[0].isdigit() and (date.endswith('BC') or date.endswith('AD') or date.endswith('BCE') or date.endswith('CE')):
                        # 10,000BC
                        idx = 0
                        for i in range(len(date)):
                            if date[i].isalpha():
                                idx = i
                        entity_tokens = [date[:idx], date[idx:]]
                        assigned_edges = [':year', ':era']
                for j, tok in enumerate(entity_tokens):
                    if assigned_edges[j]:
                        continue
                    if tok.lower() in date_entity_rules[':weekday']:
                        assigned_edges[j] = ':weekday'
                        continue
                    if tok in date_entity_rules[':timezone']:
                        assigned_edges[j] = ':timezone'
                        continue
                    if tok.lower() in date_entity_rules[':calendar']:
                        assigned_edges[j] = ':calendar'
                        if tok.lower() == 'lunar':
                            entity_tokens[j] = 'moon'
                        continue
                    if tok.lower() in date_entity_rules[':dayperiod']:
                        assigned_edges[j] = ':dayperiod'
                        for idx, tok in enumerate(entity_tokens):
                            if tok.lower() == 'this':
                                entity_tokens[idx] = 'today'
                            elif tok.lower() == 'last':
                                entity_tokens[idx] = 'yesterday'
                        idx = j - 1
                        if idx >= 0 and entity_tokens[idx].lower() == 'one':
                            assigned_edges[idx] = ':quant'
                        continue
                    if tok in date_entity_rules[':era'] or tok.lower() in date_entity_rules[':era'] \
                            or ('"' in tok and tok.replace('"', '') in date_entity_rules[':era']):
                        assigned_edges[j] = ':era'
                        continue
                    if tok.lower() in date_entity_rules[':season']:
                        assigned_edges[j] = ':season'
                        continue

                    months = entity_rules_json['normalize']['months']
                    if tok.lower() in months or len(tok.lower()) == 4 and tok.lower().endswith(
                            '.') and tok.lower()[:3] in months:
                        if ':month' in assigned_edges:
                            idx = assigned_edges.index(':month')
                            if entity_tokens[idx].isdigit():
                                assigned_edges[idx] = ':day'
                        assigned_edges[j] = ':month'
                        continue
                    ntok = self.normalize_token(tok)
                    if ntok.isdigit():
                        if j + 1 < len(entity_tokens) and entity_tokens[j + 1].lower() == 'century':
                            assigned_edges[j] = ':century'
                            continue
                        if 1 <= int(ntok) <= 12 and ':month' not in assigned_edges:
                            if not (tok.endswith('th') or tok.endswith('st')
                                    or tok.endswith('nd') or tok.endswith('nd')):
                                assigned_edges[j] = ':month'
                                continue
                        if 1 <= int(ntok) <= 31 and ':day' not in assigned_edges:
                            assigned_edges[j] = ':day'
                            continue
                        if 1 <= int(ntok) <= 10001 and ':year' not in assigned_edges:
                            assigned_edges[j] = ':year'
                            continue
                    if tok.startswith("'") and len(tok) == 3 and tok[1:].isdigit():
                        # 'yy
                        assigned_edges[j] = ':year'
                        entity_tokens[j] = tok[1:]
                        continue
                    decades = entity_rules_json['normalize']['decades']
                    if tok.lower() in decades:
                        assigned_edges[j] = ':decade'
                        entity_tokens[j] = str(decades[tok.lower()])
                        continue
                    if tok.endswith('s') and len(tok) > 2 and tok[:2].isdigit():
                        assigned_edges[j] = ':decade'
                        entity_tokens[j] = tok[:-1]
                        continue
                    assigned_edges[j] = ':mod'

                self.amr.nodes[entity_id] = 'date-entity'
                new_concepts.append('date-entity')
                for tok, rel in zip(entity_tokens, assigned_edges):
                    if tok.lower() in ['-comma-', 'of', 'the', 'in', 'at',
                                       'on', 'century', '-', '/', '', '(', ')', '"']:
                        continue
                    tok = tok.replace('"', '')
                    if rel in [':year', ':decade']:
                        year = tok
                        if len(year) == 2:
                            tok = '20' + year if (0 <= int(year) <= 30) else '19' + year
                    if rel in [':month', ':day'] and tok.isdigit() and int(tok) == 0:
                        continue
                    if tok.isdigit():
                        while tok.startswith('0') and len(tok) > 1:
                            tok = tok[1:]
                    if rel in [':day', ':month', ':year', ':era', ':calendar', ':century', ':quant', ':timezone']:
                        self.amr.nodes[self.new_node_id] = self.normalize_token(tok)
                    else:
                        self.amr.nodes[self.new_node_id] = tok.lower()
                    self.amr.edges.append((entity_id, rel, self.new_node_id))
                    self.alignments[self.new_node_id] = model_entity_alignments
                    new_concepts.append(rel + ' ' + self.amr.nodes[self.new_node_id])
                    self.new_node_id += 1
                if gold_amr and set(gold_concepts) == set(new_concepts):
                    entity_rule_stats['date-entity'] += 1
                entity_rule_totals['date-entity'] += 1
                continue

            rule = entity_type + '\t' + ','.join(entity_tokens).lower()
            # check if singular is in fixed rules
            if rule not in entity_rules_json['fixed'] and len(entity_tokens) == 1 and entity_tokens[0].endswith('s'):
                rule = entity_type + '\t' + entity_tokens[0][:-1]

            # fixed rules
            if rule in entity_rules_json['fixed']:
                edges = entity_rules_json['fixed'][rule]['edges']
                nodes = entity_rules_json['fixed'][rule]['nodes']
                root = entity_rules_json['fixed'][rule]['root']
                id_map = {}
                for j, n in enumerate(nodes):
                    node_label = nodes[n]
                    n = int(n)

                    id_map[n] = entity_id if n == root else self.new_node_id
                    self.new_node_id += 1
                    self.amr.nodes[id_map[n]] = node_label
                    self.alignments[id_map[n]] = model_entity_alignments
                    new_concepts.append(node_label)
                for s, r, t in edges:
                    self.amr.edges.append((id_map[s], r, id_map[t]))
                    concept = self.amr.nodes[id_map[t]]
                    if concept in new_concepts:
                        idx = new_concepts.index(concept)
                        new_concepts[idx] = r + ' ' + new_concepts[idx]
                if gold_amr and set(gold_concepts) == set(new_concepts):
                    entity_rule_stats['fixed'] += 1
                else:
                    entity_rule_fails[entity_type] += 1
                entity_rule_totals['fixed'] += 1
                continue

            rule = entity_type + '\t' + str(len(entity_tokens))

            # variable rules
            if rule in entity_rules_json['var']:
                edges = entity_rules_json['var'][rule]['edges']
                nodes = entity_rules_json['var'][rule]['nodes']
                root = entity_rules_json['var'][rule]['root']
                node_map = {}
                ntok = None
                for i, tok in enumerate(entity_tokens):
                    ntok = self.normalize_token(tok)
                    node_map[f'X{i}'] = ntok if not ntok.startswith('"') else tok.lower()
                id_map = {}
                for j, n in enumerate(nodes):
                    node_label = nodes[n]
                    n = int(n)

                    id_map[n] = entity_id if n == root else self.new_node_id
                    self.new_node_id += 1
                    self.amr.nodes[id_map[n]] = node_map[node_label] if node_label in node_map else node_label
                    self.alignments[id_map[n]] = model_entity_alignments
                    new_concepts.append(self.amr.nodes[id_map[n]])
                for s, r, t in edges:
                    node_label = self.amr.nodes[id_map[t]]
                    if 'date-entity' not in entity_type and (node_label.isdigit()
                                                             or node_label in
                                                             ['many', 'few', 'some', 'multiple', 'none']):
                        r = ':quant'
                    self.amr.edges.append((id_map[s], r, id_map[t]))
                    concept = self.amr.nodes[id_map[t]]
                    if concept in new_concepts:
                        idx = new_concepts.index(concept)
                        new_concepts[idx] = r + ' ' + new_concepts[idx]
                if gold_amr and set(gold_concepts) == set(new_concepts):
                    entity_rule_stats['var'] += 1
                else:
                    entity_rule_fails[entity_type] += 1
                entity_rule_totals['var'] += 1
                continue

            rule = entity_type

            # named entities rules
            if entity_type.endswith(',name') or entity_type == 'name':
                name_id = None
                if rule in entity_rules_json['names']:
                    edges = entity_rules_json['names'][rule]['edges']
                    nodes = entity_rules_json['names'][rule]['nodes']
                    root = entity_rules_json['names'][rule]['root']
                    id_map = {}
                    for j, n in enumerate(nodes):
                        node_label = nodes[n]
                        n = int(n)

                        id_map[n] = entity_id if n == root else self.new_node_id
                        if node_label == 'name':
                            name_id = id_map[n]
                        self.new_node_id += 1
                        self.amr.nodes[id_map[n]] = node_label
                        self.alignments[id_map[n]] = model_entity_alignments
                        new_concepts.append(node_label)
                    for s, r, t in edges:
                        self.amr.edges.append((id_map[s], r, id_map[t]))
                        concept = self.amr.nodes[id_map[t]]
                        if concept in new_concepts:
                            idx = new_concepts.index(concept)
                            new_concepts[idx] = r + ' ' + new_concepts[idx]
                else:
                    nodes = entity_type.split(',')
                    nodes.remove('name')
                    name_id = entity_id if len(nodes) == 0 else self.new_node_id
                    self.amr.nodes[name_id] = 'name'
                    self.alignments[name_id] = model_entity_alignments
                    self.new_node_id += 1
                    if len(nodes) == 0:
                        new_concepts.append('name')
                    for j, node in enumerate(nodes):
                        new_id = entity_id if j == 0 else self.new_node_id
                        self.amr.nodes[new_id] = node
                        self.alignments[new_id] = model_entity_alignments
                        if j == 0:
                            new_concepts.append(node)
                        self.new_node_id += 1
                        if j == len(nodes) - 1:
                            rel = ':name'
                            self.amr.edges.append((new_id, rel, name_id))
                            new_concepts.append(':name ' + 'name')
                        else:
                            rel = default_rel
                            self.amr.edges.append((new_id, rel, self.new_node_id))
                            new_concepts.append(default_rel + ' ' + self.amr.nodes[new_id])

                op_idx = 1
                for tok in entity_tokens:
                    tok = tok.replace('"', '')
                    if tok in ['(', ')', '']:
                        continue
                    new_tok = '"' + tok[0].upper() + tok[1:] + '"'
                    self.amr.nodes[self.new_node_id] = new_tok
                    self.alignments[self.new_node_id] = model_entity_alignments
                    rel = f':op{op_idx}'
                    self.amr.edges.append((name_id, rel, self.new_node_id))
                    new_concepts.append(rel + ' ' + new_tok)
                    self.new_node_id += 1
                    op_idx += 1
                if gold_amr and set(gold_concepts) == set(new_concepts):
                    entity_rule_stats['names'] += 1
                entity_rule_totals['names'] += 1
                continue

            # unknown entity types
            nodes = entity_type.split(',')
            idx = 0
            prev_id = None
            for node in nodes:
                if node in ['(', ')', '"', '']:
                    continue
                new_id = entity_id if idx == 0 else self.new_node_id
                self.amr.nodes[new_id] = node
                self.alignments[new_id] = model_entity_alignments
                self.new_node_id += 1
                if idx > 0:
                    self.amr.edges.append((prev_id, default_rel, new_id))
                    new_concepts.append(default_rel + ' ' + node)
                else:
                    new_concepts.append(node)
                prev_id = new_id
            for tok in entity_tokens:
                tok = tok.replace('"', '')
                if tok in ['(', ')', '']:
                    continue
                self.amr.nodes[self.new_node_id] = tok.lower()
                self.alignments[new_id] = model_entity_alignments
                self.amr.edges.append((prev_id, default_rel, self.new_node_id))
                new_concepts.append(default_rel + ' ' + tok.lower())
                self.new_node_id += 1
            if gold_amr and set(gold_concepts) == set(new_concepts):
                entity_rule_stats['unknown'] += 1
            else:
                entity_rule_fails[entity_type] += 1
            entity_rule_totals['unknown'] += 1

    def normalize_token(self, string):
        global entity_rules_json

        if not entity_rules_json:
            with open(entities_path, 'r', encoding='utf8') as f:
                entity_rules_json = json.load(f)

        lstring = string.lower()
        months = entity_rules_json['normalize']['months']
        units = entity_rules_json['normalize']['units']
        cardinals = entity_rules_json['normalize']['cardinals']
        ordinals = entity_rules_json['normalize']['ordinals']

        # number or ordinal
        if NUM_RE.match(lstring):
            return lstring.replace(',', '').replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')

        # months
        if lstring in months:
            return str(months[lstring])
        if len(lstring) == 4 and lstring.endswith('.') and lstring[:3] in months:
            return str(months[lstring[:3]])

        # cardinal numbers
        if lstring in cardinals:
            return str(cardinals[lstring])

        # ordinal numbers
        if lstring in ordinals:
            return str(ordinals[lstring])

        # unit abbreviations
        if lstring in units:
            return str(units[lstring])
        if lstring.endswith('s') and lstring[:-1] in units:
            return str(units[lstring[:-1]])
        if lstring in units.values():
            return lstring
        if string.endswith('s') and lstring[:-1] in units.values():
            return lstring[:-1]

        return string
        #return '"' + string + '"'

    def clean_amr(self):
        if self.amr_graph:
            # clean concepts
            for n in self.amr.nodes:
                if self.amr.nodes[n] in ['.', '?', '!', ',', ';', '"', "'"]:
                    self.amr.nodes[n] = 'PUNCT'
                if self.amr.nodes[n].startswith('"') and self.amr.nodes[n].endswith('"'):
                    self.amr.nodes[n] = '"' + self.amr.nodes[n].replace('"', '') + '"'
                if not (self.amr.nodes[n].startswith('"') and self.amr.nodes[n].endswith('"')):
                    for ch in ['/', ':', '(', ')', '\\']:
                        if ch in self.amr.nodes[n]:
                            self.amr.nodes[n] = self.amr.nodes[n].replace(ch, '-')
                if not self.amr.nodes[n]:
                    self.amr.nodes[n] = 'None'
                if ',' in self.amr.nodes[n]:
                    self.amr.nodes[n] = '"' + self.amr.nodes[n].replace('"', '') + '"'
                if not self.amr.nodes[n][0].isalpha() and not self.amr.nodes[n][0].isdigit(
                ) and not self.amr.nodes[n][0] in ['-', '+']:
                    self.amr.nodes[n] = '"' + self.amr.nodes[n].replace('"', '') + '"'
            # clean edges
            for j, e in enumerate(self.amr.edges):
                s, r, t = e
                if not r.startswith(':'):
                    r = ':' + r
                e = (s, r, t)
                self.amr.edges[j] = e
            # handle missing nodes (this shouldn't happen but a bad sequence of actions can produce it)
            for s, r, t in self.amr.edges:
                if s not in self.amr.nodes:
                    self.amr.nodes[s] = 'NA'
                if t not in self.amr.nodes:
                    self.amr.nodes[t] = 'NA'

    def convert_state_machine_alignments_to_amr_alignments(self):
        # In the state machine, we get the alignments with index 0
        # However, in the AMR, alignments are stored with index 1, since that is the way the oracle expects it

        for node in self.alignments:
            if type(self.alignments[node]) == int:
                self.amr.alignments[node] = self.alignments[node] + 1
            else:
                new_list = list()
                for alignment in self.alignments[node]:
                    assert type(alignment) == int
                    new_list.append(alignment + 1)
                self.amr.alignments[node] = deepcopy(new_list)

    def connect_graph(self):
        assigned_root = None
        root_edges = []
        if -1 in self.amr.nodes:
            del self.amr.nodes[-1]
        for s, r, t in self.amr.edges:
            if s == -1 and r == "root":
                assigned_root = t
            if s == -1 or t == -1:
                root_edges.append((s, r, t))
        for e in root_edges:
            self.amr.edges.remove(e)

        if not self.amr.nodes:
            return

        descendents = {n: {n} for n in self.amr.nodes}
        potential_roots = [n for n in self.amr.nodes]
        for x, r, y in self.amr.edges:
            if y in potential_roots and x not in descendents[y]:
                potential_roots.remove(y)
            descendents[x].update(descendents[y])
            for n in descendents:
                if x in descendents[n]:
                    descendents[n].update(descendents[x])

        disconnected = potential_roots.copy()
        for n in potential_roots.copy():
            if len([e for e in self.amr.edges if e[0] == n]) == 0:
                potential_roots.remove(n)

        # assign root
        if potential_roots:
            self.amr.root = potential_roots[0]
            for n in potential_roots:
                if self.amr.nodes[n] == 'multi-sentence' or n == assigned_root:
                    self.amr.root = n
            disconnected.remove(self.amr.root)
        else:
            self.amr.root = max(self.amr.nodes.keys(),
                                key=lambda x: len([e for e in self.amr.edges if e[0] == x])
                                - len([e for e in self.amr.edges if e[2] == x]))
        # connect graph
        if len(disconnected) > 0:
            for n in disconnected:
                self.amr.edges.append((self.amr.root, default_rel, n))
