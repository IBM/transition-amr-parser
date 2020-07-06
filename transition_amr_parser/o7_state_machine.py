import json
import os
import re
from collections import Counter
from copy import deepcopy

import spacy
from spacy.tokens.doc import Doc

from transition_amr_parser.amr import AMR

"""
AMRStateMachine applies operations in a transition-based AMR parser, but combined with a pointer for arcs.
It maintains a cursor on the token sequence and moves from left to right and apply actions to generate an AMR graph.

Actions are
    SHIFT : move cursor to next position in the token sequence
    REDUCE : delete current token
    MERGE : merge two tokens (for MWEs)
    ENTITY(type) : form a named entity
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


def get_spacy_lemmatizer():
    # TODO: Unclear why this configuration
    # from spacy.cli.download import download
    try:
        lemmatizer = spacy.load('en', disable=['parser', 'ner'])
    except OSError:
        # Assume the problem was the spacy models were not downloaded
        from spacy.cli.download import download
        download('en')
        lemmatizer = spacy.load('en', disable=['parser', 'ner'])
    lemmatizer.tokenizer = NoTokenizer(lemmatizer.vocab)
    return lemmatizer


class AMRStateMachine:
    def __init__(self, tokens, verbose=False, add_unaligned=0,
                 actions_by_stack_rules=None, amr_graph=True,
                 spacy_lemmatizer=None):
        # TODO verbose not used, actions_by_stack_rules not used
        # word tokens of sentence
        self.tokens = tokens.copy()

        # build and store amr graph (needed e.g. for oracle)
        self.amr_graph = amr_graph

        # spacy lemmatizer
        self.spacy_lemmatizer = spacy_lemmatizer
        self.lemmas = None

        # add unaligned to the token sequence
        if add_unaligned and '<unaligned>' not in self.tokens:
            for i in range(add_unaligned):
                self.tokens.append('<unaligned>')
        # add root to the token sequence
        if '<ROOT>' not in self.tokens:
            self.tokens.append("<ROOT>")

        # machine is active
        self.time_step = 0
        self.is_closed = False    # when True, no action can be applied except CLOSE
        self.is_postprocessed = False    # when True, no action can be applied

        # init current processing position in the token sequence
        self.tok_cursor = 0
        self.tokseq_len = len(self.tokens)    # this includes the '<ROOT>', which is not treated specially

        # init amr
        self.tokid_to_nodeid = [0] * self.tokseq_len
        """
        Decouple node id and token id, in case node ids are not initialized with token ids (e.g. if root node id -1).
        Therefore, `self.tok_cursor` can be kept positive all the time when the cursor is moving without worrying
        about special node ids which may not be consecutive integers starting from 0.
        """
        self.root_id = -1    # or `self.tokseq_len - 1` for consistent positive values with other nodes
        # TODO 'root' should be tied with -1 <-- since -1 is a must for self.connectGraph() processing
        if self.amr_graph:
            self.amr = AMR(tokens=self.tokens)
            for i, tok in enumerate(self.tokens):
                if tok != "<ROOT>":
                    # note that the node id is NOT shifted by 1, compared with the AMR alignments
                    self.amr.nodes[i] = tok
                    self.tokid_to_nodeid[i] = i
                else:
                    self.amr.nodes[self.root_id] = tok
                    self.tokid_to_nodeid[i] = self.root_id
        self.nodeid_to_tokid = {n: i for i, n in enumerate(self.tokid_to_nodeid)}

        # initial node ids are the token indices, and new nodes are given ids counting from there
        self.new_id = len(self.tokens)

        # action sequence and parser AMR target output
        self.actions = []
        self.actions_to_nodes = []    # node ids generated by each action; None for no node by this action
        self.actions_to_nlabels = []
        self.actions_to_elabels = []
        self.alignments = {}

        # information for oracle
        self.is_confirmed = set()                  # node ids
        self.is_confirmed.add(self.root_id)
        self.merged_tokens = {}                    # keys are token ids of the last merged token
        self.entities = []                         # node ids

    @property
    def current_node_id(self):
        # TODO use this function to replace `self.tokid_to_nodeid[self.tok_cursor]` in this file
        return self.tokid_to_nodeid[self.tok_cursor]

    def tok_index_neg2pos(self, ind):
        """Negative index to positive value on the token sequence.

        This is for consistency when doing algorithmic calculation on the indexes.
        TODO this is currently not used
        """
        if ind >= 0:
            return ind
        else:
            return self.tokseq_len + ind

    def get_current_token(self, lemma=False):
        """Get the token at current cursor position."""
        if lemma:
            # Compute lemmas for this sentence and cache it
            if self.lemmas is None:
                assert self.spacy_lemmatizer, "No spacy_lemmatizer provided"
                self.lemmas = [
                    x.lemma_ for x in self.spacy_lemmatizer(self.tokens[:-1])
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
            # root arc is 'LA(pos,root)' if '<ROOT>' token at the end

            items = action.split('(')
            arc_name = items[0]
            arc_args = items[1][:-1].split(',')
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

            # To keep original name to keep learner happy
            if action_label == 'DEPENDENT':
                action_label = action

            return action_label, props

    def apply_action(self, action):
        # read in action and properties
        action_label, properties = self.read_action(action)

        # check ending
        if self.is_postprocessed:
            assert self.is_closed, '"is_closed" flag must be raised before "is_postprocessed" flag'
            print('AMR state machine: completed --- no more actions can be applied.')
            return
        else:
            if self.is_closed:
                assert action_label == 'CLOSE', 'AMR state machine: token sequence finished --- only CLOSE action' \
                                                'can be applied for AMR postprocessing'

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
            self.CLOSE()
        else:
            raise Exception(f'Unrecognized action: {action}')

        # Increase time step
        self.time_step += 1

        return

    def apply_actions(self, actions):
        # no special extra actions such as CLOSE, thus `apply_actions` can be applied multiple times sequentially
        for action in actions:
            self.apply_action(action)

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
        return

    def _postprocessing(self, training=False, gold_amr=None):
        # TODO this part of postprocessing code untouched and unorganized; minimally modified previous code
        if self.amr_graph:
            if training:
                self.postprocessing_training(gold_amr)
            else:
                self.postprocessing(gold_amr)
            self.clean_amr()
            # do not do multiple close, cuz of this
            self.convert_state_machine_alignments_to_amr_alignments()
            self.connectGraph()
        pass

    def REDUCE(self):
        """REDUCE : delete token when there is no alignment"""
        if self.amr_graph:
            del self.amr.nodes[self.tokid_to_nodeid[self.tok_cursor]]
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
        node_id = self.tokid_to_nodeid[self.tok_cursor]
        if self.amr_graph:
            self.amr.nodes[node_id] = node_label

        # update machine state
        # note: node ids default to token index at the beginning; names decoupled to make code clearer
        self.is_confirmed.add(node_id)
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
        node_id = self.tokid_to_nodeid[self.tok_cursor]
        node_label = self.get_current_token(lemma=True)
        if self.amr_graph:
            self.amr.nodes[node_id] = node_label

        # update machine state
        self.is_confirmed.add(node_id)
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
        node_id = self.tokid_to_nodeid[self.tok_cursor]
        node_label = self.get_current_token(lemma=True)
        node_label = node_label + '-01'
        if self.amr_graph:
            self.amr.nodes[node_id] = node_label

        # update machine state
        self.is_confirmed.add(node_id)
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
        head = self.tokid_to_nodeid[self.tok_cursor]
        child_id = self.new_id
        self.new_id += 1

        if self.amr_graph:
            self.amr.nodes[child_id] = self.amr.nodes[head]
            self.amr.nodes[head] = entity_type
            self.amr.edges.append((head, 'entity', child_id))
        self.entities.append(head)

        # update machine state
        self.is_confirmed.add(head)
        # keep node to token alignment
        self.alignments[head] = self.merged_tokens.get(self.tok_cursor, self.tok_cursor)

        # record action info
        self.actions.append(f'ENTITY({entity_type})')
        self.actions_to_nodes.append(head)
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
                node_id = self.new_id
                self.new_id += 1
                self.amr.nodes[node_id] = node_label
            self.amr.edges.append((self.tokid_to_nodeid[self.tok_cursor], edge_label, node_id))

        # record action info
        self.actions.append(f'DEPENDENT({node_label},{edge_label.replace(":","")})')
        # TODO or use this action to link to node and remove previous link (consider)
        self.actions_to_nodes.append(None)
        # TODO in previous code, below are all set to None
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

        # change the node dict
        if self.amr_graph:
            del self.amr.nodes[self.tokid_to_nodeid[sec]]
            self.amr.nodes[self.tokid_to_nodeid[lead]] = merged

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
                assert self.tokid_to_nodeid[self.tok_cursor] == self.root_id
            self.amr.edges.append((self.tokid_to_nodeid[self.tok_cursor], edge_label, self.actions_to_nodes[pos]))

        # record action info
        self.actions.append(f'LA({pos},{edge_label})')
        self.actions_to_nodes.append(None)
        self.actions_to_nlabels.append(None)
        self.actions_to_elabels.append(edge_label)
        return

    def RA(self, pos, label):
        """RA : add an arc from a previous node (linked with a previous action) to the current node"""
        edge_label = label if label.startswith(':') else (':'+label if label != 'root' else 'root')
        if self.amr_graph:
            if edge_label == 'root':
                # note: in principle, '<ROOT>' token can be at any position
                assert self.tokid_to_nodeid[self.tok_cursor] == self.root_id
            self.amr.edges.append((self.actions_to_nodes[pos], edge_label, self.tokid_to_nodeid[self.tok_cursor]))

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
                    self.amr.nodes[self.new_id] = gold_amr.nodes[n]
                    new_node_ids.append(self.new_id)
                    self.new_id += 1

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
                            gold_concepts.append(edge+' '+node)
            # -------------------------------------------

            new_concepts = []

            entity_type = self.amr.nodes[entity_id]
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
                        idx = j-1
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
                    if tok.lower() in months or len(tok.lower()) == 4 and tok.lower().endswith('.') and tok.lower()[:3] in months:
                        if ':month' in assigned_edges:
                            idx = assigned_edges.index(':month')
                            if entity_tokens[idx].isdigit():
                                assigned_edges[idx] = ':day'
                        assigned_edges[j] = ':month'
                        continue
                    ntok = self.normalize_token(tok)
                    if ntok.isdigit():
                        if j+1 < len(entity_tokens) and entity_tokens[j+1].lower() == 'century':
                            assigned_edges[j] = ':century'
                            continue
                        if 1 <= int(ntok) <= 12 and ':month' not in assigned_edges:
                            if not (tok.endswith('th') or tok.endswith('st') or tok.endswith('nd') or tok.endswith('nd')):
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
                    if tok.lower() in ['-comma-',  'of', 'the', 'in', 'at', 'on', 'century', '-', '/', '', '(', ')', '"']:
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
                        self.amr.nodes[self.new_id] = self.normalize_token(tok)
                    else:
                        self.amr.nodes[self.new_id] = tok.lower()
                    self.amr.edges.append((entity_id, rel, self.new_id))
                    self.alignments[self.new_id] = model_entity_alignments
                    new_concepts.append(rel+' '+self.amr.nodes[self.new_id])
                    self.new_id += 1
                if gold_amr and set(gold_concepts) == set(new_concepts):
                    entity_rule_stats['date-entity'] += 1
                entity_rule_totals['date-entity'] += 1
                continue

            rule = entity_type+'\t'+','.join(entity_tokens).lower()
            # check if singular is in fixed rules
            if rule not in entity_rules_json['fixed'] and len(entity_tokens) == 1 and entity_tokens[0].endswith('s'):
                rule = entity_type+'\t'+entity_tokens[0][:-1]

            # fixed rules
            if rule in entity_rules_json['fixed']:
                edges = entity_rules_json['fixed'][rule]['edges']
                nodes = entity_rules_json['fixed'][rule]['nodes']
                root = entity_rules_json['fixed'][rule]['root']
                id_map = {}
                for j, n in enumerate(nodes):
                    node_label = nodes[n]
                    n = int(n)

                    id_map[n] = entity_id if n == root else self.new_id
                    self.new_id += 1
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

            rule = entity_type+'\t'+str(len(entity_tokens))

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

                    id_map[n] = entity_id if n == root else self.new_id
                    self.new_id += 1
                    self.amr.nodes[id_map[n]] = node_map[node_label] if node_label in node_map else node_label
                    self.alignments[id_map[n]] = model_entity_alignments
                    new_concepts.append(self.amr.nodes[id_map[n]])
                for s, r, t in edges:
                    node_label = self.amr.nodes[id_map[t]]
                    if 'date-entity' not in entity_type and (node_label.isdigit() or node_label in ['many', 'few', 'some', 'multiple', 'none']):
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

                        id_map[n] = entity_id if n == root else self.new_id
                        if node_label == 'name':
                            name_id = id_map[n]
                        self.new_id += 1
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
                    name_id = entity_id if len(nodes) == 0 else self.new_id
                    self.amr.nodes[name_id] = 'name'
                    self.alignments[name_id] = model_entity_alignments
                    self.new_id += 1
                    if len(nodes) == 0:
                        new_concepts.append('name')
                    for j, node in enumerate(nodes):
                        new_id = entity_id if j == 0 else self.new_id
                        self.amr.nodes[new_id] = node
                        self.alignments[new_id] = model_entity_alignments
                        if j == 0:
                            new_concepts.append(node)
                        self.new_id += 1
                        if j == len(nodes)-1:
                            rel = ':name'
                            self.amr.edges.append((new_id, rel, name_id))
                            new_concepts.append(':name '+'name')
                        else:
                            rel = default_rel
                            self.amr.edges.append((new_id, rel, self.new_id))
                            new_concepts.append(default_rel+' ' + self.amr.nodes[new_id])

                op_idx = 1
                for tok in entity_tokens:
                    tok = tok.replace('"', '')
                    if tok in ['(', ')', '']:
                        continue
                    new_tok = '"' + tok[0].upper()+tok[1:] + '"'
                    self.amr.nodes[self.new_id] = new_tok
                    self.alignments[self.new_id] = model_entity_alignments
                    rel = f':op{op_idx}'
                    self.amr.edges.append((name_id, rel, self.new_id))
                    new_concepts.append(rel+' ' + new_tok)
                    self.new_id += 1
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
                new_id = entity_id if idx == 0 else self.new_id
                self.amr.nodes[new_id] = node
                self.alignments[new_id] = model_entity_alignments
                self.new_id += 1
                if idx > 0:
                    self.amr.edges.append((prev_id, default_rel, new_id))
                    new_concepts.append(default_rel+' ' + node)
                else:
                    new_concepts.append(node)
                prev_id = new_id
            for tok in entity_tokens:
                tok = tok.replace('"', '')
                if tok in ['(', ')', '']:
                    continue
                self.amr.nodes[self.new_id] = tok.lower()
                self.alignments[new_id] = model_entity_alignments
                self.amr.edges.append((prev_id, default_rel, self.new_id))
                new_concepts.append(default_rel+' ' + tok.lower())
                self.new_id += 1
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

        return '"' + string + '"'

    def clean_amr(self):
        if self.amr_graph:
            # delete (reduce) the nodes that were never confirmed or attached
            to_del = []
            for n in self.amr.nodes:
                found=False
                if n in self.is_confirmed:
                    found=True
                else:
                    for s, r, t in self.amr.edges:
                        if n == s or n == t:
                            found=True
                if not found:
                    to_del.append(n)
            for n in to_del:
                del self.amr.nodes[n]

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
                if not self.amr.nodes[n][0].isalpha() and not self.amr.nodes[n][0].isdigit() and not self.amr.nodes[n][0] in ['-', '+']:
                    self.amr.nodes[n] = '"' + self.amr.nodes[n].replace('"', '') + '"'
            # clean edges
            for j, e in enumerate(self.amr.edges):
                s, r, t = e
                if not r.startswith(':'):
                    r = ':'+r
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

    def connectGraph(self):
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
