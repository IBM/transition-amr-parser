import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy

import spacy
from spacy.tokens.doc import Doc

from transition_amr_parser.amr import AMR

"""
    AMRStateMachine applies operations in a transition-based AMR parser

    Actions are
        SHIFT : move buffer[-1] to stack[-1]
        REDUCE : delete token from stack[-1]
        CONFIRM : assign a node concept
        SWAP : move stack[-2] to buffer
        LA(label) : stack[-1] parent of stack[-2]
        RA(label) : stack[-2] parent of stack[-1]
        ENTITY(type) : form a named entity
        MERGE : merge two tokens (for MWEs)
        DEPENDENT(edge,node) : Add a node which is a dependent of stack[-1]
        CLOSE : complete AMR, run post-processing
"""
entity_rules_json = None
NUM_RE = re.compile(r'^([0-9]|,)+(st|nd|rd|th)?$')
entity_rule_stats = Counter()
entity_rule_totals = Counter()
entity_rule_fails = Counter()

# get path of provided entity_rules
# repo_root = os.path.realpath(f'{os.path.dirname(__file__)}')
# entities_path = f'{repo_root}/entity_rules.json'

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


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


def stack_style(string, confirmed=False):
    if confirmed:
        return black_font(green_background(string))
    else:
        return black_font(white_background(string))


def reduced_style(string):
    return black_font(red_background(string))


def get_forbidden_actions(stack, amr):
    '''
    Return actions that create already existing edges (to forbid them)
    '''

    if len(stack) == 0:
        return []

    # Regex for ARGs
    unique_re = re.compile(r'^(snt|op)([0-9]+)$')
    arg_re = re.compile(r'^ARG([0-9]+)$')
    argof_re = re.compile(r'^ARG([0-9]+-of)$')

    invalid_actions = []
    for t in amr.edges:

        # if we find an edge in the list of non repeatable edges, check if
        # this parent has already had one such edge and forbid repetition in
        # that case. 

        head_id = t[0]
        edge = t[1][1:]
        child_id = t[2]

        # FIXME: There is some bug somewhere by which edges are pointing to
        # unexisting nodes. We skip those cases
        if head_id not in amr.nodes or child_id not in amr.nodes:
            warning = yellow_font('WARNING')
            print(f'{warning}: Edge node id missing from amr.nodes')
            continue

        # info about this edge
        head = amr.nodes[head_id]
        child = amr.nodes[child_id]

        # unique constants
        # this edge label can not be repeated with same child name
        # note that DEPENDENT can be used as well as RA/LA
        if edge in ['polarity', 'mode']:
            if head_id == stack[-1]:
                # DEPENDENT
                invalid_actions.append(f'DEPENDENT({child},{edge})')
            if len(stack) > 1:
                if head_id == stack[-1] and (child == amr.nodes[stack[-2]]):
                    # LA (stack1 <-- stack0)
                    invalid_actions.append(f'LA({edge})')
                elif head_id == stack[-2] and (child == amr.nodes[stack[-1]]):
                    # RA (stack1 --> stack0)            
                    invalid_actions.append(f'RA({edge})')

        # snt[0-9] op[0-9] 
        # this edge label can not be repeated regardless of child label
        elif unique_re.match(edge) and len(stack) > 1: 
            if head_id == stack[-1]:
                # Left Arcs (stack1 <-- stack0)
                invalid_actions.append(f'LA({edge})')
            elif head_id == stack[-2]:
                # Right Arcs (stack1 --> stack0)            
                invalid_actions.append(f'RA({edge})')

        # ARG[0-9]
        # this edge label can not be repeated regardless of child label
        # watch for reverse ARG-of arcs
        elif arg_re.match(edge) and len(stack) > 1: 
            if head_id == stack[-1]:
                # Left Arcs (stack1 <-- stack0)
                invalid_actions.append(f'LA({edge})')
                invalid_actions.append(f'RA({edge}-of)')
            elif head_id == stack[-2]:
                # Right Arcs (stack1 --> stack0)            
                invalid_actions.append(f'RA({edge})')
                invalid_actions.append(f'LA({edge}-of)')

        # ARG[0-9]-of 
        # this edge label can not be repeated regardless of parent label
        # watch for direct ARG arcs
        # NOTE: father and child roles reverse wrt ARG[0-9]
        elif argof_re.match(edge) and len(stack) > 1:
            if child_id == stack[-1]:
                # Left Arcs (stack1 <-- stack0)
                argn = edge.split('-')[0]
                invalid_actions.append(f'LA({argn})')
                invalid_actions.append(f'RA({edge})')
                
            elif child_id == stack[-2]:
                # Right Arcs (stack1 --> stack0)            
                argn = edge.split('-')[0]
                invalid_actions.append(f'RA({argn})')
                invalid_actions.append(f'LA({edge})')

    return invalid_actions


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


def get_graph_str(amr, alignments):

    # nodes
    graph_str = ''
    nodes = []
    for stack0, token_pos in alignments.items():
        if stack0 in amr.nodes:
            nodes.append(stack0)
    
    # directed edges
    child2parents = defaultdict(list)        
    parent2child = defaultdict(list)        
    edge_labels = {}
    for parent, label, child in amr.edges:
        child2parents[child].append(parent)
        parent2child[parent].append(child)
        edge_labels[(parent, child)] = label
    
    # root nodes
    root_nodes = [node for node in nodes if node not in child2parents]
    
    # transverse depth first and print graph
    pad = '    '
    for node in root_nodes:
        graph_str += f'{amr.nodes[node]}\n'
        path = [node]
        while path:
            if len(parent2child[path[-1]]):
                new_node = parent2child[path[-1]].pop()
                depth = len(path) 
                edge = blue_font(edge_labels[(path[-1], new_node)])
                graph_str += f'{pad*depth} {edge} {amr.nodes[new_node]}\n'
                path.append(new_node)
            else:
                # leaf found
                path.pop()

    return graph_str


class AMRStateMachine:

    def __init__(self, tokens, verbose=False, add_unaligned=0,
                 actions_by_stack_rules=None, amr_graph=True,
                 post_process=True, spacy_lemmatizer=None, entity_rules=None):
        """
        TODO: action_list containing list of allowed actions should be
        mandatory
        """

        self.entity_rules_path = entity_rules

        # word tokens of sentence
        self.tokens = tokens.copy()

        # build and store amr graph (needed e.g. for oracle and PENMAN)
        self.amr_graph = amr_graph
        self.post_process = post_process

        # spacy lemmatizer
        self.spacy_lemmatizer = spacy_lemmatizer
        self.lemmas = None

        # add unaligned
        if add_unaligned and '<unaligned>' not in self.tokens:
            for i in range(add_unaligned):
                self.tokens.append('<unaligned>')
        # add root
        if '<ROOT>' not in self.tokens:
            self.tokens.append("<ROOT>")
        # machine is active
        self.time_step = 0
        self.is_closed = False
        # init stack, buffer
        self.stack = []
        self.buffer = list(reversed([
            i+1 for i, tok in enumerate(self.tokens) if tok != '<unaligned>'
        ]))
        # add root
        self.buffer[0] = -1
        self.latent = list(reversed([
            i+1 for i, tok in enumerate(self.tokens) if tok == '<unaligned>'
        ]))

        # init amr
        if self.amr_graph:
            self.amr = AMR(tokens=self.tokens)
            for i, tok in enumerate(self.tokens):
                if tok != "<ROOT>":
                    self.amr.nodes[i+1] = tok
            self.amr.nodes[-1] = "<ROOT>"

        self.new_id = len(self.tokens) + 1
        self.verbose = verbose
        # parser target output
        self.actions = []
        self.labels = []
        self.labelsA = []
        self.predicates = []
        self.alignments = {}

        # information for oracle
        self.merged_tokens = {}
        self.entities = []
        self.is_confirmed = set()
        self.is_confirmed.add(-1)
        self.swapped_words = {}

        self.actions_by_stack_rules = actions_by_stack_rules

        if self.verbose:
            print('INIT')
            print(self.printStackBuffer())

    def __deepcopy__(self, memo):
        """
        Manual deep copy of the machine

        avoid deep copying spacy lemmatizer
        """
        cls = self.__class__
        result = cls.__new__(cls)
        # DEBUG: usew this to detect very heavy constants that can be refered
        # import time
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # start = time.time()
            if k in ['spacy_lemmatizer', 'actions_by_stack_rules']:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
            # print(k, time.time() - start)
        return result

    def get_buffer_stack_copy(self): 
        """Return copy of buffer and stack"""
        return list(self.buffer), list(self.stack)

    def __str__(self):
        """Command line styling"""

        display_str = ""

        # Actions
        action_str = ' '.join([a for a in self.actions])
        # update display str
        display_str += "%s\n%s\n\n" % (green_font("# Actions:"), action_str)

        # Buffer
        buffer_idx = [
            i - 1 if i != -1 else len(self.tokens) - 1
            for i in reversed(self.buffer)
        ]

        # Stack
        stack_idx = [i - 1 for i in self.stack]
        stack_str = []
        for i in stack_idx:
            if i in self.merged_tokens:
                # Take into account merged tokens
                stack_str.append("(" + " ".join([
                    self.tokens[j - 1] for j in self.merged_tokens[i]
                ]) + ")")
            else:
                stack_str.append(self.tokens[i])
        stack_str = " ".join(stack_str)

        merged_pos = [y - 1 for x in self.merged_tokens.values() for y in x]

        # mask view
        mask_view = []
        pointer_view = []
        for position in range(len(self.tokens)):
            # token
            token = str(self.tokens[position])
            len_token = len(token)
            # color depending on position
            if position in buffer_idx:
                token = token + ' '
            elif position in stack_idx:
                token = stack_style(token, position + 1 in self.is_confirmed) + ' '
            elif position in merged_pos:
                token = stack_style(token + ' ', position + 1 in self.is_confirmed)
            else:
                token = reduced_style(token) + ' '
            # position cursor
            if position in stack_idx and stack_idx.index(position) == len(stack_idx) - 1:
                pointer_view.append('_' * len_token + ' ')
            elif position in stack_idx and stack_idx.index(position) == len(stack_idx) - 2:
                pointer_view.append('-' * len_token + ' ')
            else:
                pointer_view.append(' ' * len_token + ' ')
            mask_view.append(token)

        mask_view_str = "".join(mask_view)
        pointer_view_str = "".join(pointer_view)
        # update display str
        display_str += "%s\n%s\n%s\n\n" % (green_font("# Buffer/Stack/Reduced:"), pointer_view_str, mask_view_str)

        # nodes (requires on the fly AMR computation)
        if self.amr_graph:

            node_items = []
            for stack0, token_pos in self.alignments.items():
                if stack0 not in self.amr.nodes:
                    continue
                node = self.amr.nodes[stack0]
                if isinstance(token_pos, tuple):
                    tokens = " ".join(self.tokens[p] for p in token_pos)
                else:
                    tokens = self.tokens[token_pos]
                node_items.append(f'{tokens}--{node}')
            nodes_str = "  ".join(node_items)
            # update display str
            display_str += "%s\n%s\n\n" % (green_font("# Alignments:"), nodes_str)
    
            # Graph 
            display_str += green_font("# Graph:\n")
            display_str += get_graph_str(self.amr, self.alignments)

        return display_str

    @classmethod
    def readAction(cls, action):
        """Read action format"""
        if '(' not in action:
            return action, None
        elif action == 'LA(root)':
            # To keep original name to keep learner happy
            return action, ['root']
        else:
            items = action.split('(')
            action_label = items[0]
            arg_string = items[1][:-1]
            if action_label not in ['PRED', 'CONFIRM']:
                # split by comma respecting quotes
                props = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', arg_string)
            else:
                props = [arg_string]

            # To keep original name to keep learner happy
            if action_label == 'DEPENDENT':
                action_label = action

            return action_label, props

    def get_top_of_stack(self, positions=False, lemma=False):
        """
        Returns surface symbols on top of the stack, inclucing merged

        positions=True  returns the positions (unique ids within sentence)
        """
        # to get the lemma, we will need the positions
        if lemma:
            # Compute lemmas for this sentence and cache it
            if self.lemmas is None:
                assert self.spacy_lemmatizer, "No spacy_lemmatizer provided"
                self.lemmas = [
                    x.lemma_ for x in self.spacy_lemmatizer(self.tokens[:-1])
                ] + ['ROOT']
            positions = True
        token = None
        merged_tokens = None
        if len(self.stack):
            stack0 = self.stack[-1]
            if positions:
                token = stack0 - 1
            else:
                token = str(self.tokens[stack0 - 1])
            # store merged tokens by separate
            if stack0 in self.merged_tokens:
                if positions:
                    merged_tokens = [i - 1 for i in self.merged_tokens[stack0]]
                else:
                    merged_tokens = [
                        str(self.tokens[i - 1])
                        for i in self.merged_tokens[stack0]
                    ]

        if lemma:
            token = self.lemmas[token]
            merged_tokens = None

        return token, merged_tokens

    def applyAction(self, act):

        action_label, properties = self.readAction(act)
        if action_label.startswith('SHIFT'):
            if self.buffer:
                self.SHIFT(properties[0] if properties else None)
            else:
                self.CLOSE()
                return True
        elif action_label in ['REDUCE', 'REDUCE1']:
            self.REDUCE()
        elif action_label in ['LA(root)', 'LA', 'LA1']:
            assert ':' not in properties, "edge format has no :"
            assert len(properties) == 1
            self.LA(properties[0])
            # Also close if LA(root)
            # FIXME: This breaks stack-LSTM (IndexError: pop from empty list)
#             if (
#                 properties[0] == 'root' and 
#                 self.tokens[self.stack[-1]] == '<ROOT>'
#             ):
#                 self.CLOSE()
        elif action_label in ['RA', 'RA1']:
            assert ':' not in properties, "edge format has no :"
            assert len(properties) == 1
            self.RA(properties[0])
        elif action_label in ['PRED', 'CONFIRM']:
            assert len(properties) == 1
            self.CONFIRM(properties[0])
        elif action_label in ['COPY_LEMMA']:
            self.COPY_LEMMA()
        elif action_label in ['COPY_SENSE01']:
            self.COPY_SENSE01()
        # TODO: Why multiple keywords for the same action?
        elif action_label in ['SWAP', 'UNSHIFT', 'UNSHIFT1']:
            self.SWAP()
        elif action_label in ['DUPLICATE']:
            self.DUPLICATE()
        elif action_label in ['INTRODUCE']:
            self.INTRODUCE()
        elif action_label.startswith('DEPENDENT'):
            self.DEPENDENT(*properties)
        elif action_label in ['ADDNODE', 'ENTITY']:
            # preprocessing
            self.ENTITY(",".join(properties))
        elif action_label in ['MERGE']:
            self.MERGE()
        elif action_label in ['CLOSE']:
            self.CLOSE()
            return True
        elif act == '</s>':
            # Do nothing action. Wait until other machines in the batch finish
            pass
        else:
            raise Exception(f'Unrecognized action: {act}')

        # Increase time step
        self.time_step += 1

    def applyActions(self, actions):
        for action in actions:
            is_closed = self.applyAction(action)
            if is_closed:
                return
        self.CLOSE()

    def get_pred_by_stack_rules(self):
        """Return valid actions given the stack rules"""

        # rule input
        token, merged_tokens = self.get_top_of_stack()
        if merged_tokens:
            token = ",".join(merged_tokens)
            # merged_token = ",".join(merged_tokens)
            # if merged_token in self.actions_by_stack_rules:
            #    token = merged_token

        # rule decision
        if token not in self.actions_by_stack_rules:
            valid_pred_actions = []
        else:
            # return nodes ordered by most common
            node_counts = sorted(
                self.actions_by_stack_rules[token].items(),
                key=lambda x: x[1],
                reverse=True
            )
            valid_pred_actions = [f'PRED({nc[0]})' for nc in node_counts]
        return valid_pred_actions

    def get_valid_actions(self):
        """Return valid actions for this state at test time"""

        # Quick exit for a closed machine
        if self.is_closed:
            return ['</s>'], []

        # NOTE: Example: valid_actions = ['LA'] invalid_actions = ['LA(:mod)']
        valid_actions = []
        invalid_actions = []

        # Buffer not empty
        if True:  # len(self.buffer):
            # Its admits a SHIFT for empty buffer interpreted as a close
            valid_actions.append('SHIFT')
            # FIXME: reduce also accepted here if node_id != None and something
            # aligned to it (see tryReduce)

        # One or more tokens in stack
        if len(self.stack) > 0:

            stack0 = self.stack[-1]

            # If not confirmed yet, it can be confirmed
            if stack0 not in self.is_confirmed:
                valid_actions.extend(['COPY_LEMMA', 'COPY_SENSE01'])
                if self.actions_by_stack_rules:
                    pred_stack_rules = self.get_pred_by_stack_rules()
                    if pred_stack_rules:
                        valid_actions.extend(pred_stack_rules)
                else:
                    valid_actions.append('PRED')

            valid_actions.extend(['REDUCE', 'DEPENDENT'])

            # Forbid entitity if top token already an entity
            if stack0 not in self.entities:
                # FIXME: Any rules involving MERGE here?
                # FIXME: Double naming to be rmoevd. This is a source of bugs.
                valid_actions.extend(['ENTITY', 'ADDNODE'])

            # Forbid introduce if no latent
            if len(self.latent) > 0:
                valid_actions.append('INTRODUCE')

        # two or more tokens in stack
        if len(self.stack) > 1:
            stack0 = self.stack[-1]
            stack1 = self.stack[-2]

            # Forbid merging if two words are identical
            # FIXME: ?? this rule does not make any sense, indices will never
            # be equal
            # if stack0 != stack1:
            valid_actions.append('MERGE')

            # Forbid SWAP if both words have been swapped already
            if (
                (
                    stack0 not in self.swapped_words or
                    stack1 not in self.swapped_words.get(stack0)
                ) and
                (
                    stack1 not in self.swapped_words or
                    stack0 not in self.swapped_words.get(stack1)
                )
            ):
                valid_actions.extend(['SWAP', 'UNSHIFT'])

            # confirmed nodes can be drawn edges between as long as they are
            # not repeated
            if (stack0 in self.is_confirmed and stack1 in self.is_confirmed):
                valid_actions.extend(['LA', 'RA'])

            # FIXME: special rule to account for oracle errors
            elif self.get_top_of_stack()[0] == 'me':
                valid_actions.extend(['RA(mode)'])

        # Forbid actions given graph. Right now avoid some edge duplicates by
        # forbidding LA, RA and DEPENDENT
        invalid_actions.extend(get_forbidden_actions(self.stack, self.amr))

        return valid_actions, invalid_actions

    # forward compatibility aliases
    def update(self, act):
        self.applyAction(act)

    def get_annotations(self):
        assert self.amr_graph, ".toJAMRString() requires amr_graph = True"
        return self.amr.toJAMRString()

    def SHIFT(self, shift_label=None):
        """SHIFT : move buffer[-1] to stack[-1]"""

        # FIXME: No nested actions. This can be handled at try time
        if not self.buffer:
            self.CLOSE()
        tok = self.buffer.pop()
        self.stack.append(tok)
        if shift_label:
            self.actions.append(f'SHIFT({shift_label})')
        else:
            self.actions.append('SHIFT')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print('SHIFT')
            print(self.printStackBuffer())

    def REDUCE(self):
        """REDUCE : delete token"""

        stack0 = self.stack.pop()
        # if stack0 has no edges, delete it from the amr
        if self.amr_graph and stack0 != -1 and stack0 not in self.entities:
            if len([e for e in self.amr.edges if stack0 in e]) == 0:
                if stack0 in self.amr.nodes:
                    del self.amr.nodes[stack0]
        self.actions.append('REDUCE')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print('REDUCE')
            print(self.printStackBuffer())

    def CONFIRM(self, node_label):
        """CONFIRM : assign a propbank label"""
        stack0 = self.stack[-1]
        if self.amr_graph:
            self.amr.nodes[stack0] = node_label
        self.actions.append(f'PRED({node_label})')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        # keep alignments
        token, merged_tokens = self.get_top_of_stack(positions=True)
        if merged_tokens:
            self.alignments[stack0] = tuple(merged_tokens)
        else:
            self.alignments[stack0] = token

        if self.verbose:
            print(f'PRED({node_label})')
            print(self.printStackBuffer())

    def COPY_LEMMA(self):
        """COPY_LEMMA: Same as CONFIRM but use lowercased top-of-stack"""
        # get top of stack and lemma
        stack0 = self.stack[-1]
        node_label, _ = self.get_top_of_stack(lemma=True)
        # update AMR graph
        if self.amr_graph:
            self.amr.nodes[stack0] = node_label
        # update statistics
        self.actions.append(f'COPY_LEMMA')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        # keep alignments
        token, merged_tokens = self.get_top_of_stack(positions=True)
        if merged_tokens:
            self.alignments[stack0] = tuple(merged_tokens)
        else:
            self.alignments[stack0] = token

        if self.verbose:
            print(f'COPY_LEMMA')
            print(self.printStackBuffer())

    def COPY_SENSE01(self):
        """COPY_SENSE01: Same as CONFIRM but use lowercased top-of-stack"""
        # get top of stack and lemma
        stack0 = self.stack[-1]
        lemma, _ = self.get_top_of_stack(lemma=True)
        node_label = f'{lemma}-01'
        # update AMR graph
        if self.amr_graph:
            self.amr.nodes[stack0] = node_label
        # update statistics
        self.actions.append(f'COPY_SENSE01')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        # keep alignments
        token, merged_tokens = self.get_top_of_stack(positions=True)
        if merged_tokens:
            self.alignments[stack0] = tuple(merged_tokens)
        else:
            self.alignments[stack0] = token

        if self.verbose:
            print(f'COPY_SENSE01')
            print(self.printStackBuffer())

    def LA(self, edge_label):
        """LA : add an edge from stack[-1] to stack[-2]"""

        # Add edge to graph
        if self.amr_graph:
            self.amr.edges.append((
                self.stack[-1],
                f':{edge_label}' if edge_label != 'root' else 'root',
                self.stack[-2],
            ))
        # keep track of other vars
        self.actions.append(f'LA({edge_label})')
        if edge_label != 'root':
            self.labels.append(edge_label)
        else:
            self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print(f'LA({edge_label})')
            print(self.printStackBuffer())

    def RA(self, edge_label):
        """RA : add an edge from stack[-2] to stack[-1]"""
        # Add edge to graph
        if self.amr_graph:
            self.amr.edges.append((
                self.stack[-2],
                f':{edge_label}' if edge_label != 'root' else 'root',
                self.stack[-1]
            ))
        # keep track of other vars
        self.actions.append(f'RA({edge_label})')
        if edge_label != 'root':
            self.labels.append(edge_label)
        else:
            self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print(f'RA({edge_label})')
            print(self.printStackBuffer())

    def MERGE(self):
        """MERGE : merge two tokens to be the same node"""

        lead = self.stack.pop()
        sec = self.stack.pop()
        self.stack.append(lead)

        # maintain merged tokens dict
        if lead not in self.merged_tokens:
            self.merged_tokens[lead] = [lead]
        if sec in self.merged_tokens:
            self.merged_tokens[lead] = self.merged_tokens[sec] + self.merged_tokens[lead]
        else:
            self.merged_tokens[lead].insert(0, sec)
        merged = ','.join(self.tokens[x - 1].replace(',', '-COMMA-') for x in self.merged_tokens[lead])

        if self.amr_graph:

            for i, e in enumerate(self.amr.edges):
                if e[1] == 'entity':
                    continue
                if sec == e[0]:
                    self.amr.edges[i] = (lead, e[1], e[2])
                if sec == e[2]:
                    self.amr.edges[i] = (e[0], e[1], lead)

            # Just in case you merge entities. This shouldn't happen but might.
            # FIXME: Now this code only active if self.amr_graph = True
            if lead in self.entities:
                entity_edges = [e for e in self.amr.edges if e[0] == lead and e[1] == 'entity']
                lead = [t for s, r, t in entity_edges][0]
            if sec in self.entities:
                entity_edges = [e for e in self.amr.edges if e[0] == sec and e[1] == 'entity']
                child = [t for s, r, t in entity_edges][0]
                del self.amr.nodes[sec]
                for e in entity_edges:
                    self.amr.edges.remove(e)
                self.entities.remove(sec)
                sec = child

            # make tokens into a single node
            del self.amr.nodes[sec]
            self.amr.nodes[lead] = merged

        elif lead in self.entities or sec in self.entities:

            # see FIXME above
            pass

        # if any token in this merged group is promoted, promote the rest
        # FIXME: sometimes lead is not in self.merged_tokens. Unclear why
        if (
            lead in self.merged_tokens and
            any(n in self.is_confirmed for n in self.merged_tokens[lead])
        ):
            for n in self.merged_tokens[lead]:
                self.is_confirmed.add(n)

        # update states
        self.actions.append(f'MERGE')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')

        if self.verbose:
            print(f'MERGE({self.amr.nodes[lead]})')
            print(self.printStackBuffer())

    def ENTITY(self, entity_type):
        """ENTITY : create a named entity"""
        head = self.stack[-1]
        child_id = self.new_id
        self.new_id += 1

        if self.amr_graph:
            # Fixes :rel
            for (i,(s,l,t)) in enumerate(self.amr.edges):
                if s == head:
                    self.amr.edges[i] = (child_id,l,t)
            self.amr.nodes[child_id] = self.amr.nodes[head]
            self.amr.nodes[head] = f'({entity_type})'
            self.amr.edges.append((head, 'entity', child_id))
        self.entities.append(head)

        self.actions.append(f'ADDNODE({entity_type})')
        self.labels.append('_')
        self.labelsA.append(f'{entity_type}')
        self.predicates.append('_')
        self.is_confirmed.add(head)

        # keep alignments
        token, merged_tokens = self.get_top_of_stack(positions=True)
        if merged_tokens:
            self.alignments[head] = tuple(merged_tokens)
        else:
            self.alignments[head] = token

        if self.verbose:
            print(f'ADDNODE({entity_type})')
            print(self.printStackBuffer())

    def DEPENDENT(self, node_label, edge_label, node_id=None):
        """DEPENDENT : add a single edge and node"""

        head = self.stack[-1]
        new_id = self.new_id

        edge_label = edge_label if edge_label.startswith(':') else ':'+edge_label

        if self.amr_graph:
            if node_id:
                new_id = node_id
            else:
                self.amr.nodes[new_id] = node_label
            self.amr.edges.append((head, edge_label, new_id))
        self.new_id += 1
        self.actions.append(f'DEPENDENT({node_label},{edge_label.replace(":","")})')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print(f'DEPENDENT({edge_label},{node_label})')
            print(self.printStackBuffer())

    def SWAP(self):
        """SWAP : move stack[1] to buffer"""

        stack0 = self.stack.pop()
        stack1 = self.stack.pop()
        self.buffer.append(stack1)
        self.stack.append(stack0)
        self.actions.append('UNSHIFT')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if stack1 not in self.swapped_words:
            self.swapped_words[stack1] = []
        self.swapped_words[stack1].append(stack0)
        if self.verbose:
            print('UNSHIFT')
            print(self.printStackBuffer())

    def INTRODUCE(self):
        """INTRODUCE : move latent[-1] to stack"""

        latent0 = self.latent.pop()
        self.stack.append(latent0)
        self.actions.append('INTRODUCE')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        if self.verbose:
            print('INTRODUCE')
            print(self.printStackBuffer())

    def CLOSE(self, training=False, gold_amr=None, use_addnonde_rules=False):
        """CLOSE : finish parsing"""

        self.buffer = []
        self.stack = []
        if self.amr_graph and self.post_process:
            if training and not use_addnonde_rules:
                self.postprocessing_training(gold_amr)
            else:
                self.postprocessing(gold_amr)

            for item in self.latent:
                if item in self.amr.nodes and not any(item == s or item == t for s, r, t in self.amr.edges):
                    del self.amr.nodes[item]
            self.latent = []

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
            self.connectGraph()

        self.actions.append('SHIFT')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append('_')
        # FIXME: Make sure that this is not needed when amr_graph = False
        if self.amr_graph:
           self.convert_state_machine_alignments_to_amr_alignments()
        if self.verbose:
            print('CLOSE')
            if self.amr_graph:
                print(self.printStackBuffer())
                print(self.amr.toJAMRString())

        # Close the machine
        self.is_closed = True

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
                    new_list.append(alignment+1)
                self.amr.alignments[node] = deepcopy(new_list)

    def printStackBuffer(self):
        s = 'STACK [' + ' '.join(self.amr.nodes[x] if x in self.amr.nodes else 'None' for x in self.stack) + '] '
        s += 'BUFFER [' + ' '.join(self.amr.nodes[x] if x in self.amr.nodes else 'None' for x in reversed(self.buffer)) + ']\n'
        if self.latent:
            s += 'LATENT [' + ' '.join(self.amr.nodes[x] if x in self.amr.nodes else 'None' for x in reversed(self.latent)) + ']\n'
        return s

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
                if n not in descendents[self.amr.root]:
                    self.amr.edges.append((self.amr.root, default_rel, n))

    def postprocessing_training(self, gold_amr):

        for entity_id in self.entities:

            entity_edges = [e for e in self.amr.edges if e[0] == entity_id and e[1] == 'entity']

            for e in entity_edges:
                self.amr.edges.remove(e)

            child_id = [t for s, r, t in entity_edges][0]
            del self.amr.nodes[child_id]

            new_node_ids = []

            entity_alignment = gold_amr.alignmentsToken2Node(entity_id)
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

        # FIXME: All of this below. 

        global entity_rules_json, entity_rule_stats, entity_rule_totals, entity_rule_fails
        assert self.entity_rules_path, "you need to provide entity_rules"
        if not entity_rules_json:
            with open(self.entity_rules_path, 'r', encoding='utf8') as f:
                entity_rules_json = json.load(f)

        for entity_id in self.entities:

            if entity_id not in self.amr.nodes:
                continue
            # Test postprocessing ----------------------------
            gold_concepts = []
            if gold_amr:
                entity_alignment = gold_amr.alignmentsToken2Node(entity_id)
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
                    if tok.lower() in date_entity_rules.get(':weekday', []):
                        assigned_edges[j] = ':weekday'
                        continue
                    if tok in date_entity_rules.get(':timezone', []):
                        assigned_edges[j] = ':timezone'
                        continue
                    if tok.lower() in date_entity_rules.get(':calendar', []):
                        assigned_edges[j] = ':calendar'
                        if tok.lower() == 'lunar':
                            entity_tokens[j] = 'moon'
                        continue
                    if tok.lower() in date_entity_rules.get(':dayperiod', []):
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
                    if tok in date_entity_rules.get(':era', []) or tok.lower() in date_entity_rules.get(':era', []) \
                            or ('"' in tok and tok.replace('"', '') in date_entity_rules.get(':era', [])):
                        assigned_edges[j] = ':era'
                        continue
                    if tok.lower() in date_entity_rules.get(':season', []):
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

                # Fixes :rel
                leaf = None
                srcs = [None]
                for s, r, t in edges:
                    srcs.append(s)
                    if leaf in srcs and t not in srcs:
                        leaf = t
                if leaf != None:
                    for (i,e) in enumerate(self.amr.edges):
                        if e[0] == child_id:
                            self.amr.edges[i] = (id_map[leaf],e[1],e[2])

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

                # Fixes :rel
                leaf = None
                srcs = [None]
                for s, r, t in edges:
                    srcs.append(s)
                    if leaf in srcs and t not in srcs:
                        leaf = t
                if leaf != None:
                    for (i,e) in enumerate(self.amr.edges):
                        if e[0] == child_id:
                            self.amr.edges[i] = (id_map[leaf],e[1],e[2])

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

                    # Fixes rel:
                    leaf = None
                    srcs = [None]
                    for s, r, t in edges:
                        srcs.append(s)
                        if leaf in srcs and t not in srcs:
                            leaf = t
                    if leaf != None:
                        for (i,e) in enumerate(self.amr.edges):
                            if e[0] == child_id:
                                self.amr.edges[i] = (id_map[leaf],e[1],e[2])

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
                    # Fixes rel:
                    for (i,e) in enumerate(self.amr.edges):
                        if e[0] == prev_id:
                            self.amr.edges[i] = (new_id,e[1],e[2])

                    self.amr.edges.append((prev_id, default_rel, new_id))
                    new_concepts.append(default_rel+' ' + node)
                else:
                    new_concepts.append(node)
                idx += 1
                prev_id = new_id

            for (i,e) in enumerate(self.amr.edges):
                if e[0] == child_id:
                    if (prev_id,e[1],e[2]) not in self.amr.edges:
                        self.amr.edges[i] = (prev_id,e[1],e[2])
                    else:
                        del self.amr.edges[i]

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


class DepParsingStateMachine():

    def __init__(self, tokens):

        # tokenized sentence
        assert tokens[-1] == "ROOT"
        self.tokens = tuple(tokens)

        # machine state
        # keep initial positions as token ids (since token type repeat
        # theselves)
        self.stack = []
        self.buffer = [i + 1 for i, tok in enumerate(self.tokens)][::-1] 
        self.buffer[0] = -1

        # extra state
        # store action history
        self.actions = []
        # FIXME: This has to be yet set
        self.is_closed = False
        # update counter
        self.time_step = 0

    def __str__(self):
        """Command line styling"""

        display_str = ""

        # Actions
        action_title = green_font("# Actions:")
        action_str = ' '.join([a for a in self.actions])
        display_str += f'{action_title}\n{action_str}\n\n' 

        # mask view
        # legacy positioning system
        token_positions = [i + 1 for i, tok in enumerate(self.tokens)][::-1] 
        token_positions[0] = -1

        mask_view = []
        pointer_view = []
        for idx, position in enumerate(token_positions):

            # token
            token = self.tokens[-(idx + 1)]
            len_token = len(token)

            # color depending on position
            if position in self.buffer:
                token = token + ' '
            elif position in self.stack:
                token = stack_style(token, False) + ' '
            else:
                token = reduced_style(token) + ' '

            # position cursor
            if (
                position in self.stack and 
                self.stack.index(position) == len(self.stack) - 1
            ):
                pointer_view.append('_' * len_token + ' ')
            elif (
                position in self.stack and 
                self.stack.index(position) == len(self.stack) - 2
            ):
                pointer_view.append('-' * len_token + ' ')
            else:
                pointer_view.append(' ' * len_token + ' ')
            mask_view.append(token)

        # update display str
        title = green_font("# Buffer/Stack/Reduced:")
        pointer_view_str = "".join(pointer_view[::-1])
        mask_view_str = "".join(mask_view[::-1])
        display_str += f'{title}\n{pointer_view_str}\n{mask_view_str}\n\n' 

        return display_str

    def get_buffer_stack_copy(self): 
        """Return copy of buffer and stack"""
        return list(self.buffer), list(self.stack)

    def applyAction(self, action):
        """alias for compatibility"""

        base_action = action.split('(')[0] 

        if base_action == 'SHIFT':

            if self.buffer == []:
                # shift on empty buffer closes machine
                self.is_closed = True
                action = "SHIFT" 
            else:    
                # move one elements from stack to buffer
                self.stack.append(self.buffer.pop())
                # if shifted_pos is not None: #?
                # action = "%s(%s)" % (action, shifted_pos)

        elif base_action == 'LEFT-ARC':
            # remove second element in stack from the top
            # remove first element in stack from the top
            dependent = self.stack.pop(-2)
            # close machine if LA(root)
            if action == 'LEFT-ARC(root)':
                self.is_closed = True

        elif action.split('(')[0] == 'RIGHT-ARC':
            # remove first element in stack from the top
            dependent = self.stack.pop()

        elif action.split('(')[0] == 'SWAP':
            # set element 1 of the stack to 0 of the buffer
            self.buffer.append(self.stack.pop(1))

        elif action == '</s>':
            # FIXME: Why is this needed now? It was not before
            pass
        else: 
            raise Exception("Invalid action %s" % action)

        # store action history
        self.actions.append(action)

        # update counter
        self.time_step += 1

        return action

    def get_valid_actions(self):

        # Quick exit for a closed machine
        if self.is_closed:
            return ['</s>'], []

        # if top of the stack contains <ROOT> only LA(root allowed)
        if self.stack and self.stack[-1] == -1:
            return ['LEFT-ARC(root)'], []

        # multiple actions possible
        valid_actions = []
        if len(self.buffer) > 0:
            valid_actions.append('SHIFT')
        if len(self.stack) >= 2:
            valid_actions.append('LEFT-ARC')
            valid_actions.append('RIGHT-ARC')
            valid_actions.append('SWAP')

        return valid_actions, []
