import json
import torch
import os
import re
from collections import Counter, defaultdict

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
repo_root = os.path.realpath(f'{os.path.dirname(__file__)}/../')
entities_path = f'{repo_root}/data/entity_rules.json'

default_rel = ':rel'


class AMRStateMachine:

    def __init__(self, tokens, verbose=False, add_unaligned=0, 
                 action_list=None, action_list_by_prefix=None, 
                 nodes_by_token=None):
        """
        TODO: action_list containing list of allowed actions should be
        mandatory
        """

        tokens = tokens.copy()

        # add unaligned
        if add_unaligned and '<unaligned>' not in tokens:
            for i in range(add_unaligned):
                tokens.append('<unaligned>')
        # add root
        if '<ROOT>' not in tokens:
            tokens.append("<ROOT>")
        # init stack, buffer
        self.stack = []
        self.buffer = list(reversed([i+1 for i, tok in enumerate(tokens) if tok != '<unaligned>']))
        self.latent = list(reversed([i+1 for i, tok in enumerate(tokens) if tok == '<unaligned>']))

        # init amr
        self.amr = AMR(tokens=tokens)
        for i, tok in enumerate(tokens):
            if tok != "<ROOT>":
                self.amr.nodes[i+1] = tok
        # add root
        self.buffer[0] = -1
        self.amr.nodes[-1] = "<ROOT>"

        self.new_id = len(tokens)+1
        self.verbose = verbose
        # parser target output
        self.actions = []
        self.labels = []
        self.labelsA = []
        self.predicates = []

        # information for oracle
        self.merged_tokens = {}
        self.entities = []
        self.is_confirmed = set()
        self.is_confirmed.add(-1)
        self.swapped_words = {}

        # This will store the nodes aligned to a token according to the train
        # set
        self.nodes_by_token = nodes_by_token

        # FIXME: This should be mandatory and eveloped to be
        # consisten with oracle by design. Need to think how to do
        # this
        if action_list:
            self.action_list = action_list
            self.action_list_by_prefix = action_list_by_prefix

        if self.verbose:
            print('INIT')
            print(self.printStackBuffer())

    def __str__(self):

        # Command line styling

        def white_background(string):
            return "\033[107m%s\033[0m" % string
        
        def red_background(string):
            return "\033[101m%s\033[0m" % string
        
        def black_font(string):
            return "\033[30m%s\033[0m" % string

        def blue_font(string):
            return "\033[94m%s\033[0m" % string

        def green_font(string):
            return "\033[92m%s\033[0m" % string
        
        def stack_style(string):
            return black_font(white_background(string))

        def reduced_style(string):
            return black_font(red_background(string))

        # Tokens
        # tokens_str = ' '.join(self.amr.tokens)

        # AMR comment nottation
        amr_str = self.amr.toJAMRString(allow_incomplete=True)

        self.str_state = ""

        # Actions
        action_str = ' '.join([a for a in self.actions])

        # Buffer
        buffer_idx = [
            i - 1 if i != -1 else len(self.amr.tokens) - 1
            for i in reversed(self.buffer)
        ]
        buffer_str = " ".join([self.amr.tokens[i] for i in buffer_idx])

        # Stack
        stack_idx = [i - 1 for i in self.stack]
        stack_str = []
        for i in stack_idx:
            if i in self.merged_tokens:
                # Take into account merged tokens
                stack_str.append("(" + " ".join([
                    self.amr.tokens[j - 1] for j in self.merged_tokens[i]
                ]) + ")")
            else:
                stack_str.append(self.amr.tokens[i])
        stack_str = " ".join(stack_str)

        merged_pos = [y - 1  for x in self.merged_tokens.values() for y in x]

        # mask view
        mask_view = []
        pointer_view = []
        for position in range(len(self.amr.tokens)):
            # token
            token = str(self.amr.tokens[position])

            len_token = len(token)

            # color depedning on position
            if position in buffer_idx:
                token = token + ' '
            elif position in stack_idx:
                token = stack_style(token) + ' '
            elif position in merged_pos:
                token = stack_style(token + ' ')
            else:
                token = reduced_style(token) + ' ' 
            # position cursor
            if position in stack_idx and stack_idx.index(position) == len(stack_idx) - 1:
                pointer_view.append('_' * len_token + ' ')
            elif position in stack_idx and stack_idx.index(position) ==  len(stack_idx) - 2:
                pointer_view.append('-' * len_token + ' ')
            else:
                pointer_view.append(' ' * len_token + ' ')
            mask_view.append(token)

        mask_view_str = "".join(mask_view)
        pointer_view_str = "".join(pointer_view)

        # nodes
        nodes_str = " ".join([x for x in self.predicates if x != '_'])

        # Edges
        edges_str = []
        for items in self.amr.edges:
            i, label, j = items
            edges_str.append("%s %s %s" % (self.amr.nodes[i], blue_font(label), self.amr.nodes[j]))
        edges_str = "\n".join(edges_str)

        return """
%s\n%s\n\n
%s\n%s\n%s\n\n
%s\n%s\n\n
%s\n%s\n
        """ % (
            green_font("# Actions:"),
            action_str,
            green_font("# Buffer/Stack/Reduced:"),
            pointer_view_str,
            mask_view_str,
            green_font("# Predicates:"),
            nodes_str,
            green_font("# Edges:"),
            edges_str
        )

        #return f'{action_str}\n\n{buffer_str}\n{stack_str}\n\n{amr_str}'


    @classmethod
    def readAction(cls, action):
        s = [action]
        if action.startswith('DEPENDENT') or action in ['LA(root)', 'RA(root)', 'LA1(root)', 'RA1(root)']:
            return s
        if '(' in action:
            paren_idx = action.index('(')
            s[0] = action[:paren_idx]
            properties = action[paren_idx+1:-1]
            if ',' in properties:
                s.extend(properties.split(','))
            else:
                s.append(properties)
        return s

    def applyAction(self, act):

#         # This should be checked first for speed
#         if self.action_list.index(act) not in self.get_valid_action_indices():
#             import ipdb; ipdb.set_trace(context=30)
#             aa = 0

        action = self.readAction(act)
        action_label = action[0]
        if action_label in ['SHIFT']:
            if self.buffer:
                self.SHIFT()
            else:
                self.CLOSE()
                return True
        elif action_label in ['REDUCE', 'REDUCE1']:
            self.REDUCE()
            # ()
        elif action_label in ['LA', 'LA1']:
            self.LA(action[1] if action[1].startswith(':') else ':'+action[1])
        elif action_label in ['RA', 'RA1']:
            self.RA(action[1] if action[1].startswith(':') else ':'+action[1])
        elif action_label in ['LA(root)', 'LA1(root)']:
            self.LA('root')
        elif action_label in ['RA(root)', 'RA1(root)']:
            self.RA('root')
        elif action_label in ['PRED', 'CONFIRM']:
            self.CONFIRM(action[-1])
        elif action_label in ['COPY']:
            self.COPY()
        elif action_label in ['COPY_LITERAL']:
            self.COPY_LITERAL()
        elif action_label in ['COPY_RULE']:
            self.COPY_RULE()
        # TODO: Why multiple keywords for the same action?
        elif action_label in ['SWAP', 'UNSHIFT', 'UNSHIFT1']:
            self.SWAP()
        elif action_label in ['DUPLICATE']:
            self.DUPLICATE()
        elif action_label in ['INTRODUCE']:
            self.INTRODUCE()
        elif action_label.startswith('DEPENDENT'):
            paren_idx = action_label.index('(')
            properties = action_label[paren_idx + 1:-1].split(',')
            self.DEPENDENT(properties[1], properties[0])
        elif action_label in ['ADDNODE', 'ENTITY']:
            self.ENTITY(','.join(action[1:]))
        elif action_label in ['MERGE']:
            self.MERGE()
        elif action_label in ['CLOSE']:
            self.CLOSE()
            return True
        elif act == '</s>':
            # Do nothing action. Wait until other machines in the batch finish
            pass
        else:
            import ipdb; ipdb.set_trace(context=30)
            raise Exception(f'Unrecognized action: {act}')

    def applyActions(self, actions):
        for action in actions:
            is_closed = self.applyAction(action)
            if is_closed:
                return
        self.CLOSE()

    def get_valid_node(self):
        stack0 = self.stack[-1]
        import ipdb; ipdb.set_trace(context=30)

    def get_valid_action_indices(self):
        """Return valid actions for this state at test time (no oracle info)"""
        valid_action_indices = []

        # Buffer not empty
        if len(self.buffer):
            valid_action_indices.extend(self.action_list_by_prefix['SHIFT'])
            # FIXME: reduce also accepted here if node_id != None and something
            # aligned to it (see tryReduce)

        # One or more tokens in stack
        if len(self.stack) > 0:    
            valid_action_indices.extend(self.action_list_by_prefix['REDUCE'])
            valid_action_indices.extend(self.action_list_by_prefix['DEPENDENT'])

            # valid_node = get_valid_node(self):
            valid_action_indices.extend(self.action_list_by_prefix['PRED'])

            # Forbid entitity if top token already an entity
            if self.stack[-1] not in self.entities:
                # FIXME: Any rules involving MERGE here?
                valid_action_indices.extend(self.action_list_by_prefix['ENTITY'])

            # Forbid introduce if no latent
            if len(self.latent) > 0:
                valid_action_indices.extend(self.action_list_by_prefix['INTRODUCE'])

        # two or more tokens in stack
        if len(self.stack) > 1:    

            valid_action_indices.extend(self.action_list_by_prefix['LA'])
            valid_action_indices.extend(self.action_list_by_prefix['RA'])

            stack0 = self.stack[-1]
            stack1 = self.stack[-2]

            # Forbid merging if two words are identical
            if stack0 != stack1:
                valid_action_indices.extend(self.action_list_by_prefix['MERGE'])

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
                valid_action_indices.extend(self.action_list_by_prefix['SWAP'])

        # If not valid action indices machine is closed, output EOL
        if valid_action_indices == []:
            valid_action_indices = [self.action_list.index('</s>')]

        return valid_action_indices

    def update_from_logp(self, action_logp):
        """
        Given log probabilities of all actions update state machine with the
        most probable and return masked probabilities including only valid
        actions
        """

        # get indices of valid actions
        valid_indices = self.get_valid_action_indices()

        # Set all invalid actions to -Inf log probability
        const_action_logp = torch.ones_like(action_logp) * float('-inf')
        const_action_logp.to(action_logp.device)
        const_action_logp[valid_indices] = action_logp[valid_indices]

        best_action_index = const_action_logp.argmax()
        best_action = self.action_list[best_action_index]
        self.applyAction(best_action)

        return const_action_logp

    def SHIFT(self):
        """SHIFT : move buffer[-1] to stack[-1]"""

        # FIXME: No nested actions. This can be handled at try time
        if not self.buffer:
            self.CLOSE()
        tok = self.buffer.pop()
        self.stack.append(tok)
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
        if stack0 != -1 and stack0 not in self.entities:
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
        # old_label = self.amr.nodes[stack0].split(',')[-1]
        # old_label = old_label.replace(',','-COMMA-').replace(')','-PAREN-')
        self.amr.nodes[stack0] = node_label
        self.actions.append(f'PRED({node_label})')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        if self.verbose:
            print(f'PRED({node_label})')
            print(self.printStackBuffer())

    def COPY(self):
        """COPY: Same as CONFIRM but use lowercased top-of-stack"""
        stack0 = self.stack[-1]
        node_label = self.amr.nodes[stack0].lower()
        self.actions.append(f'COPY')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        if self.verbose:
            print(f'COPY')
            print(self.printStackBuffer())

    def COPY_LITERAL(self):
        """COPY_LITERAL: Same as CONFIRM but use top-of-stack between quotes"""
        stack0 = self.stack[-1]
        node_label = '\"%s\"' % self.amr.nodes[stack0]
        self.actions.append(f'COPY_LITERAL')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        if self.verbose:
            print(f'COPY_LITERAL')
            print(self.printStackBuffer())

    def COPY_RULE(self):
        """COPY_RULE:"""
        stack0 = self.stack[-1]
        # Get most common aligned node name
        node_label = self.nodes_by_token[self.amr.nodes[stack0]].most_common(1)[0][0]
        self.actions.append(f'COPY_RULE')
        self.labels.append('_')
        self.labelsA.append('_')
        self.predicates.append(node_label)
        self.is_confirmed.add(stack0)
        if self.verbose:
            print(f'COPY_RULE')
            print(self.printStackBuffer())

    def LA(self, edge_label):
        """LA : add an edge from stack[-1] to stack[-2]"""

        head = self.stack[-1]
        dependent = self.stack[-2]
        self.amr.edges.append((head, edge_label, dependent))
        self.actions.append(f'LA({edge_label.replace(":","")})')
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

        head = self.stack[-2]
        dependent = self.stack[-1]
        self.amr.edges.append((head, edge_label, dependent))
        self.actions.append(f'RA({edge_label.replace(":","")})')
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
        merged = ','.join(self.amr.tokens[x - 1].replace(',', '-COMMA-') for x in self.merged_tokens[lead])

        for i, e in enumerate(self.amr.edges):
            if e[1] == 'entity':
                continue
            if sec == e[0]:
                self.amr.edges[i] = (lead, e[1], e[2])
            if sec == e[2]:
                self.amr.edges[i] = (e[0], e[1], lead)

        # Just in case you merge entities. This shouldn't happen but might.
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

        self.amr.nodes[child_id] = self.amr.nodes[head]
        self.amr.nodes[head] = f'({entity_type})'
        self.amr.edges.append((head, 'entity', child_id))
        self.entities.append(head)

        self.actions.append(f'ADDNODE({entity_type})')
        self.labels.append('_')
        self.labelsA.append(f'{entity_type}')
        self.predicates.append('_')
        if self.verbose:
            print(f'ADDNODE({entity_type})')
            print(self.printStackBuffer())

    def DEPENDENT(self, edge_label, node_label, node_id=None):
        """DEPENDENT : add a single edge and node"""

        head = self.stack[-1]
        new_id = self.new_id

        edge_label = edge_label if edge_label.startswith(':') else ':'+edge_label

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
        if training and not use_addnonde_rules:
            self.postprocessing_training(gold_amr)
        else:
            self.postprocessing(gold_amr)

        for item in self.latent:
            if item in self.amr.nodes and not any(item == s or item == t for s, r, t in self.amr.edges):
                del self.amr.nodes[item]
        self.latent = []
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
        if self.verbose:
            print('CLOSE')
            print(self.printStackBuffer())
            print(self.amr.toJAMRString())

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
                    self.new_id += 1
                    if len(nodes) == 0:
                        new_concepts.append('name')
                    for j, node in enumerate(nodes):
                        new_id = entity_id if j == 0 else self.new_id
                        self.amr.nodes[new_id] = node
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
