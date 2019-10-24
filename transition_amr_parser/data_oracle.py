import json
import re
import argparse
from collections import Counter, defaultdict

from tqdm import tqdm

from transition_amr_parser.utils import print_log
from transition_amr_parser.io import writer
from transition_amr_parser.amr import JAMR_CorpusReader
from transition_amr_parser.state_machine import (
    AMRStateMachine,
    entity_rule_stats,
    entity_rule_totals,
    entity_rule_fails
)

"""
    This algorithm contains heuristics for solving
    transition-based AMR parsing in a rule based way.

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

use_addnode_rules = True


# Replacement rules for unicode chartacters
replacement_rules = {
    'ˈtʃærɪti': 'charity',
    '\x96': '_',
    '⊙': 'O'
}


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser oracle')
    # Single input parameters
    parser.add_argument(
        "--in-amr",
        help="AMR notation in LDC format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--in-propbank-args",
        help="Propbank argument data",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-oracle",
        help="tokens, AMR notation and actions given by oracle",
        default='oracle_actions.txt',
        type=str
    )
    parser.add_argument(
        "--out-sentences",
        help="tokenized sentences from --in-amr",
        type=str
    )
    parser.add_argument(
        "--out-actions",
        help="actions given by oracle",
        type=str
    )
    parser.add_argument(
        "--out-action-stats",
        help="statistics about actions",
        type=str
    )
    parser.add_argument(
        "--out-rule-stats",
        help="statistics about alignments",
        type=str
    )
    # Multiple input parameters
    parser.add_argument(
        "--out-amr",
        default='oracle_amrs.txt',
        help="corresponding AMR",
        type=str
    )
    #
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="verbose processing"
    )
    parser.add_argument(
        "--no-whitespace-in-actions",
        action='store_true',
        help="avoid tab separation in actions and sentences by removing whitespaces"
    )
    args = parser.parse_args()

    return args


def read_propbank(propbank_file):

    # Read frame argument description
    arguments_by_sense = {}
    with open(propbank_file) as fid:
        for line in fid:
            line = line.rstrip()
            sense = line.split()[0]
            arguments = [
                re.match('^(ARG.+):$', x).groups()[0]
                for x in line.split()[1:] if re.match('^(ARG.+):$', x)
            ]
            arguments_by_sense[sense] = arguments

    return arguments_by_sense


def get_node_alignment_counts(gold_amrs_train):

    node_by_token = defaultdict(lambda: Counter())
    for train_amr in gold_amrs_train:

        # Get alignments
        alignments = defaultdict(list)
        for i in range(len(train_amr.tokens)):
            for al_node in train_amr.alignmentsToken2Node(i+1):
                alignments[al_node].append(
                    train_amr.tokens[i]
                )

        for node_id, aligned_tokens in alignments.items():
            # join multiple words into one single expression
            if len(aligned_tokens) > 1:
                token_str = " ".join(aligned_tokens)
            else:
                token_str = aligned_tokens[0]

            node = train_amr.nodes[node_id]

            # count number of time a node is aligned to a token, indexed by
            # token
            node_by_token[token_str].update([node])

    return node_by_token


def is_most_common(node_counts, node, rank=0):

    return (
        (
            # as many results as the rank and node in that rank matches
            len(node_counts) == rank + 1 and
            node_counts.most_common(rank + 1)[-1][0] == node
        ) or (
            # more results than the rank, node in that rank matches, and rank
            # results is more probable than rank + 1
            len(node_counts) > rank + 1 and
            node_counts.most_common(rank + 1)[-1][0] == node and
            node_counts.most_common(rank + 1)[-1][1] >
            node_counts.most_common(rank + 2)[-1][1]
        )
     )


def compute_rules(gold_amrs, propbank_args):
    """
    Compute node to token alignmen statistic based on train data
    separate senses and non-senses aligned to tokens
    """
    nodes_by_token = dict(get_node_alignment_counts(gold_amrs))
    # sense counts
    sense_by_token = {
        key: Counter({
            value: count
            for value, count in counts.items()
            if value in propbank_args
        })
        for key, counts in nodes_by_token.items()
    }
    # lemma (or whatever ends up aligned)
    lemma_by_token = {
        key: Counter({
            value: count
            for value, count in counts.items()
            if value not in propbank_args
        })
        for key, counts in nodes_by_token.items()
    }
    return {
        'sense_by_token': sense_by_token,
        'lemma_by_token': lemma_by_token,
        'propbank_args_by_sense': propbank_args
    }


class AMR_Oracle:

    def __init__(self, verbose=False):
        self.amrs = []
        self.gold_amrs = []
        self.transitions = []
        self.verbose = verbose

        # predicates
        self.preds2Ints = {}
        self.possiblePredicates = {}

        self.new_edge = ''
        self.new_node = ''
        self.entity_type = ''
        self.dep_id = None

        self.swapped_words = {}

        self.possibleEntityTypes = Counter()

        self.stats = {
            'CONFIRM': Counter(),
            'COPY': Counter(),
            'COPY_LITERAL': Counter(),
            'COPY_SENSE': Counter(),
            'COPY_LEMMA': Counter(),
            'COPY_SENSE2': Counter(),
            'COPY_LEMMA2': Counter(),
            'REDUCE': Counter(),
            'SWAP': Counter(),
            'LA': Counter(),
            'RA': Counter(),
            'ENTITY': Counter(),
            'MERGE': Counter(),
            'DEPENDENT': Counter(),
            'INTRODUCE': Counter()
        }

    def read_actions(self, actions_file):
        transitions = []
        with open(actions_file, 'r', encoding='utf8') as f:
            sentences = f.read()
        sentences = sentences.replace('\r', '')
        sentences = sentences.split('\n\n')
        for sent in sentences:
            if not sent.strip():
                continue
            s = sent.split('\n')
            if len(s) < 2:
                raise IOError(f'Action file formatted incorrectly: {sent}')
            tokens = s[0].split('\t')
            actions = s[1].split('\t')
            transitions.append(AMRStateMachine(tokens))
            transitions[-1].applyActions(actions)
        self.transitions = transitions

    def runOracle(self, gold_amrs, propbank_args, out_oracle=None,
                  out_amr=None, out_sentences=None, out_actions=None,
                  out_rule_stats=None, add_unaligned=0,
                  no_whitespace_in_actions=False):

        print_log("oracle", "Parsing data")
        # deep copy of gold AMRs
        self.gold_amrs = [gold_amr.copy() for gold_amr in gold_amrs]

        # compute alignment statistics from JAMR and other alignments
        rule_stats = compute_rules(self.gold_amrs, propbank_args)
        if out_rule_stats:
            with open(out_rule_stats, 'w') as fid:
                fid.write(json.dumps(rule_stats))

        # open all files (if paths provided) and get writers to them
        oracle_write = writer(out_oracle)
        amr_write = writer(out_amr)
        sentence_write = writer(out_sentences, add_return=True)
        actions_write = writer(out_actions, add_return=True)

        # unaligned tokens
        included_unaligned = [
            '-', 'and', 'multi-sentence', 'person', 'cause-01', 'you', 'more',
            'imperative', '1', 'thing',
        ]

        # Loop over golf AMRs
        for sent_idx, gold_amr in tqdm(
            enumerate(self.gold_amrs),
            desc=f'computing oracle',
            total=len(self.gold_amrs)
        ):

            if self.verbose:
                print("New Sentence " + str(sent_idx) + "\n\n\n")

            # Initialize state machine
            tr = AMRStateMachine(
                gold_amr.tokens,
                verbose=self.verbose,
                add_unaligned=add_unaligned,
                rule_stats=rule_stats
            )
            self.transitions.append(tr)
            self.amrs.append(tr.amr)

            # clean alignments
            # TODO: describe this
            for i, tok in enumerate(gold_amr.tokens):
                align = gold_amr.alignmentsToken2Node(i+1)
                if len(align) == 2:
                    edges = [(s, r, t) for s, r, t in gold_amr.edges if s in align and t in align]
                    if not edges:
                        remove = 1
                        if (
                            gold_amr.nodes[align[1]].startswith(tok[:2]) or
                            len(gold_amr.alignments[align[0]]) >
                                len(gold_amr.alignments[align[1]])
                        ):
                            remove = 0
                        gold_amr.alignments[align[remove]].remove(i+1)
                        gold_amr.token2node_memo = {}

            # TODO: describe this
            if add_unaligned:
                for i in range(add_unaligned):
                    gold_amr.tokens.append("<unaligned>")
                    for n in gold_amr.nodes:
                        if n not in gold_amr.alignments or not gold_amr.alignments[n]:
                            if gold_amr.nodes[n] in included_unaligned:
                                gold_amr.alignments[n] = [len(gold_amr.tokens)]
                                break

            # add root node
            gold_amr.tokens.append("<ROOT>")
            gold_amr.nodes[-1] = "<ROOT>"
            gold_amr.edges.append((-1, "root", gold_amr.root))
            gold_amr.alignments[-1] = [-1]

            while tr.buffer or tr.stack:

                # top and second to top of the stack
                stack0 = tr.stack[-1] if tr.stack else 'NA'
                stack1 = tr.stack[-2] if len(tr.stack) > 1 else 'NA'

                if self.tryMerge(tr, tr.amr, gold_amr):
                    tr.MERGE()
                    toks = [tr.amr.tokens[x-1] for x in tr.merged_tokens[stack0]]
                    self.stats['MERGE'].update([','.join(toks)])

                elif self.tryEntity(tr, tr.amr, gold_amr):
                    # get top of the stack including merged symbols
                    if stack0 in tr.merged_tokens:
                        toks = [tr.amr.tokens[x-1] for x in tr.merged_tokens[stack0]]
                    else:
                        toks = [tr.amr.nodes[stack0]]
                    rule_str = ','.join(toks) + ' (' + self.entity_type + ')'
                    self.stats['ENTITY'].update([rule_str])
                    tr.ENTITY(entity_type=self.entity_type)

                elif self.tryConfirm(tr, tr.amr, gold_amr):

                    # TODO: Multi-word expressions
                    top_of_stack_tokens = tr.amr.tokens[stack0 - 1]

                    if top_of_stack_tokens.lower() == self.new_node:

                        # COPY (lowercased)
                        rule_str = top_of_stack_tokens.lower() + ' => ' + self.new_node
                        self.stats['COPY'].update([rule_str])
                        tr.COPY()

                    elif '\"%s\"' % top_of_stack_tokens == self.new_node:

                        # Copy literal
                        rule_str = '\"%s\"' % top_of_stack_tokens + ' => ' + self.new_node
                        self.stats['COPY_LITERAL'].update([rule_str])
                        tr.COPY_LITERAL()

                    elif (
                        top_of_stack_tokens in tr.sense_by_token or
                        top_of_stack_tokens in tr.lemma_by_token
                    ):

                        # COPY RULES
                        if is_most_common(
                            tr.sense_by_token[top_of_stack_tokens],
                            self.new_node
                        ):
                            # most common propbank sense matches
                            rule_str = f'{top_of_stack_tokens}  => {self.new_node}'
                            self.stats['COPY_SENSE'].update([rule_str])
                            tr.COPY_SENSE()
                        elif is_most_common(
                            tr.lemma_by_token[top_of_stack_tokens],
                            self.new_node
                        ):
                            # most common lemma sense matches
                            rule_str = f'{top_of_stack_tokens} => {self.new_node}'
                            self.stats['COPY_LEMMA'].update([rule_str])
                            tr.COPY_LEMMA()
                        elif is_most_common(
                            tr.sense_by_token[top_of_stack_tokens],
                            self.new_node,
                            rank=1
                        ):
                            # second most common propbank sense matches
                            rule_str = f'{top_of_stack_tokens} => {self.new_node}'
                            self.stats['COPY_SENSE2'].update([rule_str])
                            tr.COPY_SENSE2()
                        elif is_most_common(
                            tr.lemma_by_token[top_of_stack_tokens],
                            self.new_node,
                            rank=1
                        ):
                            # second most common lemma sense matches
                            rule_str = f'{top_of_stack_tokens} => {self.new_node}'
                            self.stats['COPY_LEMMA2'].update([rule_str])
                            tr.COPY_LEMMA2()
                        else:
                            # Confirm
                            rule_str = f'{tr.amr.nodes[stack0]} => {self.new_node}'
                            self.stats['CONFIRM'].update([rule_str])
                            tr.CONFIRM(node_label=self.new_node)

                    else:
                        # Confirm
                        rule_str = f'{tr.amr.nodes[stack0]} => {self.new_node}'
                        self.stats['CONFIRM'].update([rule_str])
                        tr.CONFIRM(node_label=self.new_node)

                elif self.tryDependent(tr, tr.amr, gold_amr):
                    tr.DEPENDENT(
                        edge_label=self.new_edge,
                        node_label=self.new_node,
                        node_id=self.dep_id
                    )
                    self.dep_id = None
                    tok = tr.amr.nodes[stack0]
                    rule_str = f'{self.new_edge} {self.new_node}'
                    self.stats['DEPENDENT'].update([rule_str])

                elif self.tryIntroduce(tr, tr.amr, gold_amr):
                    tok1 = tr.amr.nodes[tr.latent[-1]]
                    tok2 = tr.amr.nodes[stack0]
                    self.stats['INTRODUCE'].update([tok1 + ' ' + tok2])
                    tr.INTRODUCE()

                elif self.tryLA(tr, tr.amr, gold_amr):
                    tr.LA(edge_label=self.new_edge)
                    tok1 = tr.amr.nodes[stack0]
                    tok2 = tr.amr.nodes[stack1]
                    rule_str = tok1 + ' ' + self.new_edge + ' ' + tok2
                    self.stats['LA'].update([rule_str])

                elif self.tryRA(tr, tr.amr, gold_amr):
                    tr.RA(edge_label=self.new_edge)
                    tok1 = tr.amr.nodes[stack1]
                    tok2 = tr.amr.nodes[stack0]
                    rule_str = tok1 + ' ' + self.new_edge + ' ' + tok2
                    self.stats['RA'].update([rule_str])

                elif self.tryReduce(tr, tr.amr, gold_amr):
                    tok = tr.amr.nodes[stack0]
                    self.stats['REDUCE'].update([tok])
                    tr.REDUCE()

                elif self.trySWAP(tr, tr.amr, gold_amr):
                    tr.SWAP()
                    tok1 = tr.amr.nodes[stack1]
                    tok2 = tr.amr.nodes[stack0]
                    rule_str = f'swapped: {tok1} stack0: {tok2}'
                    self.stats['SWAP'].update([rule_str])

                elif tr.buffer:
                    tr.SHIFT()

                else:
                    tr.stack = []
                    tr.buffer = []
                    break

            tr.CLOSE(
                training=True,
                gold_amr=gold_amr,
                use_addnonde_rules=use_addnode_rules
            )

            # update files
            oracle_write(str(tr))
            amr_write(tr.amr.toJAMRString())
            sentence_write(" ".join(tr.amr.tokens))
            actions = " ".join([a for a in tr.actions])

            # TODO: Make sure this normalizing strategy is denornalized
            # elsewhere
            if no_whitespace_in_actions:
                confirmed = re.findall('PRED\(([^\)]*)\)', actions)
                whitepace_confirmed = [x for x in confirmed if ' ' in x]
                # ensure we do not have the normalization sign
                for concept in whitepace_confirmed:
                    assert '_' not in concept
                    normalized_concept = concept.replace(' ', '_')
                    actions = actions.replace(
                        f'PRED({concept})',
                        f'PRED({normalized_concept})'
                    )
            actions_write(actions)

            del gold_amr.nodes[-1]
        print_log("oracle", "Done")

        # close files if open
        oracle_write()
        amr_write()
        sentence_write()
        actions_write()

    def tryConfirm(self, transitions, amr, gold_amr):
        """
        Check if the next action is CONFIRM

        If the gold node label is different from the assigned label,
        return the gold label.
        """

        if not transitions.stack:
            return False

        # Rules that use oracle info

        stack0 = transitions.stack[-1]
        tok_alignment = gold_amr.alignmentsToken2Node(stack0)

        # TODO: What is the logic here?
        if 'DEPENDENT' not in transitions.actions[-1] and len(tok_alignment) != 1:
            return False

        # TODO: What is the logic here?
        if stack0 in transitions.entities:
            return False

        if len(tok_alignment) == 1:
            gold_id = tok_alignment[0]
        else:
            gold_id = gold_amr.findSubGraph(tok_alignment).root
        isPred = stack0 not in transitions.is_confirmed

        if isPred:
            # FIXME: state altering code should be outside of tryACTION
            new_node = gold_amr.nodes[gold_id]
            old_node = amr.nodes[stack0]

            if old_node not in self.possiblePredicates:
                self.possiblePredicates[old_node] = Counter()
            if new_node not in self.preds2Ints:
                self.preds2Ints.setdefault(new_node, len(self.preds2Ints))
            self.possiblePredicates[old_node][new_node] += 1
            self.new_node = new_node
        return isPred

    def tryLA(self, transitions, amr, gold_amr):
        """
        Check if the next action is LA (left arc)

        If there is an unpredicted edge from stack[-1] to stack[-2]
        return the edge label.
        """

        if len(transitions.stack) < 2:
            return False

        # Rules that use oracle info

        # check if we should MERGE instead
        if len(transitions.buffer) > 0:
            buffer0 = transitions.buffer[-1]
            stack0 = transitions.stack[-1]
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=buffer0):
                return False

        head = transitions.stack[-1]
        dependent = transitions.stack[-2]
        isLeftHead, labelL = self.isHead(amr, gold_amr, head, dependent)

        # FIXME: state altering code should be outside of tryACTION
        if isLeftHead:
            self.new_edge = labelL
        return isLeftHead

    def tryRA(self, transitions, amr, gold_amr):
        """
        Check if the next action is RA (right arc)

        If there is an unpredicted edge from stack[-2] to stack[-1]
        return the edge label.
        """

        if len(transitions.stack) < 2:
            return False

        # Rules that use oracle info

        # check if we should MERGE instead
        if len(transitions.buffer) > 0:
            buffer0 = transitions.buffer[-1]
            stack0 = transitions.stack[-1]
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=buffer0):
                return False

        head = transitions.stack[-2]
        dependent = transitions.stack[-1]
        isRightHead, labelR = self.isHead(amr, gold_amr, head, dependent)

        # FIXME: state altering code should be outside of tryACTION
        if isRightHead:
            self.new_edge = labelR
        return isRightHead

    def tryReduce(self, transitions, amr, gold_amr, node_id=None):
        """
        Check if the next action is REDUCE

        If
        1) there is nothing aligned to a token, or
        2) all gold edges are already predicted for the token,
        then return True.
        """

        if not transitions.stack and not node_id:
            return False

        # Rules that use oracle info

        stack0 = transitions.stack[-1]
        # FIXME: where is id defined?
        node_id = stack0 if not node_id else id

        tok_alignment = gold_amr.alignmentsToken2Node(node_id)
        if len(tok_alignment) == 0:
            return True

        # if we should merge, i.e. the alignment is the same as the next token,
        # do not reduce
        if transitions.buffer:
            buffer0 = transitions.buffer[-1]
            buffer0_alignment = gold_amr.alignmentsToken2Node(buffer0)
            if buffer0_alignment == tok_alignment:
                return False

        if len(tok_alignment) == 1:
            gold_id = tok_alignment[0]
        else:
            gold_id = gold_amr.findSubGraph(tok_alignment).root

        # check if all edges are already predicted

        countSource = 0
        countTarget = 0
        countSourceGold = 0
        countTargetGold = 0
        for s, r, t in amr.edges:
            if r == 'entity':
                continue
            if s == node_id:
                countSource += 1
            if t == node_id:
                countTarget += 1
        for s, r, t in gold_amr.edges:
            if s == gold_id:
                countSourceGold += 1
            if t == gold_id:
                countTargetGold += 1
        if node_id in transitions.entities:
            for s, r, t in gold_amr.edges:
                if s == gold_id and t in tok_alignment:
                    countSource += 1
                if t == gold_id and s in tok_alignment:
                    countTarget += 1
        if countSourceGold == countSource and countTargetGold == countTarget:
            return True
        return False

    def tryMerge(self, transitions, amr, gold_amr, first=None, second=None):
        """
        Check if the next action is MERGE

        Merge if two tokens have the same alignment.
        """

        # conditions
        if not first or not second:
            if len(transitions.stack) < 2:
                return False
            first = transitions.stack[-1]
            second = transitions.stack[-2]
        if first == second:
            return False

        # Rules that use oracle info

        first_alignment = gold_amr.alignmentsToken2Node(first)
        second_alignment = gold_amr.alignmentsToken2Node(second)
        if not first_alignment or not second_alignment:
            return False

        # If both tokens aremapped to same node or overlap
        if first_alignment == second_alignment:
            return True
        if set(first_alignment).intersection(set(second_alignment)):
            return True
        return False

    def trySWAP(self, transitions, amr, gold_amr):
        """
        Check if the next action is SWAP

        SWAP if there is an unpredicted gold edge between stack[-1]
        and some other node in the stack (blocked by stack[-2])
        or if stack1 can be reduced.
        """
        if len(transitions.stack) < 2:
            return False

        stack0 = transitions.stack[-1]
        stack1 = transitions.stack[-2]

        # Forbid if both words have been swapped already
        if stack0 in transitions.swapped_words and stack1 in transitions.swapped_words.get(stack0):
            return False
        if stack1 in transitions.swapped_words and stack0 in transitions.swapped_words.get(stack1):
            return False

        # Rules that use oracle info

        # check if we should MERGE instead
        if len(transitions.buffer) > 0:
            buffer0 = transitions.buffer[-1]
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=buffer0):
                return False

        # Look for tokens other than stack-top-two that can be head or child
        # of stack-top
        tok_alignment = gold_amr.alignmentsToken2Node(stack0)
        for tok in transitions.stack:
            if tok == stack1 or tok == stack0:
                continue
            isHead, labelL = self.isHead(amr, gold_amr, stack0, tok)
            if isHead:
                return True
            isHead, labelR = self.isHead(amr, gold_amr, tok, stack0)
            if isHead:
                return True
            # check if we need to merge two tokens separated by stack1
            k_alignment = gold_amr.alignmentsToken2Node(tok)
            if k_alignment == tok_alignment:
                return True
        # if not REPLICATE and self.tryReduce(transitions, amr, gold_amr, stack1):
        #     return True
        return False

    def tryDependent(self, transitions, amr, gold_amr):
        """
        Check if the next action is DEPENDENT


        Only for :polarity and :mode, if an edge and node is aligned
        to this token in the gold amr but does not exist in the predicted amr,
        the oracle adds it using the DEPENDENT action.
        """

        if not transitions.stack:
            return False

        # Rules that use oracle info

        stack0 = transitions.stack[-1]
        tok_alignment = gold_amr.alignmentsToken2Node(stack0)

        if not tok_alignment:
            return False

        if len(tok_alignment) == 1:
            source = tok_alignment[0]
        else:
            source = gold_amr.findSubGraph(tok_alignment).root

        for s, r, t in gold_amr.edges:
            if s == source and r in [":polarity", ":mode"]:
                # FIXME: state altering code should be outside of tryACTION
                # in this case we need to recompute ...
                if (stack0, r) in [(e[0], e[1]) for e in amr.edges]:
                    continue
                if t not in tok_alignment and (t in gold_amr.alignments and gold_amr.alignments[t]):
                    continue
                self.new_edge = r
                self.new_node = gold_amr.nodes[t]
                return True
        return False

    def tryEntity(self, transitions, amr, gold_amr):
        """
        Check if the next action is ENTITY
        """

        if not transitions.stack:
            return False

        stack0 = transitions.stack[-1]

        # check if already an entity
        if stack0 in transitions.entities:
            return False

        # Rules that use oracle info

        tok_alignment = gold_amr.alignmentsToken2Node(stack0)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return False

        # check if we should MERGE instead
        if len(transitions.stack) > 1:
            id = transitions.stack[-2]
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=id):
                return False
        for id in reversed(transitions.buffer):
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=id):
                return False

        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return False

        # check if we should use DEPENDENT instead
        if len(tok_alignment) == 2:
            if len(edges) == 1 and edges[0][1] in [':mode', ':polarity']:
                return False

        # FIXME: state altering code should be outside of tryACTION
        final_nodes = [n for n in tok_alignment if not any(s == n for s, r, t in edges)]
        new_nodes = [gold_amr.nodes[n] for n in tok_alignment if n not in final_nodes]
        self.entity_type = ','.join(new_nodes)
        self.possibleEntityTypes[self.entity_type] += 1

        return True

    def isHead(self, amr, gold_amr, x, y):
        """
        Check if the x is the head of y in the gold AMR graph

        If (the root of) x has an edge to (the root of) y in the gold AMR
        which is not in the predicted AMR, return True.
        """

        x_alignment = gold_amr.alignmentsToken2Node(x)
        y_alignment = gold_amr.alignmentsToken2Node(y)

        if not y_alignment or not x_alignment:
            return False, ''
        # get root of subgraph aligned to x
        if len(x_alignment) > 1:
            source = gold_amr.findSubGraph(x_alignment).root
        else:
            source = x_alignment[0]
        # get root of subgraph aligned to y
        if len(y_alignment) > 1:
            target = gold_amr.findSubGraph(y_alignment).root
        else:
            target = y_alignment[0]

        for s, r, t in gold_amr.edges:
            if source == s and target == t:
                # check if already assigned
                if (x, r, y) not in amr.edges:
                    return True, r
        return False, ''

    def tryIntroduce(self, transitions, amr, gold_amr):
        """
        TODO:
        """
        if not transitions.stack or not transitions.latent:
            return False
        stack0 = transitions.stack[-1]

        # Rules that use oracle info

        # check if we should MERGE instead
        if len(transitions.buffer) > 0:
            buffer0 = transitions.buffer[-1]
            stack0 = transitions.stack[-1]
            if self.tryMerge(transitions, amr, gold_amr, first=stack0, second=buffer0):
                return False

        idx = len(transitions.latent)-1
        for latentk in reversed(transitions.latent):
            isHead, label = self.isHead(amr, gold_amr, stack0, latentk)

            if isHead:
                # rearrange latent if necessary
                transitions.latent.append(transitions.latent.pop(idx))
                return True
            isHead, label = self.isHead(amr, gold_amr, latentk, stack0)

            if isHead:
                # rearrange latent if necessary
                transitions.latent.append(transitions.latent.pop(idx))
                return True
            idx -= 1
        return False


def main():

    # Argument handling
    args = argument_parser()

    # Load AMR
    corpus = JAMR_CorpusReader()
    corpus.load_amrs(args.in_amr)
    # FIXME: normalization shold be more robust. Right now use the tokens of
    # the amr inside the oracle. This is why we need to normalize them.
    for amr in corpus.amrs:
        amr.tokens = [
            replacement_rules.get(token, token) for token in amr.tokens
        ]

    # Load propbank
    propbank_args = read_propbank(args.in_propbank_args)

    print_log("amr", "Processing oracle")
    oracle = AMR_Oracle(verbose=args.verbose)
    oracle.runOracle(
        corpus.amrs,
        propbank_args,
        out_oracle=args.out_oracle,
        out_amr=args.out_amr,
        out_sentences=args.out_sentences,
        out_actions=args.out_actions,
        out_rule_stats=args.out_rule_stats,
        add_unaligned=0,
        no_whitespace_in_actions=args.no_whitespace_in_actions
    )

    # inform user
    for stat in oracle.stats:
        if args.verbose:
            print_log("amr", stat)
            print_log("amr", oracle.stats[stat].most_common(100))
            print_log("amr", "")

    if args.out_action_stats:
        # Store rule statistics
        with open(args.out_action_stats, 'w') as fid:
            fid.write(json.dumps(oracle.stats))

    if use_addnode_rules:
        for x in entity_rule_totals:
            perc = entity_rule_stats[x] / entity_rule_totals[x]
            if args.verbose:
                print_log(x, entity_rule_stats[x], '/',
                          entity_rule_totals[x], '=', f'{perc:.2f}')
        perc = sum(entity_rule_stats.values()) / \
            sum(entity_rule_totals.values())
        print_log('Totals:', f'{perc:.2f}')
        print_log('Totals:', 'Failed Entity Predictions:')
        print_log('Totals:', entity_rule_fails.most_common(1000))
