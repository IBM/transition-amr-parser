import json
import re
import argparse
from collections import Counter, defaultdict

from tqdm import tqdm

from transition_amr_parser.utils import print_log
from transition_amr_parser.io import writer, read_propbank
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
    )
    parser.add_argument(
        "--out-oracle",
        help="tokens, AMR notation and actions given by oracle",
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
        help="corresponding AMR",
        type=str
    )
    #
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="verbose processing"
    )
    #
    parser.add_argument(
        "--multitask-max-words",
        type=int,
        help="number of woprds to use for multi-task"
    )
    parser.add_argument(
        "--out-multitask-words",
        type=str,
        help="where to store top-k words for multi-task"
    )
    parser.add_argument(
        "--in-multitask-words",
        type=str,
        help="where to read top-k words for multi-task"
    )
    parser.add_argument(
        "--no-whitespace-in-actions",
        action='store_true',
        help="avoid tab separation in actions and sentences by removing whitespaces"
    )
    args = parser.parse_args()

    return args


def preprocess_amr(gold_amr, add_unaligned):

    # clean alignments
    for i, tok in enumerate(gold_amr.tokens):
        align = gold_amr.alignmentsToken2Node(i+1)
        if len(align) == 2:
            edges = [
                (s, r, t) 
                for s, r, t in gold_amr.edges 
                if s in align and t in align
            ]
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

    return gold_amr


def get_node_alignment_counts(gold_amrs_train):
    """Get statistics of alignments between nodes and surface words"""

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


def alert_inconsistencies(gold_amrs):

    def yellow_font(string):
        return "\033[93m%s\033[0m" % string

    num_sentences = len(gold_amrs)

    sentence_count = Counter() 
    amr_by_amrkey_by_sentence = defaultdict(dict)
    amr_counts_by_sentence = defaultdict(lambda: Counter())
    for amr in gold_amrs:

        # hash of sentence
        skey = " ".join(amr.tokens)

        # count number of time sentence repeated
        sentence_count.update([skey])

        # hash of AMR labeling
        akey = amr.toJAMRString()

        # store different amr labels for same sent, keep has map
        if akey not in amr_by_amrkey_by_sentence[skey]:
            amr_by_amrkey_by_sentence[skey][akey] = amr

        # count how many time each hash appears
        amr_counts_by_sentence[skey].update([akey])

    num_unique_sents = len(sentence_count)

    num_labelings = 0
    for skey, sent_count in sentence_count.items():
        num_labelings += len(amr_counts_by_sentence[skey]) 
        if len(amr_counts_by_sentence[skey]) > 1:
            pass
            # There is more than one labeling for this sentence
            # amrs = list(amr_by_amrkey_by_sentence[skey].values())

    # inform user
    if num_sentences > num_unique_sents:
        num_repeated = num_sentences - num_unique_sents 
        perc = num_repeated / num_sentences
        alert_str = '{:d}/{:d} {:2.1f} % repeated sents (max {:d} times)'.format(
            num_repeated,
            num_sentences,
            100 *perc,
            max(
                count 
                for counter in amr_counts_by_sentence.values() 
                for count in counter.values()
            )
        )
        print(yellow_font(alert_str))

    if num_labelings > num_unique_sents:
        num_inconsistent = num_labelings - num_unique_sents 
        perc = num_inconsistent / num_sentences 
        alert_str = '{:d}/{:d} {:2.4f} % inconsistent labelings from repeated sents'.format(
            num_inconsistent,
            num_sentences,
            perc
        )
        print(yellow_font(alert_str))


def read_multitask_words(multitask_list):
    multitask_words = []
    with open(multitask_list) as fid:
        for line in fid:
            items = line.strip().split('\t')
            if len(items) > 2:
                multitask_words.append(items[1])
    return multitask_words


def get_multitask_actions(max_symbols, tokenized_corpus):

    word_count = Counter()
    for sentence in tokenized_corpus:
        word_count.update([x for x in sentence])

    # Restrict to top-k words
    allowed_words = dict(list(sorted(
        word_count.items(),
        key=lambda x: x[1])
    )[-max_symbols:])
    # Add root regardless
    allowed_words.update({'ROOT': word_count['ROOT']})

    return allowed_words


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

        # DEBUG
        # self.copy_rules = False

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

    def runOracle(self, gold_amrs, propbank_args=None, out_oracle=None,
                  out_amr=None, out_sentences=None, out_actions=None,
                  out_rule_stats=None, add_unaligned=0,
                  no_whitespace_in_actions=False, multitask_words=None):
                    

        print_log("oracle", "Parsing data")
        # deep copy of gold AMRs
        self.gold_amrs = [gold_amr.copy() for gold_amr in gold_amrs]

        # print about inconsistencies in annotations
        alert_inconsistencies(self.gold_amrs)

        # open all files (if paths provided) and get writers to them
        oracle_write = writer(out_oracle)
        amr_write = writer(out_amr)
        sentence_write = writer(out_sentences, add_return=True)
        actions_write = writer(out_actions, add_return=True)

        # This will store overall stats
        self.stats = {
            'possible_predicates': Counter(),
            'action_vocabulary': Counter()
        }

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

            # TODO: Describe what is this pre-processing
            gold_amr = preprocess_amr(gold_amr, add_unaligned)

            # Initialize state machine
            tr = AMRStateMachine(
                gold_amr.tokens,
                verbose=self.verbose,
                add_unaligned=add_unaligned
            )
            self.transitions.append(tr)
            self.amrs.append(tr.amr)

            # Loop over potential actions
            while tr.buffer or tr.stack:

                # top and second to top of the stack
                stack0 = tr.stack[-1] if tr.stack else 'NA'
                stack1 = tr.stack[-2] if len(tr.stack) > 1 else 'NA'

                if self.tryMerge(tr, tr.amr, gold_amr):
                    action = 'MERGE'

                elif self.tryEntity(tr, tr.amr, gold_amr):
                    action = f'ADDNODE({self.entity_type})'

                elif self.tryConfirm(tr, tr.amr, gold_amr):
                    action = f'PRED({self.new_node})'

                elif self.tryDependent(tr, tr.amr, gold_amr):
                    # FIXME: Internally this is stored as
                    #
                    # DEPENDENT({node_label},{edge_label.replace(":","")})
                    #
                    edge = self.new_edge[1:] \
                        if self.new_edge.startswith(':') else self.new_edge
                    action = f'DEPENDENT({self.new_node},{edge})'
#                     tr.DEPENDENT(
#                         edge_label=self.new_edge,
#                         node_label=self.new_node,
#                         node_id=self.dep_id
#                     )
                    self.dep_id = None

                elif self.tryIntroduce(tr, tr.amr, gold_amr):
                    action = 'INTRODUCE'

                elif self.tryLA(tr, tr.amr, gold_amr):
                    if self.new_edge == 'root':
                        action = f'LA({self.new_edge})'
                    else:
                        action = f'LA({self.new_edge[1:]})'

                elif self.tryRA(tr, tr.amr, gold_amr):
                    if self.new_edge == 'root':
                        action = f'RA({self.new_edge})'
                    else:
                        action = f'RA({self.new_edge[1:]})'

                elif self.tryReduce(tr, tr.amr, gold_amr):
                    action = 'REDUCE'

                elif self.trySWAP(tr, tr.amr, gold_amr):
                    action = 'UNSHIFT'

                elif tr.buffer:
                    action = 'SHIFT'

                else:
                    tr.stack = []
                    tr.buffer = []
                    break

                # Store stats
                # get token(s) at the top of the stack
                token, merged_tokens = tr.get_top_of_stack()
                items = action.split('(')
                action_label = action.split('(')[0]
                if len(items) > 1:
                    tag = items[1][:-1]
                else:    
                    tag = None

                # check action has not invalid chars and normalize
                # TODO: --no-whitespace-in-actions being deprecated
                if no_whitespace_in_actions and action_label == 'PRED':
                    assert '_' not in action, \
                        "--no-whitespace-in-actions prohibits use of _ in actions"
                    if ' ' in action_label:
                        action = action.replace(' ', '_')
                  
                # Add prediction ot top of the buffer
                if action == 'SHIFT' and multitask_words is not None:
                    # top of buffer
                    top_of_buffer = tr.amr.tokens[tr.buffer[-1] - 1]
                    if top_of_buffer in multitask_words:
                        action = f'SHIFT({top_of_buffer})'
 
                # APPLY ACTION
                tr.applyAction(action)

            # Close machine
            tr.CLOSE(
                training=True,
                gold_amr=gold_amr,
                use_addnonde_rules=use_addnode_rules
            )

            # update files
            if out_oracle:
                # to avoid printing
                oracle_write(str(tr))
            amr_write(tr.amr.toJAMRString())
            if no_whitespace_in_actions:
                sentence_write(" ".join(tr.amr.tokens))
            else:    
                sentence_write("\t".join(tr.amr.tokens))
            # TODO: Make sure this normalizing strategy is denornalized
            # elsewhere
            if no_whitespace_in_actions:
                actions = " ".join([a for a in tr.actions])
            else:
                # used in stack-LSTM
                actions = "\t".join([a for a in tr.actions])
            actions_write(actions)
            # Update action count
            if no_whitespace_in_actions:
                self.stats['action_vocabulary'].update(actions.split())
            else:
                self.stats['action_vocabulary'].update(actions.split("\t"))
            del gold_amr.nodes[-1]

        print_log("oracle", "Done")

        # close files if open
        oracle_write()
        amr_write()
        sentence_write()
        actions_write()

        # State machine stats for this senetnce
        if out_rule_stats:

            # Add possible predicates to state machine rules
            # apply same normalization rule if solicited
            possible_predicates = defaultdict(lambda: Counter())
            for token, counts in self.possiblePredicates.items():
                for node, count in counts.items():
                    possible_predicates[token][node] = count
                    
            self.stats['possible_predicates'] = self.possiblePredicates

            with open(out_rule_stats, 'w') as fid:
                fid.write(json.dumps(self.stats))

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
    for idx, amr in enumerate(corpus.amrs):
        new_tokens = []
        for token in amr.tokens:
            forbidden = [x for x in replacement_rules.keys() if x in token]
            if forbidden:
                token = token.replace(forbidden[0], replacement_rules[forbidden[0]])
            new_tokens.append(token)    
        amr.tokens = new_tokens

    # Load propbank
    if args.in_propbank_args:
        propbank_args = read_propbank(args.in_propbank_args)
    else:    
        propbank_args = None

    # Load/Save words for multi-task
    if args.multitask_max_words:
        assert args.multitask_max_words
        assert args.out_multitask_words
        # get top words
        multitask_words = get_multitask_actions(
            args.multitask_max_words,
            [list(amr.tokens) for amr in corpus.amrs]
        )
        # store in file
        with open(args.out_multitask_words, 'w') as fid:
            for word in multitask_words.keys():
                fid.write('{word}\n')
    elif args.in_multitask_words:
        assert not args.multitask_max_words
        assert not args.out_multitask_words
        # store in file
        with open(args.in_multitask_words) as fid:
            multitask_words = [line.strip() for line in fid.readlines()]

    # TODO: At the end, an oracle is just a parser with oracle info. This could
    # be turner into a loop similar to parser.py (ore directly use that and a
    # AMROracleParser())
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
        no_whitespace_in_actions=args.no_whitespace_in_actions,
        multitask_words=multitask_words
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
