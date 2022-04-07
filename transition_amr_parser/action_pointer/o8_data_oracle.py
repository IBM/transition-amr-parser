import json
import argparse
from collections import Counter, defaultdict
import re

from tqdm import tqdm

from transition_amr_parser.io import read_propbank, read_amr, write_tokenized_sentences
from transition_amr_parser.action_pointer.o8_state_machine import (
    AMRStateMachine,
    get_spacy_lemmatizer
)


"""
This algorithm contains heuristics for generating linearized action sequences for AMR graphs in a rule based way.
The parsing algorithm is transition-based combined with pointers for long distance arcs.

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
"""

use_addnode_rules = True


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
        "--out-rule-stats",         # TODO this is accessed by replacing '-' to '_'
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
    # Labeled shift args
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
    # copy lemma action
    parser.add_argument(
        "--copy-lemma-action",
        action='store_true',
        help="Use copy action from Spacy lemmas"
    )
    # copy lemma action
    parser.add_argument(
        "--addnode-count-cutoff",
        help="forbid all addnode actions appearing less times than count",
        type=int
    )

    parser.add_argument(
        "--in-pred-entities",
        type=str,
        default="person,thing",
        help="comma separated list of entity types that can have pred"
    )
    
    args = parser.parse_args()

    return args


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


entities_with_preds = []

def preprocess_amr(gold_amr, add_unaligned=None, included_unaligned=None, root_id=-1):

    # clean alignments
    for i, tok in enumerate(gold_amr.tokens):
        align = gold_amr.alignmentsToken2Node(i + 1)
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
                gold_amr.alignments[align[remove]].remove(i + 1)
                gold_amr.token2node_memo = {}

    # clean invalid alignments: sometimes the alignments are outside of the sentence boundary
    # TODO check why this happens in the data reading process
    # TODO and fix that and remove this cleaning process
    # an example is in training data, when the sentence is
    # ['Among', 'common', 'birds', ',', 'a', 'rather', 'special', 'one', 'is',
    # 'the', 'black', '-', 'faced', 'spoonbill', '.']
    # TODO if not dealt with, this causes a problem when the root aligned token id is sentence length (not -1)
    for nid, tids in gold_amr.alignments.items():
        gold_amr.alignments[nid] = [tid for tid in tids if tid <= len(gold_amr.tokens)]

    # TODO: describe this
    # append a special token at the end of the sentence for the first unaligned node
    # whose label is in `included_unaligned` to align to
    # repeat `add_unaligned` times
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
    gold_amr.nodes[root_id] = "<ROOT>"
    gold_amr.edges.append((root_id, "root", gold_amr.root))
    # gold_amr.alignments[root_id] = [-1]   # NOTE do not do this; we have made all the token ids natural positive index
    # setting a token id to -1 will break the code
    gold_amr.alignments[root_id] = [len(gold_amr.tokens)]    # NOTE shifted by 1 for AMR alignment

    return gold_amr


def get_node_alignment_counts(gold_amrs_train):
    """Get statistics of alignments between nodes and surface words"""

    node_by_token = defaultdict(lambda: Counter())
    for train_amr in gold_amrs_train:

        # Get alignments
        alignments = defaultdict(list)
        for i in range(len(train_amr.tokens)):
            for al_node in train_amr.alignmentsToken2Node(i + 1):
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


def sanity_check_amr(gold_amrs):

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
        alert_str = '{:d}/{:d} {:2.1f} % {:s} (max {:d} times)'.format(
            num_repeated,
            num_sentences,
            100 * perc,
            'repeated sents',
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
        alert_str = '{:d}/{:d} {:2.4f} % {:s}'.format(
            num_inconsistent,
            num_sentences,
            perc,
            'inconsistent labelings from repeated sents'
        )
        print(yellow_font(alert_str))


def sanity_check_actions(sentence_tokens, oracle_actions):

    pointer_arc_re = re.compile(r'^(LA|RA)\(([0-9]+),(.*)\)$')

    assert len(sentence_tokens) == len(oracle_actions)
    source_lengths = []
    target_lengths = []
    action_count = Counter()
    for tokens, actions in zip(sentence_tokens, oracle_actions):
        # filter actions to remove pointer
        for action in actions:
            if pointer_arc_re.match(action):
                items = pointer_arc_re.match(action).groups()
                action = f'{items[0]}({items[2]})'
            action_count.update([action])
        source_lengths.append(len(tokens))
        target_lengths.append(len(actions))
        pass

    singletons = [k for k, c in action_count.items() if c == 1]
    print('Base actions:')
    print(Counter([k.split('(')[0] for k in action_count.keys()]))
    print('Most frequent actions:')
    print(action_count.most_common(10))
    if singletons:
        base_action_count = [x.split('(')[0] for x in singletons]
        msg = f'{len(singletons)} singleton actions'
        print(yellow_font(msg))
        print(Counter(base_action_count))


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
            100 * perc,
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


def label_shift(state_machine, multitask_words):
    tok = state_machine.get_current_token(lemma=False)
    if tok in multitask_words:
        return f'SHIFT({tok})'
    else:
        return 'SHIFT'


def get_multitask_actions(max_symbols, tokenized_corpus, add_root=False):

    word_count = Counter()
    for sentence in tokenized_corpus:
        word_count.update([x for x in sentence])

    # Restrict to top-k words
    allowed_words = dict(list(sorted(
        word_count.items(),
        key=lambda x: x[1])
    )[-max_symbols:])

    if add_root:
        # Add root regardless
        allowed_words.update({'ROOT': word_count['ROOT']})

    return allowed_words


def process_multitask_words(tokenized_corpus, multitask_max_words,
                            in_multitask_words, out_multitask_words,
                            add_root=False):

    # Load/Save words for multi-task
    if multitask_max_words:
        assert multitask_max_words
        assert out_multitask_words
        # get top words
        multitask_words = get_multitask_actions(
            multitask_max_words,
            tokenized_corpus,
            add_root=add_root
        )
        # store in file
        with open(out_multitask_words, 'w') as fid:
            for word in multitask_words.keys():
                fid.write(f'{word}\n')
    elif in_multitask_words:
        assert not multitask_max_words
        assert not out_multitask_words
        # store in file
        with open(in_multitask_words) as fid:
            multitask_words = [line.strip() for line in fid.readlines()]
    else:
        multitask_words = None

    return multitask_words


def print_corpus_info(amrs):

    # print some info
    print(f'{len(amrs)} sentences')
    node_label_count = Counter([
        n for amr in amrs for n in amr.nodes.values()
    ])
    node_tokens = sum(node_label_count.values())
    print(f'{len(node_label_count)}/{node_tokens} node types/tokens')
    edge_label_count = Counter([t[1] for amr in amrs for t in amr.edges])
    edge_tokens = sum(edge_label_count.values())
    print(f'{len(edge_label_count)}/{edge_tokens} edge types/tokens')
    word_label_count = Counter([w for amr in amrs for w in amr.tokens])
    word_tokens = sum(word_label_count.values())
    print(f'{len(word_label_count)}/{word_tokens} word types/tokens')


class AMROracleBuilder:
    """Build AMR oracle for one sentence."""
    def __init__(self, gold_amr, lemmatizer, copy_lemma_action, multitask_words):

        self.gold_amr = gold_amr
        # initialize the state machine
        self.machine = AMRStateMachine(gold_amr.tokens, spacy_lemmatizer=lemmatizer, amr_graph=True, entities_with_preds=entities_with_preds)

        self.copy_lemma_action = copy_lemma_action
        self.multitask_words = multitask_words
        # TODO deprecate `multitask_words` or change for a better name such as `shift_label_words`

        # AMR construction states info
        self.nodeid_to_gold_nodeid = {}  # key: node id in the state machine, value: list of node ids in gold AMR
        self.nodeid_to_gold_nodeid[self.machine.root_id] = [-1]  # NOTE gold amr root id is fixed at -1
        self.built_gold_nodeids = []

    @property
    def tokens(self):
        return self.gold_amr.tokens

    @property
    def time_step(self):
        return self.machine.time_step

    @property
    def actions(self):
        return self.machine.actions

    def get_valid_actions(self):
        """Get the valid actions and invalid actions based on the current AMR state machine status and the gold AMR."""

        # find the next action
        # NOTE the order here is important, which is based on priority
        #      e.g. within node-arc actions, arc subsequence comes highest, then named entity subsequence, etc.

        # debug
        # on dev set, sentence id 459 (starting from 0) -> for DEPENDENT missing
        # if self.tokens == ['The', 'cyber', 'attacks', 'were', 'unprecedented', '.', '<ROOT>']:
        #     if self.time_step >= 8:
        #         breakpoint()

        action = self.try_reduce()
        if not action:
            action = self.try_merge()
        #if not action:
        #    action = self.try_dependent()
        if not action:
            action = self.try_arcs()
        #if not action:
        #    action = self.try_named_entities()
        if not action:
            action = self.try_entities_with_pred()
        if not action:
            action = self.try_entity()
        if not action:
            action = self.try_pred()

        if not action:
            if len(self.machine.actions) and self.machine.actions[-1] == 'SHIFT' and  self.machine.tok_cursor != self.machine.tokseq_len - 1 :
                action = 'REDUCE'
            else:
                action = 'SHIFT'

        if action == 'SHIFT' and self.multitask_words is not None:
            action = label_shift(self.machine, self.multitask_words)

        valid_actions = [action]
        invalid_actions = []

        return valid_actions, invalid_actions

    def build_oracle_actions(self):
        """Build the oracle action sequence for the current token sentence, based on the gold AMR
        and the alignment.
        """
        # Loop over potential actions
        # NOTE "<ROOT>" token at last position is added as a node from the beginning, so no prediction
        # for it here; the ending sequence is always [... SHIFT CLOSE] or [... LA(pos,'root') SHIFT CLOSE]
        machine = self.machine
        while not machine.is_closed:
            valid_actions, invalid_actions = self.get_valid_actions()
            # for now
            assert len(valid_actions) == 1, "Oracle must be deterministic"
            assert len(invalid_actions) == 0, "Oracle can\'t blacklist actions"
            action = valid_actions[0]

            # update the machine
            machine.apply_action(action)

        # close machine
        # below are equivalent
        # machine.apply_action('CLOSE', training=True, gold_amr=self.gold_amr)
        machine.CLOSE(training=True, gold_amr=self.gold_amr)

        return self.actions

    def try_reduce(self):
        """
        Check if the next action is REDUCE.

        If
        1) there is nothing aligned to a token.
        """
        machine = self.machine
        gold_amr = self.gold_amr

        if machine.current_node_id is not None:
            # not on the first time on a new token
            return None

        tok_id = machine.tok_cursor
        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)    # NOTE the index + 1
        if len(tok_alignment) == 0:
            return 'REDUCE'
        else:
            return None

    def try_merge(self):
        """
        Check if the next action is MERGE.

        If
        1) the current and the next token have the same node alignment.
        """
        machine = self.machine
        gold_amr = self.gold_amr

        if machine.current_node_id is not None:
            # not on the first time on a new token
            return None

        if machine.tok_cursor < machine.tokseq_len - 1:
            cur = machine.tok_cursor
            nxt = machine.tok_cursor + 1
            cur_alignment = gold_amr.alignmentsToken2Node(cur + 1)
            nxt_alignment = gold_amr.alignmentsToken2Node(nxt + 1)
            if not cur_alignment or not nxt_alignment:
                return None
            # If both tokens are mapped to same node or overlap
            if cur_alignment == nxt_alignment:
                return 'MERGE'
            if set(cur_alignment).intersection(set(nxt_alignment)):
                return 'MERGE'
            return None
        else:
            return None

    def try_named_entities(self):
        """
        Get the named entity sub-sequences one by one from the current surface token (segments).

        E.g.
        a) for one entity
        ENTITY('name') PRED('city') [other arcs] LA(pos,':name')
        b) for two entities with same surface tokens
        ENTITY('name') PRED('city') [other arcs] LA(pos,':name') PRED('city') [other arcs] LA(pos,':name')
        c) for two entities with two surface tokens
        ENTITY('name') PRED('city') [other arcs] LA(pos,':name') ENTITY('name') PRED('city') [other arcs] LA(pos,':name')
        """
        machine = self.machine
        gold_amr = self.gold_amr

        tok_id = machine.tok_cursor

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return None

        # check if there is any edge with the aligned nodes
        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return None

        # check if named entity case: (entity_category, ':name', 'name')
        entity_edges = []
        name_node_ids = []
        for s, r, t in edges:
            if r == ':name' and gold_amr.nodes[t] == 'name':
                entity_edges.append((s, r, t))
                name_node_ids.append(t)

        if not name_node_ids:
            return None

        for s, r, t in entity_edges:
            if t not in self.built_gold_nodeids:
                self.built_gold_nodeids.append(t)
                self.nodeid_to_gold_nodeid.setdefault(machine.new_node_id, []).append(t)
                return 'ENTITY(name)'
            if s not in self.built_gold_nodeids:
                self.built_gold_nodeids.append(s)
                self.nodeid_to_gold_nodeid.setdefault(machine.new_node_id, []).append(s)
                return f'PRED({gold_amr.nodes[s]})'

        return None

    def try_entities_with_pred(self):
        """
        allow pred inside entities that frequently need it i.e. person, thing
        """

        machine = self.machine
        gold_amr = self.gold_amr

        tok_id = machine.tok_cursor

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return None

        # check if there is any edge with the aligned nodes
        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return None

        is_dependent = False
        for s, r, t in edges:
            if r == ':name' and gold_amr.nodes[t] == 'name':
                return None
            if r in [':polarity', ':mode']:
                is_dependent = True

        root = gold_amr.findSubGraph(tok_alignment).root
        if gold_amr.nodes[root] not in entities_with_preds and not is_dependent:                                                                                 
            return None

        new_id = None
        for s, r, t in edges:
            if s not in self.built_gold_nodeids:
                new_id = s
                break
            if t not in self.built_gold_nodeids:
                new_id = t
                break
            
        if new_id != None:

            self.built_gold_nodeids.append(new_id)
            self.nodeid_to_gold_nodeid.setdefault(machine.new_node_id, []).append(new_id)
            new_node = gold_amr.nodes[new_id]

            if self.copy_lemma_action:
                lemma = machine.get_current_token(lemma=True)
                if lemma == new_node:
                    action = 'COPY_LEMMA'
                elif f'{lemma}-01' == new_node:
                    action = 'COPY_SENSE01'
                else:
                    action = f'PRED({new_node})'
            else:
                action = f'PRED({new_node})'

            return action
                
        return None

    def try_entity(self):
        """
        Check if the next action is ENTITY.
        TryENTITY before tryPRED.

        If
        1) aligned to more than 1 nodes, and
        2) there are edges in the aligned subgraph, and then
        3) take the source nodes in the aligned subgraph altogether.
        """
        machine = self.machine
        gold_amr = self.gold_amr

        tok_id = machine.tok_cursor

        # to avoid subgraph ENTITY after named entities
        if tok_id in machine.entity_tokenids:
            return None

        # NOTE currently do not allow multiple ENTITY here on a single token
        if machine.current_node_id in machine.entities:
            return None

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return None

        # check if there is any edge with the aligned nodes
        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return None

        # check if named entity case: (entity_category, ':name', 'name')
        # no need, since named entity check happens first

        is_dependent = False
        is_named = False
        for s, r, t in edges:
            if r == ':name' and gold_amr.nodes[t] == 'name':
                is_named = True
            if r in [':polarity', ':mode']:
                is_dependent = True

        root = gold_amr.findSubGraph(tok_alignment).root
        if not is_named and ( gold_amr.nodes[root] in entities_with_preds or is_dependent):
            return None

        gold_nodeids = [n for n in tok_alignment if any(s == n for s, r, t in edges)]
        new_nodes = ','.join([gold_amr.nodes[n] for n in gold_nodeids])

        action = f'ENTITY({new_nodes})'

        self.built_gold_nodeids.extend(gold_nodeids)
        self.nodeid_to_gold_nodeid.setdefault(machine.new_node_id, []).extend(gold_nodeids)

        return action

    def try_pred(self):
        """
        Check if the next action is PRED, COPY_LEMMA, COPY_SENSE01.

        If
        1) the current token is aligned to a single node, or multiple nodes? (figure out)
        2) the aligned node has not been predicted yet
        """
        machine = self.machine
        gold_amr = self.gold_amr

        tok_id = machine.tok_cursor

        if tok_id == machine.tokseq_len - 1:
            # never do PRED(<ROOT>) currently, as the root node is automatically added at the beginning
            # NOTE to change this behavior, we need to be careful about the root node id which should be -1 now
            #      that is also massively used in postprocessing to find/add root.
            return None

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)    # NOTE we make all token ids positive natural index

        # check if the alignment is empty
        # no need since the REDUCE check happens first

        if len(tok_alignment) == 1:
            gold_nodeid = tok_alignment[0]
        else:
            # TODO check when this happens -> should we do multiple PRED?
            gold_nodeid = gold_amr.findSubGraph(tok_alignment).root

        # TODO for multiple PREDs, we need to do a for loop here

        # check if the node has been constructed, for multiple PREDs
        if gold_nodeid not in self.built_gold_nodeids:
            self.built_gold_nodeids.append(gold_nodeid)
            self.nodeid_to_gold_nodeid.setdefault(machine.new_node_id, []).append(gold_nodeid)

            new_node = gold_amr.nodes[gold_nodeid]

            if self.copy_lemma_action:
                lemma = machine.get_current_token(lemma=True)
                if lemma == new_node:
                    action = 'COPY_LEMMA'
                elif f'{lemma}-01' == new_node:
                    action = 'COPY_SENSE01'
                else:
                    action = f'PRED({new_node})'
            else:
                action = f'PRED({new_node})'

            return action

        else:
            return None

    def try_dependent(self):
        """
        Check if the next action is DEPENDENT.

        If
        1) the aligned node has been predicted already
        2)

        Only for :polarity and :mode, if an edge and node is aligned
        to this token in the gold amr but does not exist in the predicted amr,
        the oracle adds it using the DEPENDENT action.
        """
        machine = self.machine
        gold_amr = self.gold_amr

        tok_id = machine.tok_cursor
        node_id = machine.current_node_id

        if node_id is None:    # NOTE if node_id could be 0, 'if not node_id' would cause a bug
            # the node has not been built at current step
            return None

        # NOTE this doesn't work for ENTITY now, as the mapping from ENTITY node is only to the source nodes in the
        #      aligned subgraph, whereas for the DEPENDENT we are checking the target nodes in the subgraph
        # gold_nodeids = self.nodeid_to_gold_nodeid[node_id]
        # gold_nodeids = list(set(gold_nodeids))    # just in case

        gold_nodeids = gold_amr.alignmentsToken2Node(tok_id + 1)

        # below is coupled with the PRED checks? and also the ENTITY
        if len(gold_nodeids) == 1:
            gold_nodeid = gold_nodeids[0]
        else:
            gold_nodeid = gold_amr.findSubGraph(gold_nodeids).root

        for s, r, t in gold_amr.edges:
            if s == gold_nodeid and r in [':polarity', ':mode']:
                if (node_id, r) in [(e[0], e[1]) for e in machine.amr.edges]:
                    # to prevent same DEPENDENT added twice, as each time we scan all the possible edges
                    continue
                if t not in gold_nodeids and (t in gold_amr.alignments and gold_amr.alignments[t]):
                    continue

                self.built_gold_nodeids.append(t)
                # NOTE this might affect the next DEPEDENT check, but is fine if we always use subgraph root
                self.nodeid_to_gold_nodeid.setdefault(node_id, []).append(t)

                new_edge = r[1:] if r.startswith(':') else r
                new_node = gold_amr.nodes[t]
                action = f'DEPENDENT({new_node},{new_edge})'
                return action

        return None

    def try_arcs(self):
        """
        Get the arcs that involve the current token aligned node.

        If
        1) currently is on a node that was just constructed
        2) there are edges that have not been built with this node
        """
        machine = self.machine

        node_id = machine.current_node_id

        if node_id is None:    # NOTE if node_id could be 0, 'if not node_id' would cause a bug
            # the node has not been built at current step
            return None

        #for act_id, act_node_id in enumerate(machine.actions_to_nodes):
        for act_id, act_node_id in reversed(list(enumerate(machine.actions_to_nodes))):
            if act_node_id is None:
                continue
            # for multiple nodes out of one token --> need to use node id to check edges
            arc = self.get_arc(act_node_id, node_id)
            if arc is None:
                continue
            arc_name, arc_label = arc

            # avoid repetitive edges
            if arc_name == 'LA':
                if (node_id, arc_label, act_node_id) in machine.amr.edges:
                    continue
            if arc_name == 'RA':
                if (act_node_id, arc_label, node_id) in machine.amr.edges:
                    continue

            # pointer value
            arc_pos = act_id

            return f'{arc_name}({arc_pos},{arc_label})'

        return None

    def get_arc(self, node_id1, node_id2):
        """
        Get the arcs between node with `node_id1` and node with `node_id2`.
        RA if there is an edge `node_id1` --> `node_id2`
        LA if there is an edge `node_id2` <-- `node_id2`
        Thus the order of inputs matter. (could also change to follow strict orders between these 2 ids)

        # TODO could there be more than one edges?
        #      currently we only return the first one.
        """
        gold_amr = self.gold_amr

        # get the node ids in the gold AMR graph
        nodes1 = self.nodeid_to_gold_nodeid[node_id1]
        nodes2 = self.nodeid_to_gold_nodeid[node_id2]

        if not isinstance(nodes1, list):
            nodes1 = [nodes1]

        if not isinstance(nodes2, list):
            nodes2 = [nodes2]

        if not nodes1 or not nodes2:
            return None

        # convert to single node aligned to each of these two tokens
        if len(nodes1) > 1:
            # get root of subgraph aligned to token 1
            node1 = gold_amr.findSubGraph(nodes1).root
        else:
            node1 = nodes1[0]
        if len(nodes2) > 1:
            # get root of subgraph aligned to token 2
            node2 = gold_amr.findSubGraph(nodes2).root
        else:
            node2 = nodes2[0]

        # find edges
        for s, r, t in gold_amr.edges:
            if node1 == s and node2 == t:
                return ('RA', r)
            if node1 == t and node2 == s:
                return ('LA', r)

        return None


def run_oracle(gold_amrs, copy_lemma_action, multitask_words):

    # Initialize lemmatizer as this is slow
    lemmatizer = get_spacy_lemmatizer()

    # This will store the oracle stats
    statistics = {
        'sentence_tokens': [],
        'oracle_actions': [],
        'oracle_amr': [],
        'rules': {
            # Will store count of PREDs given pointer position
            'possible_predicates': defaultdict(lambda: Counter())
        }
    }
    pred_re = re.compile(r'^PRED\((.*)\)$')

    # Process AMRs one by one
    for sent_idx, gold_amr in tqdm(enumerate(gold_amrs), desc='Oracle'):

        # TODO: See if we can remove this part
        gold_amr = gold_amr.copy()
        gold_amr = preprocess_amr(gold_amr)

        # Initialize oracle builder
        oracle_builder = AMROracleBuilder(gold_amr, lemmatizer, copy_lemma_action, multitask_words)
        # build the oracle actions sequence
        actions = oracle_builder.build_oracle_actions()

        # store data
        statistics['sentence_tokens'].append(oracle_builder.tokens)
        # do not write CLOSE action at the end;
        # CLOSE action is internally managed, and treated same as <eos> in training
        statistics['oracle_actions'].append(actions[:-1])
        statistics['oracle_amr'].append(oracle_builder.machine.amr.toJAMRString())
        # pred rules
        for idx, action in enumerate(actions):
            if pred_re.match(action):
                node_name = pred_re.match(action).groups()[0]
                token = oracle_builder.machine.actions_tokcursor[idx]
                statistics['rules']['possible_predicates'][token].update(node_name)

    return statistics


def main():

    # Argument handling
    args = argument_parser()
    global entities_with_preds
    entities_with_preds = args.in_pred_entities.split(",")

    # Load AMR (replace some unicode characters)
    # TODO: unicode fixes and other normalizations should be applied more
    # transparently
    print(f'Reading {args.in_amr}')
    corpus = read_amr(args.in_amr, unicode_fixes=True)
    gold_amrs = corpus.amrs
    # sanity check AMRS
    print_corpus_info(gold_amrs)
    sanity_check_amr(gold_amrs)

    # Load propbank if provided
    # TODO: Use here XML propbank reader instead of txt reader
    propbank_args = None
    if args.in_propbank_args:
        propbank_args = read_propbank(args.in_propbank_args)

    # read/write multi-task (labeled shift) action
    # TODO: Add conditional if here
    multitask_words = process_multitask_words(
        [list(amr.tokens) for amr in gold_amrs],
        args.multitask_max_words,
        args.in_multitask_words,
        args.out_multitask_words,
        add_root=True
    )

    # run the oracle for the entire corpus
    stats = run_oracle(gold_amrs, args.copy_lemma_action, multitask_words)

    # print stats about actions
    sanity_check_actions(stats['sentence_tokens'], stats['oracle_actions'])

    # Save statistics
    write_tokenized_sentences(
        args.out_actions,
        stats['oracle_actions'],
        separator='\t'
    )
    write_tokenized_sentences(
        args.out_sentences,
        stats['sentence_tokens'],
        separator='\t'
    )
    # State machine stats for this sentence
    if args.out_rule_stats:
        with open(args.out_rule_stats, 'w') as fid:
            fid.write(json.dumps(stats['rules']))


if __name__ == '__main__':
    main()
