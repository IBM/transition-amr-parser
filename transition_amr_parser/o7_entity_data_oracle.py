import json
import argparse
from collections import Counter, defaultdict

from tqdm import tqdm

from transition_amr_parser.utils import print_log
from transition_amr_parser.io import writer, read_propbank, read_amr
from transition_amr_parser.o7_entity_state_machine import (
    AMRStateMachine,
    get_spacy_lemmatizer,
    entity_rule_stats,
    entity_rule_totals,
    entity_rule_fails
)


"""
This algorithm contains heuristics for generating linearized action sequences for AMR graphs in a rule based way.
The parsing algorithm is transition-based combined with pointers for long distance arcs.

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

    #
    args = parser.parse_args()

    return args


def preprocess_amr(gold_amr, add_unaligned, included_unaligned, root_id=-1):

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


class AMR_Oracle:

    def __init__(self, verbose=False):
        self.amrs = []
        self.gold_amrs = []
        self.transitions = []
        self.verbose = verbose

        # predicates
        # TODO change names to use underscore
        self.preds2Ints = {}
        self.possible_predicates = {}

        self.new_edge = ''
        self.new_node = ''
        self.entity_type = ''
        self.dep_id = None

        self.swapped_words = {}

        self.possibleEntityTypes = Counter()

        # DEBUG
        # self.copy_rules = False

        # node id to gold AMR node id mapping (used for multiple PRED from a single token)
        self.node2goldnode = {}
        self.node2goldnode[-1] = -1    # for root node
        self.goldnode = None    # gold node id

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
            transitions[-1].apply_actions(actions)
        self.transitions = transitions

    def run_oracle(self, gold_amrs,
                   out_oracle=None,
                   out_amr=None,
                   out_sentences=None,
                   out_actions=None,
                   copy_lemma_action=False,
                   root_id=-1,            # root node id to be added to gold AMR graph
                   propbank_args=None,    # TODO from old code; not used anywhere in the function
                   out_rule_stats=None,
                   add_unaligned=0,        # number of unaligned tokens at the end
                   no_whitespace_in_actions=False,
                   multitask_words=None,
                   addnode_count_cutoff=None):

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
        # TODO deal with this later
        self.stats = {
            'possible_predicates': Counter(),
            'action_vocabulary': Counter(),
            'addnode_counts': Counter()
        }

        # unaligned tokens
        included_unaligned = ['-', 'and', 'multi-sentence', 'person', 'cause-01',
                              'you', 'more', 'imperative', '1', 'thing']

        # initialize spacy lemmatizer out of the sentence loop for speed
        spacy_lemmatizer = None
        if copy_lemma_action:
            spacy_lemmatizer = get_spacy_lemmatizer()

        # Loop over gold AMRs
        for sent_idx, gold_amr in tqdm(
            enumerate(self.gold_amrs),
            desc='computing oracle',
            total=len(self.gold_amrs)
        ):

            if self.verbose:
                print("New Sentence " + str(sent_idx) + "\n\n\n")

            # TODO the alignments are outside of the sentence boundary for this sentence in training set
            # TODO check why this is the case in data loading
            # if gold_amr.tokens[0] == 'Among' and gold_amr.tokens[1] == 'common':
            #     import pdb; pdb.set_trace()

            # TODO: Describe what is this pre-processing
            gold_amr = preprocess_amr(gold_amr, add_unaligned, included_unaligned, root_id)

            # Initialize state machine
            tr = AMRStateMachine(
                gold_amr.tokens,
                verbose=self.verbose,
                add_unaligned=add_unaligned,
                spacy_lemmatizer=spacy_lemmatizer
            )
            self.transitions.append(tr)
            self.amrs.append(tr.amr)

            # Loop over potential actions
            # NOTE "<ROOT>" token at last position is in the confirmed node list from the beginning, so no prediction
            # for it here; the ending sequence is always [... SHIFT CLOSE] or [... LA(pos,'root') SHIFT CLOSE]
            while not tr.is_closed:

                if self.tryREDUCE(tr, tr.amr, gold_amr):
                    actions = ['REDUCE']

                elif self.tryMERGE(tr, tr.amr, gold_amr):
                    actions = ['MERGE']

                elif self.tryENTITY(tr, tr.amr, gold_amr):

                    actions, gold_node_ids = self.get_named_entities(tr, tr.amr, gold_amr)

                    if len(actions) > 3:
                        import pdb; pdb.set_trace()

                    if actions:
                        self.goldnode = gold_node_ids
                    else:
                        actions = [f'ENTITY({self.entity_type})']

                # note: need to put tryPRED after tryENTITY, since ENTITY does not work on single alignment
                # note: and PRED is broader
                elif self.tryPRED(tr, tr.amr, gold_amr):
                    if copy_lemma_action:
                        lemma = tr.get_current_token(lemma=True)
                        if lemma == self.new_node:
                            actions = ['COPY_LEMMA']
                        elif f'{lemma}-01' == self.new_node:
                            actions = ['COPY_SENSE01']
                        else:
                            actions = [f'PRED({self.new_node})']
                            # record statistics of word-to-PRED-labels for generation restriction
                            token = tr.get_current_token(lemma=False)
                            if token not in self.possible_predicates:
                                self.possible_predicates[token] = Counter()
                            self.possible_predicates[token][self.new_node] += 1
                    else:
                        actions = [f'PRED({self.new_node})']
                        # record statistics of word-to-PRED-labels for generation restriction
                        token = tr.get_current_token(lemma=False)
                        if token not in self.possible_predicates:
                            self.possible_predicates[token] = Counter()
                        self.possible_predicates[token][self.new_node] += 1

                elif self.tryDEPENDENT(tr, tr.amr, gold_amr):
                    edge = self.new_edge[1:] \
                        if self.new_edge.startswith(':') else self.new_edge
                    # 'SHIFT' will block all the other edges: should not be set here
                    # actions = [f'DEPENDENT({self.new_node},{edge})', 'SHIFT']    # this was a bug!
                    actions = [f'DEPENDENT({self.new_node},{edge})']
                    self.dep_id = None
                # TODO clean up the logic for a better one
                else:
                    # debug difference from the refactored code
                    # if sent_idx == 995 and tr.time_step >= 46:
                    #     import pdb; pdb.set_trace()

                    actions = self.get_previous_arcs(tr, tr.amr, gold_amr)
                    if actions:
                        actions.append('SHIFT')
                    else:
                        actions = ['SHIFT']
                        # TODO clean up the logic here, whether to predict the ROOT at last position
                        # TODO whether to add CLOSE explicitly, and ensure it runs so as to get complete AMR graph

                for i, action in enumerate(actions):
                    action_label = action.split('(')[0]

                    # check action has not invalid chars and normalize
                    # TODO: --no-whitespace-in-actions being deprecated
                    if no_whitespace_in_actions and action_label == 'PRED':
                        assert '_' not in action, \
                            "--no-whitespace-in-actions prohibits use of _ in actions"
                        if ' ' in action_label:
                            action = action.replace(' ', '_')

                    # Add prediction to top of the buffer
                    if action == 'SHIFT' and multitask_words is not None:
                        action = label_shift(tr, multitask_words)

                    actions[i] = action

                # APPLY ACTION
                tr.apply_actions(actions, node2goldnode=self.node2goldnode, goldnode=self.goldnode)
                self.goldnode = None

            # Close machine
            tr.CLOSE(
                training=True,
                gold_amr=gold_amr,
                use_addnonde_rules=use_addnode_rules    # TODO not used flag
            )
            # TODO this should be related to the AMR machine
            self.node2goldnode = {}
            self.node2goldnode[-1] = -1    # for root node

            # update files
            if out_oracle:
                # to avoid printing
                oracle_write(str(tr))
            # JAMR format AMR
            amr_write(tr.amr.toJAMRString())
            # Tokens and actions
            # extra tag to be reduced at start
            tokens = tr.amr.tokens
            actions = tr.actions[:-1]    # do not write CLOSE action at the end; CLOSE action is internally managed

            # Update action count
            self.stats['action_vocabulary'].update(actions)
            del gold_amr.nodes[-1]
            # addnode_actions = [a for a in actions if a.startswith('ADDNODE')]
            addnode_actions = [a for a in actions if a.startswith('ENTITY')]
            self.stats['addnode_counts'].update(addnode_actions)

            # separator
            if no_whitespace_in_actions:
                sep = " "
            else:
                sep = "\t"
            tokens = sep.join(tokens)
            actions = sep.join(actions)
            # Write
            sentence_write(tokens)
            actions_write(actions)

        print_log("oracle", "Done")

        # close files if open
        oracle_write()
        amr_write()
        sentence_write()
        actions_write()

        self.labelsO2idx = {'<pad>': 0}
        self.labelsA2idx = {'<pad>': 0}
        self.pred2idx = {'<pad>': 0}
        self.action2idx = {'<pad>': 0}

        for tr in self.transitions:
            for a in tr.actions:
                a = AMRStateMachine.read_action(a)[0]
                self.action2idx.setdefault(a, len(self.action2idx))
            # TODO deal with these
            # for p in tr.predicates:
            #     self.pred2idx.setdefault(p, len(self.pred2idx))
            # for l in tr.labels:
            #     self.labelsO2idx.setdefault(l, len(self.labelsO2idx))
            # for l in tr.labelsA:
            #     self.labelsA2idx.setdefault(l, len(self.labelsA2idx))

        self.stats["action2idx"] = self.action2idx
        self.stats["pred2idx"] = self.pred2idx
        self.stats["labelsO2idx"] = self.labelsO2idx
        self.stats["labelsA2idx"] = self.labelsA2idx

        # Compute the word dictionary

        self.char2idx = {'<unk>': 0}
        self.word2idx = {'<unk>': 0, '<eof>': 1, '<ROOT>': 2, '<unaligned>': 3}
        self.node2idx = {}
        self.word_counter = Counter()

        for amr in self.gold_amrs:
            for tok in amr.tokens:
                self.word_counter[tok] += 1
                self.word2idx.setdefault(tok, len(self.word2idx))
                for ch in tok:
                    self.char2idx.setdefault(ch, len(self.char2idx))
            for n in amr.nodes:
                self.node2idx.setdefault(amr.nodes[n], len(self.node2idx))

        self.stats["char2idx"] = self.char2idx
        self.stats["word2idx"] = self.word2idx
        self.stats["node2idx"] = self.node2idx
        self.stats["word_counter"] = self.word_counter

        self.stats['possible_predicates'] = self.possible_predicates

        if addnode_count_cutoff:
            self.stats['addnode_blacklist'] = [
                a
                for a, c in self.stats['addnode_counts'].items()
                if c <= addnode_count_cutoff
            ]
            num_addnode_blackl = len(self.stats['addnode_blacklist'])
            num_addnode = len(self.stats['addnode_counts'])
            print(f'{num_addnode_blackl}/{num_addnode} blacklisted ADDNODES')
            del self.stats['addnode_counts']

        # State machine stats for this senetnce
        if out_rule_stats:
            with open(out_rule_stats, 'w') as fid:
                fid.write(json.dumps(self.stats))

    def tryREDUCE(self, transitions, amr, gold_amr):
        """
        Check if the next action is REDUCE.

        If
        1) there is nothing aligned to a token
        then return True.
        """
        # TODO change name for transitions (maybe state)
        node_id = transitions.current_node_id
        if node_id in transitions.is_confirmed:
            return False

        tok_id = transitions.tok_cursor
        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)    # NOTE the index + 1
        if len(tok_alignment) == 0:
            return True
        else:
            return False

    def tryMERGE(self, transitions, amr, gold_amr):
        """
        Check if the next action is MERGE.

        If
        1) the current and the next token have the same node alignment
        then return True.
        """
        if transitions.tok_cursor < transitions.tokseq_len - 1:
            cur = transitions.tok_cursor
            nxt = transitions.tok_cursor + 1
            cur_alignment = gold_amr.alignmentsToken2Node(cur + 1)
            try:
                nxt_alignment = gold_amr.alignmentsToken2Node(nxt + 1)
            except:
                import pdb; pdb.set_trace()
            if not cur_alignment or not nxt_alignment:
                return False
            # If both tokens are mapped to same node or overlap
            if cur_alignment == nxt_alignment:
                return True
            if set(cur_alignment).intersection(set(nxt_alignment)):
                return True
            return False
        else:
            return False

    def get_named_entities(self, transitions, amr, gold_amr):
        """
        Get the named entity sequences from the current surface token (segments).
        E.g.
        a) for one entity
        ENTITY('name') PRED('city') LA(pos,':name')
        b) for two entities with same surface tokens
        ENTITY('name') PRED('city') LA(pos,':name') PRED('city') LA(pos,':name')
        c) for two entities with two surface tokens
        ENTITY('name') PRED('city') LA(pos,':name') ENTITY('name') PRED('city') LA(pos,':name')

        TODO currently doesn't consider DEPENDENT inside; see if it is needed by checking the data
        """
        tok_id = transitions.tok_cursor
        node_id = transitions.current_node_id

        # TODO currently ignore these, as we do the subsequence all at once here
        if node_id in transitions.entities:
            # do not do twice
            return False

        if node_id in transitions.is_confirmed:
            return False

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return False

        # check if we should MERGE instead
        # no need, since tryMERGE happens first

        # check if there is any edge with the aligned nodes
        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return False

        # check if we should use DEPENDENT instead
        # no need, since DEPENDENT happens when the node is added

        # separate named entity case: (entity_category, ':name', 'name')
        entity_edges = []
        name_node_ids = []
        for s, r, t in edges:
            if r == ':name' and gold_amr.nodes[t] == 'name':
                entity_edges.append((s, r, t))
                name_node_ids.append(t)

        # debug: check if there could be more than one entity from a single token
        # if len(entity_edge) > 1:
        #     edges_named = [(gold_amr.nodes[e[0]], e[1], gold_amr.nodes[e[2]]) for e in edges]
        #     print('-' * 80)
        #     print(transitions.get_current_token())
        #     print(edges_named)
        #     print('-' * 80)
        #     import pdb; pdb.set_trace()

        # debug: check if there could be more than one node with name "name" aligned to a single token
        # if len(set(name_node_ids)) > 1:
        #     edges_named = [(gold_amr.nodes[e[0]], e[1], gold_amr.nodes[e[2]]) for e in edges]
        #     print('-' * 80)
        #     print(transitions.get_current_token())
        #     print(edges_named)
        #     print('-' * 80)
        #     import pdb
        #     pdb.set_trace()

        named_entity_actions = []
        gold_node_ids = []
        if name_node_ids:
            added_name_node_ids = set()
            pos = len(transitions.actions)
            for nid, (s, r, t) in zip(name_node_ids, entity_edges):
                if nid not in added_name_node_ids:
                    pos += len(named_entity_actions)
                    named_entity_actions.append('ENTITY(name)')
                    added_name_node_ids.add(nid)
                    gold_node_ids.append([nid])
                named_entity_actions.append(f'PRED({gold_amr.nodes[s]})')
                gold_node_ids.append([s])
                named_entity_actions.append(f'LA({pos},:name)')
                gold_node_ids.append([None])

        return named_entity_actions, gold_node_ids

    def tryENTITY(self, transitions, amr, gold_amr):
        """
        Check if the next action is ENTITY.
        TyeENTITY before tryPRED.

        If
        1) aligned to more than 1 nodes, and ?
        2) the aligned node has not been predicted yet
        then return True, pass the entity type via `self.entity_type`.
        """
        tok_id = transitions.tok_cursor
        node_id = transitions.current_node_id

        if node_id in transitions.entities:
            # do not do twice
            return False

        if node_id in transitions.is_confirmed:
            return False

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # check if alignment empty (or singleton)
        if len(tok_alignment) <= 1:
            return False

        # check if we should MERGE instead
        # no need, since tryMERGE happens first

        # check if there is any edge with the aligned nodes
        edges = gold_amr.findSubGraph(tok_alignment).edges
        if not edges:
            return False

        # check if we should use DEPENDENT instead
        # no need, since DEPENDENT happens when the node is added

        # separate named entity case: (entity_category, ':name', 'name')
        entity_edge = []
        name_node_ids = []
        for s, r, t in edges:
            if r == ':name' and gold_amr.nodes[t] == 'name':
                entity_edge.append((s, r, t))
                name_node_ids.append(t)

        # debug: check if there could be more than one entity from a single token
        # if len(entity_edge) > 1:
        #     edges_named = [(gold_amr.nodes[e[0]], e[1], gold_amr.nodes[e[2]]) for e in edges]
        #     print('-' * 80)
        #     print(transitions.get_current_token())
        #     print(edges_named)
        #     print('-' * 80)
        #     import pdb; pdb.set_trace()

        # debug: check if there could be more than one node with name "name" aligned to a single token
        # if len(set(name_node_ids)) > 1:
        #     edges_named = [(gold_amr.nodes[e[0]], e[1], gold_amr.nodes[e[2]]) for e in edges]
        #     print('-' * 80)
        #     print(transitions.get_current_token())
        #     print(edges_named)
        #     print('-' * 80)
        #     import pdb; pdb.set_trace()

        # what is the rule here? -->
        # 1) find all the aligned nodes that do not have any outcoming edges in the aligned subgraph
        # 2) exclude the above nodes
        #
        # final_nodes = [n for n in tok_alignment if not any(s == n for s, r, t in edges)]
        # new_nodes = [gold_amr.nodes[n] for n in tok_alignment if n not in final_nodes]
        #
        # this is equivalent to
        new_nodes = [gold_amr.nodes[n] for n in tok_alignment if any(s == n for s, r, t in edges)]

        self.goldnode = [n for n in tok_alignment if any(s == n for s, r, t in edges)]

        self.entity_type = ','.join(new_nodes)
        self.possibleEntityTypes[self.entity_type] += 1

        # debug: look at the entity case
        # edges_named = [(gold_amr.nodes[e[0]], e[1], gold_amr.nodes[e[2]]) for e in edges]
        # print('-' * 80)
        # print(transitions.get_current_token())
        # print(edges_named)
        # print(new_nodes)
        # print('-' * 80)
        # import pdb; pdb.set_trace()

        return True

    def tryPRED(self, transitions, amr, gold_amr):
        """
        Check if the next action is PRED, COPY_LEMMA, COPY_SENSE01.

        If
        1) the current token is aligned to a single node, or multiple nodes? (figure out)
        2) the aligned node has not been predicted yet
        then return True, and set the gold label via self.node.
        """
        tok_id = transitions.tok_cursor
        node_id = transitions.current_node_id

        if node_id in transitions.is_confirmed:
            # do not generate node from a single token twice
            return False

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)    # NOTE we make all token ids positive natural index
        if len(tok_alignment) == 1:
            gold_id = tok_alignment[0]
        else:
            gold_id = gold_amr.findSubGraph(tok_alignment).root
        # TODO a better to pass back new node label instead of class attribute
        self.new_node = gold_amr.nodes[gold_id]

        self.goldnode = gold_id
        return True

    def tryDEPENDENT(self, transitions, amr, gold_amr):
        """
        Check if the next action is DEPENDENT.

        If
        1) the aligned node has been predicted already
        2)
        then return True, and set the gold edge label and node label via self.new_edge and self.new_node.

        Only for :polarity and :mode, if an edge and node is aligned
        to this token in the gold amr but does not exist in the predicted amr,
        the oracle adds it using the DEPENDENT action.
        """
        tok_id = transitions.tok_cursor
        node_id = transitions.current_node_id

        if node_id not in transitions.is_confirmed:
            return False

        tok_alignment = gold_amr.alignmentsToken2Node(tok_id + 1)

        # TODO think if this is necessary
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
                if (node_id, r) in [(e[0], e[1]) for e in amr.edges]:
                    # to prevent same DEPENDENT added twice, as each time we scan all the possible edges
                    continue
                if t not in tok_alignment and (t in gold_amr.alignments and gold_amr.alignments[t]):
                    continue
                self.new_edge = r
                self.new_node = gold_amr.nodes[t]
                self.goldnode = t
                return True

        return False

    def get_previous_arcs(self, transitions, amr, gold_amr):
        """
        Get all the arcs the involve the current token aligned node.

        If
        1)
        2)
        then return True, and pass the action pointer and edge label via self.arc_pos and self.new_edge.
        """
        tok_id = transitions.tok_cursor
        node_id = transitions.current_node_id

        if node_id not in transitions.is_confirmed:
            return []

        # debug
        # if transitions.tokens[:2] == ['Some', 'people']:
        #     breakpoint()

        arcs = []
        for act_id, (act_name, act_node_id) in enumerate(zip(transitions.actions, transitions.actions_to_nodes)):
            if act_node_id is None:
                continue
            # arc = self.get_arc(gold_amr, transitions.nodeid_to_tokid[act_node_id], tok_id)
            # for multiple nodes out of one token --> need to use node id to check edges
            arc = self.get_arc(gold_amr, act_node_id, node_id)
            if arc is None:
                continue
            arc_name, arc_label = arc

            # avoid repetitive edges
            if arc_name == 'LA':
                if (node_id, arc_label, act_node_id) in transitions.amr.edges:
                    continue
            if arc_name == 'RA':
                if (act_node_id, arc_label, node_id) in transitions.amr.edges:
                    continue

            arc_pos = act_id
            # if (transitions.tokens ==
            #     ['here', ',', 'you', 'can', 'come', 'up', 'close', 'with', 'the', 'stars', 'in', 'your', 'mind', '.', '<ROOT>']) \
            #     or (transitions.tokens ==
            #     ['Promotion', 'of', 'Hong', 'Kong', 'Disneyland', 'has', 'long', 'since', 'begun', '.', '<ROOT>']) \
            # and arc_label == 'root':
            #     import pdb; pdb.set_trace()

            arcs.append(f'{arc_name}({arc_pos},{arc_label})')
        # TODO the return value to be better managed
        return arcs

    def get_arc(self, gold_amr, node_id1, node_id2):
        """
        Get the arcs between token with `tok_id1` and token with `tok_id2`.
        RA if there is an edge `tok_id1` --> `tok_id2`
        LA if there is an edge `tok_id2` <-- `tok_id2`
        Thus the order of inputs matter. (could also change to follow strict orders between these 2 ids)

        #TODO could there be more than one edges?
        currently we only return the first one.
        """
        nodes1 = self.node2goldnode[node_id1]
        nodes2 = self.node2goldnode[node_id2]

        if not isinstance(nodes1, list):
            nodes1 = [nodes1]

        if not isinstance(nodes2, list):
            nodes2 = [nodes2]

        # import pdb; pdb.set_trace()

        # nodes1 = gold_amr.alignmentsToken2Node(tok_id1 + 1)
        # nodes2 = gold_amr.alignmentsToken2Node(tok_id2 + 1)

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

    # def get_arc(self, gold_amr, tok_id1, tok_id2):
    #     """
    #     Get the arcs between token with `tok_id1` and token with `tok_id2`.
    #     RA if there is an edge `tok_id1` --> `tok_id2`
    #     LA if there is an edge `tok_id2` <-- `tok_id2`
    #     Thus the order of inputs matter. (could also change to follow strict orders between these 2 ids)

    #     #TODO could there be more than one edges?
    #     currently we only return the first one.
    #     """
    #     nodes1 = gold_amr.alignmentsToken2Node(tok_id1 + 1)
    #     nodes2 = gold_amr.alignmentsToken2Node(tok_id2 + 1)

    #     if not nodes1 or not nodes2:
    #         return None

    #     # convert to single node aligned to each of these two tokens
    #     if len(nodes1) > 1:
    #         # get root of subgraph aligned to token 1
    #         node1 = gold_amr.findSubGraph(nodes1).root
    #     else:
    #         node1 = nodes1[0]
    #     if len(nodes2) > 1:
    #         # get root of subgraph aligned to token 2
    #         node2 = gold_amr.findSubGraph(nodes2).root
    #     else:
    #         node2 = nodes2[0]

    #     # find edges
    #     for s, r, t in gold_amr.edges:
    #         if node1 == s and node2 == t:
    #             return ('RA', r)
    #         if node1 == t and node2 == s:
    #             return ('LA', r)

    #     return None


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


def main():

    # Argument handling
    args = argument_parser()

    # Load AMR (replace some unicode characters)
    corpus = read_amr(args.in_amr, unicode_fixes=True)

    # Load propbank
    propbank_args = None
    if args.in_propbank_args:
        propbank_args = read_propbank(args.in_propbank_args)

    # read/write multi-task (labeled shift) action
    multitask_words = process_multitask_words(
        [list(amr.tokens) for amr in corpus.amrs],
        args.multitask_max_words,
        args.in_multitask_words,
        args.out_multitask_words,
        add_root=True
    )

    # TODO: At the end, an oracle is just a parser with oracle info. This could
    # be turner into a loop similar to parser.py (ore directly use that and a
    # AMROracleParser())
    print_log("amr", "Processing oracle")
    oracle = AMR_Oracle(verbose=args.verbose)
    oracle.run_oracle(
        corpus.amrs,
        out_oracle=args.out_oracle,
        out_amr=args.out_amr,
        out_sentences=args.out_sentences,
        out_actions=args.out_actions,
        out_rule_stats=args.out_rule_stats,
        propbank_args=propbank_args,
        add_unaligned=0,
        no_whitespace_in_actions=args.no_whitespace_in_actions,
        multitask_words=multitask_words,
        copy_lemma_action=args.copy_lemma_action,
        addnode_count_cutoff=args.addnode_count_cutoff
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
#         perc = sum(entity_rule_stats.values()) / \
#             sum(entity_rule_totals.values())
#         print_log('Totals:', f'{perc:.2f}')
        print_log('Totals:', 'Failed Entity Predictions:')


if __name__ == '__main__':
    main()
