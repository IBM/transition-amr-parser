from operator import itemgetter
import json
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
import re
from random import shuffle
from difflib import get_close_matches
from functools import wraps
# pip install penman spacy ipdb numpy
import numpy as np
try:
    # version?
    import spacy
    from spacy.tokens.doc import Doc
except ImportError as e:
    print('\nThe simple AMR aligner needs Spacy\n')
    raise e

# pip install matplotlib
from transition_amr_parser.plots import plot_graph
from transition_amr_parser.io import read_amr, write_neural_alignments
from transition_amr_parser.clbar import clbar
# for debugging
from ipdb import set_trace
# import warnings
# warnings.filterwarnings('error')


def memoize(method):
    """
    Store function output as function of decorator intercepted cache_key
    variable
    """

    memoized_results = {}

    @wraps(method)
    def memoized_method(*args, **kwargs):
        assert 'cache_key' in kwargs, \
            "If you use @memoize you need to provide cache_key (can be None)"
        cache_key = kwargs['cache_key']
        if cache_key is None:
            del kwargs['cache_key']
            return method(*args, **kwargs)
        del kwargs['cache_key']
        if cache_key not in memoized_results:
            memoized_results[cache_key] = method(*args, **kwargs)
        return memoized_results[cache_key]

    return memoized_method


# pronoun mapping
normalize_pronouns = {
    'i': 'i',
    'you': 'you',
    'he': 'he',
    'him': 'he',
    'her': 'she',
    'we': 'we',
    'us': 'we',
    'them': 'they',
    'their': 'they',
    'those': 'that',
    'me': 'i',
    'my': 'i',
    'our': 'we',
    'your': 'you',
    'his': 'he',
}

normalize_article = {
    'these': 'this'
}


class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)


try:
    lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except OSError:
    # Assume the problem was the spacy models were not downloaded
    from spacy.cli.download import download
    download('en_core_web_sm')
    lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
lemmatizer.tokenizer = NoTokenizer(lemmatizer.vocab)


def constrain_posterior(amr, align_posterior):
    '''
    amr:              transition_amr_parser.io.AMR
    align_posterior:  array.shape (len(amr.tokens), len(amr.nodes)) with
                      amr.nodes order for dim 1
    '''

    # get map from node id to posterior node index
    nid2idx = {nid: i for i, nid in enumerate(amr.nodes.keys())}
    # idx2nid = {v: k for k, v in nid2idx.items()}

    # expected positions
    exp_pos = (
        np.arange(len(amr.tokens))[:, None] * align_posterior
    ).sum(0)

    # Realign named entities
    for ner in get_ner_ids(amr):
        # all nodes inherit alignment posterior of the left-most :opN
        ref_id = sorted(ner['children_ids'], key=lambda x: x[1])[0][0]
        for nid in [ner['id'], ner['name_id']]:
            align_posterior[:, nid2idx[nid]] = \
                align_posterior[:, nid2idx[ref_id]]

    # Realign uniform posteriors by using child posterior
    max_p = align_posterior.max(0, keepdims=True)
    for node_pos, max_rep in enumerate((max_p == align_posterior).sum(0)):
        if max_rep > 1:
            node_id = list(amr.nodes.keys())[node_pos]
            children = amr.children(node_id)
            if len(children) > 1:
                # average with left-most (expected position) children
                left_child = sorted(
                    children,
                    key=lambda x: exp_pos[nid2idx[x[0]]]
                )[0]
                rid = nid2idx[left_child[0]]
                align_posterior[:, node_pos] += align_posterior[:, rid]
                align_posterior[:, node_pos] *= 0.5

    # Realign uniform posteriors by using parent posterior
    max_p = align_posterior.max(0, keepdims=True)
    for node_pos, max_rep in enumerate((max_p == align_posterior).sum(0)):
        if max_rep > 1:
            node_id = list(amr.nodes.keys())[node_pos]
            parents = amr.parents(node_id)
            if len(parents) > 1:
                # average with left-most (expected position) parents
                left_parent = sorted(
                    parents,
                    key=lambda x: exp_pos[nid2idx[x[0]]]
                )[0]
                rid = nid2idx[left_parent[0]]
                align_posterior[:, node_pos] += align_posterior[:, rid]
                align_posterior[:, node_pos] *= 0.5

    return align_posterior


def get_sparse_prob_indices(probs, alpha=0.0):
    """
    Here alpha is used to make a probability distribution sparse by
    limiting the maximum allowed relative decay between sorted
    probabilities
    """

    if probs.shape[0] == 1:
        return np.array([0])

    # sort the probabilities and compute the ratio from larger to smaller
    indices = np.argsort(probs)[::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_decay = 1 - probs[indices][1:] / probs[indices][:-1]
    # get first sorted index that does satisfy the threshold
    search = (ratio_decay > alpha).nonzero()[0]
    if search.shape[0] == 0:
        return indices
    else:
        cut_index = search[0]
        return indices[:cut_index + 1]


def get_ner_ids(amr):

    ners = []
    for node_id in amr.nodes.keys():
        if (
            amr.nodes[node_id] == 'name'
            and any([x[1] == ':name' for x in amr.parents(node_id)])
        ):
            # all nodes below the "name" node
            child_nodes = amr.children(node_id)
            # parent node of "name" node (NER tag)
            tag_id = [e[0] for e in amr.parents(node_id) if e[1] == ':name'][0]
            ner_alignments = [x for x in child_nodes if x[1].startswith(':op')]
            ner_alignments = sorted(ner_alignments, key=lambda x: x[1])
            ners.append({
                'id': tag_id,
                'name_id': node_id,
                'children_ids': ner_alignments
            })

    return ners


class AMRAligner():

    def __init__(self, rule_prior_strength=1, not_align_tokens=None,
                 smoothing=0.01, force_align_ner=False, ignore_nodes=None,
                 ignore_node_regex=None, node_by_token_counts=None):

        # Initialize empty or from load data
        self.node_by_token_counts = defaultdict(lambda: defaultdict(float))
        if node_by_token_counts is None:
            self.prev_node_by_token_counts = \
                defaultdict(lambda: defaultdict(float))
        else:
            self.prev_node_by_token_counts = node_by_token_counts

        self.rule_prior_strength = rule_prior_strength
        self.not_align_tokens = not_align_tokens
        self.smoothing = smoothing
        self.force_align_ner = force_align_ner
        self.ignore_nodes = ignore_nodes
        self.ignore_node_regex = ignore_node_regex

        # accumulators for training
        self.train_loglik = 0
        self.train_num_examples = 0

        # TODO: factor out cache
        self.memoize_rule_alignments = {}

    @classmethod
    def from_checkpoint(cls, checkpoint_json):

        with open(checkpoint_json) as fid:
            data = json.loads(fid.read())

        # enxure all data provided
        assert 'rule_prior_strength' in data
        assert 'not_align_tokens' in data
        assert 'smoothing' in data
        assert 'force_align_ner' in data
        assert 'ignore_nodes' in data
        assert 'ignore_node_regex' in data
        assert 'node_by_token_counts' in data

        # initialize counts as defaultdict
        node_by_token_counts = defaultdict(lambda: defaultdict(float))
        for token, node_counts in data['node_by_token_counts'].items():
            node_by_token_counts[token] = defaultdict(float, node_counts)

        return cls(
            rule_prior_strength=data['rule_prior_strength'],
            not_align_tokens=data['not_align_tokens'],
            smoothing=data['smoothing'],
            force_align_ner=data['force_align_ner'],
            ignore_nodes=data['ignore_nodes'],
            ignore_node_regex=[
                re.compile(x) for x in data['ignore_node_regex']
            ],
            node_by_token_counts=node_by_token_counts
        )

    def save(self, out_json):

        # convert into dicts
        node_by_token_counts = dict()
        for token, node_counts in self.prev_node_by_token_counts.items():
            node_by_token_counts[token] = dict(node_counts)

        # data definining the model
        data = {
            'rule_prior_strength': self.rule_prior_strength,
            'not_align_tokens': self.not_align_tokens,
            'smoothing': self.smoothing,
            'force_align_ner': self.force_align_ner,
            'ignore_nodes': self.ignore_nodes,
            'ignore_node_regex': [x.pattern for x in self.ignore_node_regex],
            'node_by_token_counts': node_by_token_counts
        }

        with open(out_json, 'w') as fid:
            fid.write(json.dumps(data))

    def update_counts(self, amr, cache_key=None):

        # Get posterior over alignments using graph topology and simple
        # alignment to define distribution. If estimates from last
        # iteration exist merge them with that info
        alignment_posterior, likelihood = \
            self.get_alignment_posterior(amr, cache_key)

        # Update counters
        self.train_loglik += np.log(likelihood).sum()
        self.train_num_examples += likelihood.shape[1]
        for node_pos, (nid, node_name) in enumerate(amr.nodes.items()):
            for token_pos, token_name in enumerate(amr.tokens):
                if alignment_posterior[token_pos, node_pos] > 0:
                    self.node_by_token_counts[token_name][node_name] \
                        += alignment_posterior[token_pos, node_pos]

    def update_parameters(self):
        """ assign current counts to previous and reset counter """
        # assign accumulated stats to model parameters
        self.prev_node_by_token_counts = self.node_by_token_counts
        # Update counters
        self.node_by_token_counts = defaultdict(lambda: defaultdict(float))
        self.train_loglik = 0
        self.train_num_examples = 0

    def get_alignment_likelihood(self, amr, cache_key=None, no_norm=False):

        # Start from rule-based alignments if solicited
        if self.rule_prior_strength > 0:
            nodeid2token = surface_aligner(
                amr.tokens, list(amr.nodes.items()), cache_key=cache_key)[0]
        else:
            nodeid2token = {}

        # Note down tokens that are aligned with probability 1 to a node
        # TODO: Not in use right now, eliminate?
        token_hard_aligned = dict()
        for nid, tokens in nodeid2token.items():
            if len(tokens) == 1:
                token_hard_aligned[tokens[0][0]] = nid

        # Compute node posterior given token prob (posterior predictive)
        node_token_counts_sent = np.zeros((len(amr.tokens), len(amr.nodes)))
        for node_pos, (nid, node_name) in enumerate(amr.nodes.items()):

            # Add rule or default smoothing as prior pseudocounts
            for (token_pos, _) in nodeid2token.get(nid, []):
                node_token_counts_sent[token_pos, node_pos] = \
                    self.rule_prior_strength

            # Add accumulated stats for each node, token pair plus smoothing
            for token_pos, token_name in enumerate(amr.tokens):
                node_token_counts_sent[token_pos, node_pos] += \
                    self.prev_node_by_token_counts[token_name][node_name] \
                    + self.smoothing

            # something should be aligned to that node
            if node_token_counts_sent[:, node_pos].sum() == 0:
                node_token_counts_sent[:, node_pos] = self.smoothing

        # normalize over nodes to make it a probability of generating a
        # sentence node given a sentence token p(y_j | x_{a_j})
        if no_norm:
            return node_token_counts_sent
        else:
            prob_node_by_token_sent = node_token_counts_sent \
                / node_token_counts_sent.sum(axis=1, keepdims=True)
            return prob_node_by_token_sent

    def get_alignment_prior(self, amr):

        # Add rule as prior pseudocounts
        token_counts_sent = np.ones(len(amr.tokens)) * self.smoothing
        for token_pos, token_name in enumerate(amr.tokens):
            if token_name in self.not_align_tokens:
                token_counts_sent[token_pos] = 0
            # token_counts_sent[token_pos] += \
            #    sum(node_by_token_counts[token_name].values())

        if token_counts_sent.sum() == 0:
            token_counts_sent[:] = self.smoothing

        prob_token = token_counts_sent / token_counts_sent.sum(keepdims=True)

        return prob_token

    def get_alignment_posterior(self, amr, cache_key=None):

        # Likelihoods of node y_j being produced by each token x_{a_j}
        # shape = (len(amr.tokens), len(amr.nodes))
        # prob_node_by_token_sen[t_pos, :] = p(y= : | x_{a_j}) w/ a_j = t_pos
        prob_node_by_token_sent = \
            self.get_alignment_likelihood(amr, cache_key=cache_key)

        # prior of alignment a_j = i given tokens x. Simplified to depend only
        # on token
        # p(a_j = t_pos | x) ~= p(x_{t_pos} aligns to something)
        # shape = len(amr.tokens)
        prob_token = self.get_alignment_prior(amr)

        # joint distribution of tokens and nodes
        # p(y_j = :, x_{:}) = joint[:, :]
        # shape = (len(amr.tokens), len(amr.nodes))
        joint = prob_node_by_token_sent * prob_token[:, None]

        # node prior by marginalizing tokens
        # p(y_j | x) = sum_{t_pos} p(y_j | x_{t_pos}) p(t_pos | x)
        # shape = (len(amr.nodes))
        node_likelihood = joint.sum(axis=0, keepdims=True)

        # hard zeros may make node_likelihood and prior cancel out
        node_likelihood[node_likelihood == 0] = 1e-8

        # Bayes rule
        # p(a_j = t_pos | y_j, x)
        alignment_posterior = joint / node_likelihood

        if np.isnan(alignment_posterior).any():
            raise Exception()

        return alignment_posterior, node_likelihood

    def align_from_posterior(self, amr, cache_key=None, alpha=0.1):
        """
        Here alpha is used to make a probability distribution sparse by
        limiting the maximum allowed relative decay between sorted
        probabilities
        """

        # get the alignment posterior
        # shape = (len(amr.tokens), len(amr.nodes))
        # p(a_j = : | y_j = node_id) = alignment_posterior[:, node_id]
        alignment_posterior, loglik = \
            self.get_alignment_posterior(amr, cache_key)

        # derive multiple hard alignments from it
        final_node2token = defaultdict(list)
        for node_pos, (node_id, node_name) in enumerate(amr.nodes.items()):
            probs = alignment_posterior[:, node_pos]
            # set low probabilities to zero based on maximum allowed relative
            # decay among sorted probabilities (alpha)
            indices = get_sparse_prob_indices(probs, alpha)
            # store non-zero probabilities as alignments
            for token_pos, p in enumerate(alignment_posterior[:, node_pos]):
                if token_pos in indices:
                    final_node2token[node_id].append(
                        (token_pos, amr.tokens[token_pos], p)
                    )

        return final_node2token, alignment_posterior

    def align_from_likelihood(self, amr, cache_key=None):

        prob_node_by_token_sent = \
            self.get_alignment_likelihood(amr, cache_key=cache_key,
                                          no_norm=True)

        final_node2token = defaultdict(list)
        missing_nodes = defaultdict(list)
        for k, v in amr.nodes.items():
            missing_nodes[v].append(k)
        missing_nodes = dict(missing_nodes)

        # Iterate until all nodes are assigned to a token. Start by the top
        # ranked nodes generated by each token and proceed downwards in the
        # rank (greedily)
        inverse_rank = 1
        while missing_nodes:

            # Look for all node predictions at this rank
            node_preds = []
            no_nodes_left = True
            for token_pos, token_name in enumerate(amr.tokens):

                # TODO: Change naming to reflect use of pseudocounts
                node_probs = prob_node_by_token_sent[token_pos, :]
                node_argsort = node_probs.argsort()
                node_pos = node_argsort[-inverse_rank]
                # TODO: Right now we ignore nodes having same prob
#                 nodes_pos_at_rank = [
#                     p for i, p in enumerate(node_probs) if p == rank_p
#                 ]
                pcount = node_probs[node_pos]

                if pcount > 0:
                    node_name = list(amr.nodes.values())[node_pos]
                    node_preds.append(
                        (token_pos, token_name, node_name, pcount)
                    )
                    no_nodes_left = False
                else:
                    node_preds.append((token_pos, token_name, None, None))
            if no_nodes_left:
                raise Exception('missing alignments')
                break

            inverse_rank += 1

            # remove all nodes that have been predicted
            remove_nodes = []
            for pred in node_preds:
                if pred[2] is None or pred[2] not in missing_nodes:
                    continue
                for nid in missing_nodes[pred[2]]:
                    final_node2token[nid].append((pred[0], pred[1], pred[3]))
                    remove_nodes.append(pred[2])
            for node in remove_nodes:
                if node in missing_nodes:
                    del missing_nodes[node]

        if any(n not in final_node2token for n in amr.nodes.keys()):
            raise Exception()

        return final_node2token

    def align(self, amr, cache_key=None, aformat=None, likelihood=False):

        assert aformat is None or aformat in ['stack'],\
            f"Unknown alignment format {aformat}"

        # Get hard alignments either from posterior or likelihood
        # these alignments are node name to token name so they can be ambiguous
        # if multiple tokens or node names are in the graph
        # TODO: Deprecate loglik
        # node2token = self.align_from_likelihood(amr, cache_key=cache_key)
        node2token, align_posterior, likelihood = self.align_from_posterior(
            amr, cache_key=cache_key
        )

        # consolidate all alignments together
        # TODO: Printer should admit a list of alignments
        node2token = {key: value[0][0] for key, value in node2token.items()}

        return node2token, align_posterior * likelihood

    def format_alignments_for_stack(self, amr, nodeid2token,
                                    unaligned_node_ids):

        # nodes in a NER subgraph are aligned to all surface nodes. This is
        # just a format that O5 and older versions expect
        ner_fixes, ner_ids = align_ners(amr, nodeid2token, aformat='stack')
        nodeid2token.update(ner_fixes)

        # forbid aligning more that one node to a single token with minor
        # exceptions
        # gather all nodes aligned to the same token position
        tokens2node = defaultdict(list)
        unused_positions = list(range(len(amr.tokens)))
        for nid, tokens in nodeid2token.items():
            #
            for token in tokens:
                if token[0] in unused_positions:
                    unused_positions.remove(token[0])
            #
            if nid in ner_ids or nid in unaligned_node_ids:
                continue
            #
            for token in tokens:
                token_pos, token_name, score = token
                tokens2node[(token_pos, token_name)].append(
                    [nid, amr.nodes[nid], score]
                )

        # Assign closest token left-to right
        overlapping_alignments = [
            (tokens, nodes)
            for tokens, nodes in tokens2node.items() if len(nodes) > 1
        ]
        for (token, nodes) in overlapping_alignments:
            for node in nodes[:-1]:
                if unused_positions == []:
                    # if no position available we have to skip
                    continue
                closest_token = sorted(
                    [(i, abs(token[0] - i)) for i in unused_positions],
                    key=itemgetter(1)
                )[0][0]
                nodeid2token[node[0]] = [
                    (closest_token, amr.tokens[closest_token], None)
                ]
                unused_positions.remove(closest_token)

        return nodeid2token

    def print_alignments(self, amr, node=None, token=None, norm=True):
        """
        Print alignments from counts for a node or tokens
        """

        if node is not None and token is not None:
            self.prev_node_by_token_counts[token][node]
        elif token is not None:
            clbar([
                (n, self.prev_node_by_token_counts[token][n])
                for n in amr.nodes.values()
            ], norm=norm)
        elif node is not None:
            clbar([
                (t, self.prev_node_by_token_counts[t][node])
                for t in amr.tokens
            ], norm=norm)

    def print_posterior(self, amr, cache_key=None, node_name=None):
        al_post, _ = self.get_alignment_posterior(amr, cache_key=cache_key)
        # This assumes IBM-model-1 posterior (same name same probabilities)
        index = None
        for i, nname in enumerate(amr.nodes.values()):
            if nname == node_name:
                index = i
                break
        items = zip(amr.tokens, list(al_post[:, index]))
        clbar(items)


def normalize_tokens(tokens):
    return [x.lower().replace('"', '') if x != '"' else '"' for x in tokens]


def get_sentence_features(tokens):

    # get transformations of surface tokens
    # get lemmas and other normalization
    lemmas = []
    for tok, x in zip(tokens, lemmatizer(tokens)):
        lemma = str(x.lemma_)
        # Special replacement for pronouns
        if re.match('-.*-', lemma):
            lemmas.append(normalize_pronouns.get(tok, tok))
        else:
            lemmas.append(normalize_article.get(lemma, lemma))

    # lower cased tokens
    # lemma bigram
    lemma_bigram = [None] \
        + ["-".join(lemmas[t:t+2]) for t in range(len(lemmas) - 1)]
    # undo hyphen tokenization
    detokenized = []
    for pos, token in enumerate(tokens):
        if pos > 1 and tokens[pos-1] == '-':
            detokenized.append("".join(tokens[pos-2:pos+1]))
        else:
            detokenized.append(token)

    return lemmas, lemma_bigram, detokenized


@memoize
def surface_aligner(tokens, nodes, cutoff=0.7):
    """
    cutoff Cutoff for the difflib close string matches
    """

    # get different transformations of input sentence tokens for matching
    # TODO: 2, 3-grams hyphen joined
    lemmas, lemma_bigram, detokenized = get_sentence_features(tokens)

    # proceed over each node try simple alignments first
    unaligned_node_ids = []
    nodeid2token = {}
    nodeid2rule = {}
    for node_id, node_name in nodes:

        # normalizem also node name: remove sense and quotes
        if re.match('.*-[0-9]+', node_name):
            node_lemma = "-".join(node_name.split('-')[:-1])
        else:
            node_lemma = node_name.lower().replace('"', '')

        # token copy(ies) match node
        token_matches = \
            [(p, t) for p, t in enumerate(tokens) if node_lemma == t]
        if any(token_matches):
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['copy-token' for _ in token_matches]
            continue

        # lemma copy(ies) match node
        token_matches = \
            [(p, t) for p, t in enumerate(lemmas) if node_lemma == t]
        if any(token_matches):
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['copy-lemma' for _ in token_matches]
            continue

        # bi-gram of lemma copy(ies) joined by hyphen match node
        token_matches = \
            [(p, t) for p, t in enumerate(lemma_bigram) if node_lemma == t]
        if any(token_matches):
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['copy-lemma-bigram' for _ in token_matches]
            continue

        # joined hyphen tokenized trigrams match node
        token_matches = \
            [(p, t) for p, t in enumerate(detokenized) if node_lemma == t]
        if any(token_matches):
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['copy-detokenized' for _ in token_matches]
            continue

        # matching algo (improvement over Ratcliff and Obershelp algorithm
        # native in Python
        token_matches = get_close_matches(node_lemma, tokens, cutoff=cutoff)
        token_matches = \
            [(p, t) for p, t in enumerate(tokens) if t in token_matches]
        if token_matches:
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['edit-token' for _ in token_matches]
            continue

        # same as before but for lemmas
        token_matches = get_close_matches(node_lemma, lemmas, cutoff=cutoff)
        token_matches = \
            [(p, t) for p, t in enumerate(lemmas) if t in token_matches]
        if token_matches:
            nodeid2token[node_id] = token_matches
            nodeid2rule[node_id] = ['edit-lemma' for _ in token_matches]
            continue

        unaligned_node_ids.append(node_id)

    return nodeid2token, unaligned_node_ids, nodeid2rule


def find_aligned_relative2(amr, node_id, nodeid2token, ignore_node_ids=None):

    if ignore_node_ids is None:
        ignore_node_ids = []

    aligned_children = find_aligned_relative(
        amr.children, node_id, nodeid2token,
        ignore_node_ids=ignore_node_ids
    )
    if aligned_children:
        return aligned_children, False

    aligned_parents = find_aligned_relative(
        amr.parents, node_id, nodeid2token,
        ignore_node_ids=ignore_node_ids
    )
    if aligned_parents:
        return aligned_parents, True

    return None, False


def find_aligned_relative(relatives, node_id, nodeid2token,
                          ignore_node_ids=None):
    """
    move through the graph indicated by the function relatives (may give
    children or parents), return all relatives of node_id.
    """

    if ignore_node_ids is None:
        ignore_node_ids = []

    node_ids = []
    candidates = []
    # keep a list of visited nodes to avoid re-entrancy loops, use this also
    # for ignored nodes
    visited_nodes = set()
    depth = 0
    while depth < 100:

        # Add relatives of current node_id to list of potential candidates
        candidates += \
            [x[0] for x in relatives(node_id) if x[0] not in visited_nodes]
        visited_nodes |= set(candidates)

        # if a candidate is aligned and is not in ignore list add it to the
        # values to be returned, if not keep it in candidates so that we can
        # analyze its relatives
        new_candidates = []
        for cand in candidates:
            # If a node to be ignored is current node_id, that means we have
            # already visitied its parents and we can remove it
            if cand == node_id and cand in ignore_node_ids:
                continue
            if cand in nodeid2token and cand not in ignore_node_ids:
                # if a node has alignment and is not to be ignored, this is one
                # of the solutions
                # append also the last token aligned to this node for later
                # sorting
                last_token = max([x[0] for x in nodeid2token[cand]])
                node_ids.append((cand, last_token))
            else:
                # keep for calling its parents() on next iteration
                new_candidates.append(cand)
        candidates = new_candidates

        # increase depth count (relative to current node position)
        depth += 1

        # if no more candidates, exit
        if not candidates:
            break

        # pick th first candidate as new node
        node_id = candidates.pop()

    # This should not happen as we took care of loops
    assert depth < 100, "Maximum recursion depth reached"

    return node_ids


def graph_vicinity_aligner(amr, nodeid2token, unaligned_node_ids):

    new_nodeid2token = {
        n: t for n, t in nodeid2token.items() if n not in unaligned_node_ids
    }
    new_nodeid2rule = {}
    still_unaligned_node_ids = []

    # start by highest unaligned ID

    for node_id in unaligned_node_ids:

        if node_id == 'person':
            continue

        # find aligned children or parents, keep the last one according to
        # sentence token order
        aligned_relatives, is_parent = \
            find_aligned_relative2(amr, node_id, new_nodeid2token)

        if aligned_relatives is None:
            if node_id in nodeid2token:
                # If no relatives, keep old alignment
                new_nodeid2token[node_id] = nodeid2token[node_id]
                continue
            else:
                # If no alignments, we have a problem
                raise Exception()

        if not is_parent:
            pos = sorted(aligned_relatives, key=itemgetter(1))[-1][1]
            new_nodeid2token[node_id] = [(pos, amr.tokens[pos], None)]
            new_nodeid2rule[node_id] = ['last-child']
            continue

        if aligned_relatives:
            pos = sorted(aligned_relatives, key=itemgetter(1))[0][1]
            new_nodeid2token[node_id] = [(pos, amr.tokens[pos], None)]
            new_nodeid2rule[node_id] = ['first-parent']
            continue

        # Some of these cases my be solvable if we use the alignments obtained
        # here
        still_unaligned_node_ids.append(node_id)

    return new_nodeid2token, still_unaligned_node_ids


def graph_vicinity_resolver(amr, nodeid2token, cutoff=0.01,
                            ignore_relative_ids=None):
    """
    for every node with multiple alignments, assign a cost for satisfiying
    graph constraints and alignment cost constraints

    assign greedly nodes with lower cost first
    """

    if ignore_relative_ids is None:
        ignore_relative_ids = []

    # store node id for which there is more than one alignment, note down those
    # for wich alignment stats are the same for all nodes
    unclear_nodes = defaultdict(dict)
    same_alignment_stats_ids = []
    for node_id, tokens in nodeid2token.items():

        # filter very low probability alignments
        tokens = [token for token in tokens if token[2] > cutoff]

        if len(tokens) == 1:
            continue

        if len(set([tk[2] for tk in tokens])) == 1:
            same_alignment_stats_ids.append(node_id)

        # Add costs of aligning to neighbouring nodes
        relatives, _ = find_aligned_relative2(
            amr, node_id, nodeid2token,
            ignore_node_ids=ignore_relative_ids)
        if relatives is None or any(x is None for x in relatives):
            unclear_nodes[node_id]['graph_costs'] = \
                [i for i in range(len(tokens))]
            unclear_nodes[node_id]['tokens'] = tokens
        else:
            graph_costs = []
            for token in tokens:
                graph_cost = sum(
                    abs(token[0] - relative[1]) for relative in relatives
                ) / len(relatives)
                graph_costs.append(graph_cost)

            unclear_nodes[node_id]['graph_costs'] = graph_costs
            unclear_nodes[node_id]['tokens'] = tokens

    # solve first the nodes for which we only have alignment cost to
    # discriminate. Store all matches by node
    # collect candidates and cost
    candidates = []
    for node_id in unclear_nodes.keys():
        for i in range(len(unclear_nodes[node_id]['tokens'])):
            cost = unclear_nodes[node_id]['graph_costs'][i]
            candidates.append(
                [cost] + list(unclear_nodes[node_id]['tokens'][i]) + [node_id]
            )

    # greedily select best candidates
    assigned_nodes = []
    assigned_tokens = []
    candidates = sorted(candidates, key=lambda x: x[0])
    while candidates:
        cost, token_pos, token_name, stats_cost, node_id = candidates.pop(0)
        if token_pos not in assigned_tokens and node_id not in assigned_nodes:
            nodeid2token[node_id] = [(token_pos, token_name, stats_cost)]
            assigned_nodes.append(node_id)
            assigned_tokens.append(token_pos)

    # Nodes that have not bee aligned by graph distance are aligned by
    # IBM-model-1 alignment cost
    fix_nodeid2token = dict()
    for nid, tokens in nodeid2token.items():
        if len(tokens) > 1:
            fix_nodeid2token[nid] = [sorted(tokens, key=lambda x: x[2])[-1]]

    nodeid2token.update(fix_nodeid2token)

    return nodeid2token


def align_ners(amr, nodeid2token, unaligned_node_ids=None,
               flat_alignments=False, aformat=None):
    """
    Align all elements on a NER subgraph that are not surface to the last
    of the surface symbols

    aformat = 'stack' ensures NER alignments match those of
    stack-lstm/Transformer oracle

    FIXME: Agreeing on a single format would end need for this
    flat_alignments=True assumes dict of alignments values are ints not lists
    """

    if unaligned_node_ids:
        target_nodes = unaligned_node_ids
    else:
        target_nodes = list(nodeid2token.keys())

    # Loop and find NER subgraphs
    node2token_fixes = {}
    ner_ids = []
    for node_id in target_nodes:
        if (
            amr.nodes[node_id] == 'name'
            and any([x[1] == ':name' for x in amr.parents(node_id)])
        ):

            # all nodes below the "name" node
            child_nodes = amr.children(node_id)
            # parent node of "name" node (NER tag)
            tag_id = [e[0] for e in amr.parents(node_id) if e[1] == ':name'][0]

            # TODO: define alignments allways as list (no flat_alignments)
            # Get
            ner_alignments = [
                nodeid2token[x[0]]
                if flat_alignments else nodeid2token[x[0]][0][0]
                for x in child_nodes
                if x[1].startswith(':op') and x[0] in nodeid2token
            ]

            # store all ids for this subgraph
            ner_ids.extend([tag_id] + [node_id] + [x[0] for x in child_nodes])

            # some entities may be unaligned
            if ner_alignments == []:
                continue

            if aformat == 'stack':

                # all graph nodes aligned to the full span
                for nid in [node_id, tag_id] + [x[0] for x in child_nodes]:
                    if flat_alignments:
                        node2token_fixes[nid] = ner_alignments
                    else:
                        node2token_fixes[nid] = [
                            (i, amr.tokens[i], None) for i in ner_alignments
                        ]

            elif aformat is None:

                # pick the last of the aligments for tag and name nodes
                index = max(ner_alignments)
                if flat_alignments:
                    node2token_fixes[node_id] = index
                    node2token_fixes[tag_id] = index
                else:
                    node2token_fixes[node_id] = [
                        (index, amr.tokens[index], None)
                    ]
                    node2token_fixes[tag_id] = [
                        (index, amr.tokens[index], None)
                    ]
            else:
                raise Exception(f'Uknown alignment format {aformat}')

    return node2token_fixes, ner_ids


def visual_eval(amr_aligner, indices, amrs, compare, aformat):

    print('Displaying some examples for eval')
    for index in indices:

        amr = amrs[index]
        final_node2token = amr_aligner.align(amr, cache_key=index,
                                             aformat=aformat)

        # if any(nid not in final_node2token for nid in amr.nodes):
        #    final_node2token[nid] = None

        mark_ids = None
        if compare:
            # Get original alignments, flag nodes with different alignment,
            # ignore comparison if there are no differences
            ref_final_node2token = \
                {k: v[0] for k, v in amr.alignments.items() if v}
            mark_ids = []
            for nid in amr.nodes.keys():
                if final_node2token[nid] != ref_final_node2token[nid]:
                    mark_ids.append(nid)
            if mark_ids == []:
                continue

        print(" ".join(amr.tokens))
        print(index)
        plot_graph(
            amr.tokens, amr.nodes, amr.edges, final_node2token,
            mark_ids=mark_ids, plot_now=not compare
        )
        if compare:
            # Reference
            plot_graph(
                amr.tokens, amr.nodes, amr.edges, ref_final_node2token,
                mark_ids=None, plot_now=True
            )

        response = input('Quit [N/y]?')
        if response == 'y':
            break


# alignment priors will disencourage these
IGNORE_TOKENS = ['.', ',', 'the', 'a', 'of', '-', 'to']


# rule prior or statistics will be ignored for these nodes and only graph
# vicinity rules will be used
IGNORE_NODES = [
   # abstract entities
   'person',
   'thing',
   'organization',
   'government-organization',
   'political-party',
   #
   'percentage-entity',
   'ordinal-entity',
   'url-entity',
   # quantities
   'area-quantity',
   'temporal-quantity',
   'distance-quantity',
   'temperature-quantity',
   'monetary-quantity',
   'seismic-quantity',
   'mass-quantity',
   # dates
   'date-interval',
   'date-entity',
   # other
   'multi-sentence',
   'amr-unknown',
]

# same as previous
IGNORE_REGEX = [
    re.compile('.*-91$')  # reified concepts
]


def align_and_score(amrs, original_tokens, indices, amr_aligner, compare):

    # compare with previous alignments
    alignment_match_counts = Counter()
    final_amrs = []
    final_joint = []
    for index in tqdm(indices, desc='Aligning data'):

        amr = amrs[index]
        alignments, joint = amr_aligner.align(amr, cache_key=index)

        if compare:
            # update comparison stats
            for node_id, node_name in amr.nodes.items():
                if [alignments[node_id]] == amr.alignments[node_id]:
                    alignment_match_counts.update([None])
                else:
                    alignment_match_counts.update([node_name])

        # recover tokens
        amr.tokens = original_tokens[index]

        # overwrite alignments
        amr.alignments = {k: [v] for k, v in alignments.items()}
        final_amrs.append(amr)

        # store joint distribution of token positions and nodes
        final_joint.append(joint)

    if compare:
        clbar(alignment_match_counts.most_common(50), ylim=(0, 0.1), norm=True)

    if out_aligned_amr:
        with open(out_aligned_amr, 'w') as fid:
            for amr in final_amrs:
                fid.write(f'{amr.__str__()}\n\n')


def stats(amr_aligner):
    """
    Show highly frequent tokens and their alignments
    """
    # update the model parameters
    token_counts = defaultdict(int)
    for token_name, ncounts in amr_aligner.prev_node_by_token_counts.items():
        token_counts[token_name] += sum(ncounts.values())
    p = np.array(list(token_counts.values()))
    p = p / p.sum()
    token_prior = dict(zip(token_counts.keys(), p))
    clbar(sorted(token_prior.items(), key=lambda x: x[1])[-30:])

    def get_node_alignments(node):
        return [
            (token, nodes[node])
            for token, nodes in amr_aligner.prev_node_by_token_counts.items()
            if nodes[node] > 0
        ]

    clbar(sorted(token_prior.items(), key=lambda x: x[1])[-30:])
    print()


def stats_debug_amr(amrs):

    token_count_per_sent = Counter()
    node_count_per_sent = Counter()
    for amr in amrs:
        for token_type in list(set(amr.tokens)):
            token_count_per_sent.update([token_type])

        for node_type in list(set(amr.nodes.values())):
            node_count_per_sent.update([node_type])

    clbar(token_count_per_sent.most_common(20))
    clbar(node_count_per_sent.most_common(20))
    set_trace()


def main(args):

    # sanity checks
    assert bool(args.in_aligned_amr) ^ bool(args.in_amr), \
        "Needs either --in-amr or --in-aligned-amr"
    if args.compare:
        assert args.in_aligned_amr, "--compare only with --in-aligned-amr"

    # files
    if args.in_amr:
        amrs = read_amr(args.in_amr, tokenize=args.tokenize)
    else:
        amrs = read_amr(args.in_aligned_amr, ibm_format=True,
                         tokenize=args.tokenize)
    # normalize tokens for matching purposes, but keep the original for writing
    original_tokens = []
    for amr in amrs:
        original_tokens.append(amr.tokens)
        amr.tokens = normalize_tokens(amr.tokens)

    assert args.em_epochs > 0 or args.rule_prior_strength > 0, \
        "Either set --em-epochs > 0 or --rule-prior-strength > 0"

    # if not given pick random order
    if args.indices is None:
        indices = list(range(len(amrs)))
        if args.shuffle:
            shuffle(indices)
    else:
        indices = args.indices

    eval_indices = indices

    # Initialize aligner. This is an IBM model 1 using surface matching rules
    # as prior and graph vicinity rules post-processing
    if args.in_checkpoint_json:
        amr_aligner = AMRAligner.from_checkpoint(args.in_checkpoint_json)
    else:
        amr_aligner = AMRAligner(
            rule_prior_strength=args.rule_prior_strength,
            force_align_ner=args.force_align_ner,
            not_align_tokens=IGNORE_TOKENS,
            ignore_nodes=IGNORE_NODES,
            ignore_node_regex=IGNORE_REGEX
        )

    # loop over EM epochs
    av_log_lik = None
    for epoch in range(args.em_epochs):
        if av_log_lik:
            bar_desc = \
                f'EM epoch {epoch+1}/{args.em_epochs} loglik {av_log_lik}'
        else:
            bar_desc = f'EM epoch {epoch+1}/{args.em_epochs}'
        for index in tqdm(indices, desc=bar_desc):
            # accumulate stats while fixing the posterior
            amr_aligner.update_counts(amrs[index], cache_key=index)

        # compute loglik
        av_log_lik = amr_aligner.train_loglik / amr_aligner.train_num_examples

        # update the model parameters
        amr_aligner.update_parameters()

    # save model
    if args.out_checkpoint_json:
        amr_aligner.save(args.out_checkpoint_json)

    # check some examples of alignment visualy
    if args.visual_eval:
        visual_eval(amr_aligner, eval_indices, amrs, args.compare,
                    args.alignment_format)

    # get final alignments and score if solicted
    if args.out_aligned_amr or args.compare:
        aligned_amrs, joints = align_and_score(
            amrs, original_tokens, indices, amr_aligner, args.compare)

    # store alignments to disk
    if args.out_aligned_amr:
        # AMR in penman + JAMR notation
        with open(args.out_aligned_amr, 'w') as fid:
            for amr in aligned_amrs:
                fid.write(f'{amr.to_jamr()}')

    if args.out_alignment_probs:
        # write joint probabilities
        write_neural_alignments(args.out_alignment_probs, aligned_amrs, joints)


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    # Single input parameters
    parser.add_argument(
        "--in-amr",
        help="In file containing AMR in penman format",
        type=str
    )
    parser.add_argument(
        "--in-aligned-amr",
        help="In file containing AMR in penman format AND IBM graph notation "
             "(::node, etc). Graph read from the latter and not penman",
        type=str
    )
    parser.add_argument(
        "--out-aligned-amr",
        help="Out File containing AMR in penman format AND IBM graph notation "
             "(::node, etc). Graph read from the latter and not penman",
        type=str
    )
    parser.add_argument(
        "--out-alignment-probs",
        help="Alignment probabilities in formaty used by neural aligner",
        type=str
    )
    parser.add_argument(
        "--alignment-format",
        help="stack alignes all nodes in the NER subgraph to entire span",
        choices=['stack'],
        type=str
    )
    parser.add_argument(
        "--tokenize",
        help="Use the simplest tokenizer applied to ::snt",
        action='store_true'
    )
    parser.add_argument(
        "--shuffle",
        help="Randomize sentence order",
        action='store_true'
    )
    parser.add_argument(
        "--indices",
        nargs='+',
        type=int,
        help="Indices in --in-amr of target examples",
    )
    parser.add_argument(
        "--em-epochs",
        help="Number of Expectation Maximization epochs (zero uses no EM)",
        type=int,
        default=2
    )
    parser.add_argument(
        "--rule-prior-strength",
        help="Prior strength of rules (interpreted as pseudocounts)",
        type=float,
        default=100
    )
    parser.add_argument(
        "--force-align-ner",
        help="(<tag> :name name :opx <token>) allways aligned to token",
        action='store_true'
    )
    parser.add_argument(
        "--visual-eval",
        help="Plot some random examples for analysis",
        action='store_true'
    )
    parser.add_argument(
        "--compare",
        help="If --in-amr aligned, compare with the new alignments "
        "(stats or visual)",
        action='store_true'
    )
    parser.add_argument(
        "--in-checkpoint-json",
        help="Previously computed alignmentg model",
        type=str,
    )
    parser.add_argument(
        "--out-checkpoint-json",
        help="Where to store model",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
