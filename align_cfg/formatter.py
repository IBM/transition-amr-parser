import collections
import copy
import datetime

import numpy as np
import torch

import penman
from penman import layout
from penman._format import _format_node

from tqdm import tqdm

from amr_utils import get_node_ids, get_tree_edges

from alignment_decoder import AlignmentDecoder


class AMRStringHelper(object):
    @staticmethod
    def tok(amr):
        return '# ::tok ' + ' '.join(amr.tokens)

    @staticmethod
    def nodes(amr, alignments):
        output = []
        for node_id, a in alignments.items():
            name = amr.nodes[node_id]
            output.append('# ::node\t{}\t{}\t{}-{}'.format(node_id, name, a[0], a[-1] + 1))
        return output

    @staticmethod
    def root(amr, alignments):
        name = amr.nodes[amr.root]
        if amr.root in alignments:
            a = alignments[amr.root]
            return '# ::root\t{}\t{}\t{}-{}'.format(amr.root, name, a[0], a[-1] + 1)
        else:
            return '# ::root\t{}\t{}'.format(amr.root, name)

    @staticmethod
    def edges(amr, alignments):
        edges = []
        for edge_in, label, edge_out in amr.edges:
            name_in = amr.nodes[edge_in]
            name_out = amr.nodes[edge_out]
            if label.startswith(':'):
                label = label[1:]
            row = [name_in, label, name_out, edge_in, edge_out]
            edges.append('# ::edge\t' + '\t'.join(row))
        return edges

    @staticmethod
    def alignments(amr, alignments):
        dt_string = datetime.datetime.isoformat(datetime.datetime.now())
        prefix = '# ::alignments '
        suffix = '::annotator neural ibm model 1 v.01 ::date {}'.format(dt_string)

        body = ''
        for i, (node_id, a) in enumerate(alignments.items()):
            if i > 0:
                body += ' '
            assert isinstance(a, list)
            start = a[0]
            end = a[-1] + 1
            assert start >= 0
            assert end >= 0
            body += '{}-{}|{}'.format(start, end, node_id)
        body += ' '

        return prefix + body + suffix

    @staticmethod
    def amr(amr):
        if amr.penman is None:
            pretty_amr = amr.__str__().split('\t')[-1].strip()
        else:
            pretty_amr = _format_node(layout.configure(amr.penman).node, -1, 0, [])
        return pretty_amr


def amr_to_string(amr, alignments=None):
    if alignments is None:
        alignments = amr.alignments

    amr = copy.deepcopy(amr)
    amr.alignments = alignments

    new_amr_nodes = {}
    new_edges = []
    mapping = {}
    new_amr_alignments = {}

    # book-keeping
    tree_edges = get_tree_edges(amr)
    node_to_children = collections.defaultdict(list)
    node_to_depth = {}
    node_to_depth[amr.root] = 0
    for a, b, c, _, node_id in tree_edges:
        depth = node_id.count('.')
        node_to_depth[c] = depth
        node_to_children[a].append(c)

    # get unique ids
    node_set = set()
    node_items = list(amr.nodes.items())
    node_to_int = {node_id: j for j, (node_id, node_name) in enumerate(node_items)}
    for i, (node_id, node_name) in enumerate(node_items):
        depth = node_to_depth.get(node_id, 'x')
        children_ints = sorted([node_to_int[c_node_id] for c_node_id in node_to_children[node_id]])
        if len(children_ints) > 0:
            children_abbrev = '-'.join([str(j) for j in children_ints])
            new_node_id = '{}.d{}-{}'.format(str(i), depth, children_abbrev)
        else:
            new_node_id = '{}.d{}'.format(str(i), depth)
        assert new_node_id not in new_amr_nodes
        new_amr_nodes[new_node_id] = node_name
        mapping[node_id] = new_node_id

    for node_id, v in amr.alignments.items():
        new_node_id = mapping[node_id]
        new_amr_alignments[new_node_id] = v

    for a, b, c in amr.edges:
        a = mapping[a]
        c = mapping[c]
        new_edges.append((a, b, c))

    assert len(amr.nodes) == len(new_amr_nodes)

    amr.nodes = new_amr_nodes
    amr.alignments = new_amr_alignments
    amr.edges = new_edges
    amr.root = mapping[amr.root]
    alignments = amr.alignments

    body = ''
    body += AMRStringHelper.tok(amr) + '\n'
    if len(amr.nodes) > 0:
        body += AMRStringHelper.alignments(amr, alignments) + '\n'
        nodes_str = AMRStringHelper.nodes(amr, alignments)
        if nodes_str:
            body += '\n'.join(nodes_str) + '\n'
        body += AMRStringHelper.root(amr, alignments) + '\n'
    if len(amr.edges) > 0:
        body += '\n'.join(AMRStringHelper.edges(amr, alignments)) + '\n'
    body += AMRStringHelper.amr(amr) + '\n'

    return body


def amr_to_pretty_format(amr, ainfo, idx):
    posterior = ainfo['posterior']

    text_tokens = amr.tokens
    node_ids = get_node_ids(amr)
    node_names = [amr.nodes[x] for x in node_ids]

    shape = (len(node_ids), len(text_tokens), 1)

    assert posterior.shape == shape, 'expected = {} , actual = {}'.format(shape, posterior.shape)

    #
    s = ''

    #
    s += '{}\n'.format(idx)           # 0
    s += ' '.join(node_names) + '\n'  # 1
    s += ' '.join(node_ids) + '\n'    # 2
    s += ' '.join(text_tokens) + '\n' # 3

    #
    for i in range(len(node_ids)):
        for j in range(len(text_tokens)):
            a = posterior[i, j].item()
            s += '{} {} {}\n'.format(i, j, a)

    s += '\n'

    return s


def read_amr_pretty_format(amr, s):
    lines = s.split('\n')

    node_ids = lines[2].strip().split()
    text_tokens = lines[3].strip().split()

    assert len(lines) == (len(node_ids) * len(text_tokens) + 4)

    posterior = np.zeros((len(node_ids), len(text_tokens)), dtype=np.float32)

    for i in range(len(node_ids)):
        for j in range(len(text_tokens)):
            x = lines[4 + i * len(text_tokens) + j]
            posterior[i, j] = float(x.split()[-1])

    return posterior


def read_amr_pretty_file(path, corpus):
    i = 0

    posterior_list = []

    with open(path) as f:
        s = None

        for line in tqdm(f, desc='read-pretty'):
            if not line.strip():
                if s is not None and s.strip():
                    amr = corpus[i]
                    posterior_list.append(read_amr_pretty_format(amr, s.strip()))
                    i += 1
                    s = None
                continue

            if s is None:
                s = ''

            s += line

    assert len(corpus) == len(posterior_list)

    return posterior_list
