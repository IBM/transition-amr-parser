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
        for node_id, name in sorted(amr.nodes.items(), key=lambda x: x[0]):
            if node_id in alignments:
                a = alignments[node_id]
                output.append('# ::node\t{}\t{}\t{}-{}'.format(node_id, name, a[0], a[-1] + 1))
            else:
                output.append('# ::node\t{}\t{}\t'.format(node_id, name))
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
            #pretty_amr = amr.__str__().split('\t')[-1].strip()
            pretty_amr = '\n'.join([x for x in amr.__str__().split('\n') if not x.startswith('#')])
            pretty_amr = pretty_amr.strip()
        else:
            pretty_amr = _format_node(layout.configure(amr.penman).node, -1, 0, [])
        assert pretty_amr.startswith('('), pretty_amr
        return pretty_amr


def amr_to_string(amr, alignments=None):
    if alignments is None:
        alignments = amr.alignments

    amr = copy.deepcopy(amr)
    amr.alignments = alignments
    alignments = amr.alignments

    body = ''

    try:
        if hasattr(amr, 'id'):
            body += '# ::id {}\n'.format(amr.id)
        else:
            body += '# ::id {}\n'.format(amr.penman.metadata['id'])

    except:
        pass

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

