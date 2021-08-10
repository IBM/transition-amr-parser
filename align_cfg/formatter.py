import datetime

import torch

import penman
from penman import layout
from penman._format import _format_node

from amr_utils import get_node_ids

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

    body = ''
    body += AMRStringHelper.tok(amr) + '\n'
    if len(amr.nodes) > 0:
        body += AMRStringHelper.alignments(amr, alignments) + '\n'
        body += '\n'.join(AMRStringHelper.nodes(amr, alignments)) + '\n'
        body += AMRStringHelper.root(amr, alignments) + '\n'
    if len(amr.edges) > 0:
        body += '\n'.join(AMRStringHelper.edges(amr, alignments)) + '\n'
    body += AMRStringHelper.amr(amr) + '\n'

    return body


class FormatAlignments(object):
    """
    Print alignments in desired format.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def format(self, batch_map, model_output, batch_indices, use_jamr=False):
        alignment_info_ = AlignmentDecoder().batch_decode(batch_map, model_output)

        for idx, ainfo in zip(batch_indices, alignment_info_):
            #
            amr = self.dataset.corpus[idx]
            node_ids = get_node_ids(amr)
            alignments = {node_ids[node_id]: a for node_id, a in ainfo['node_alignments']}

            out_pred = amr_to_string(amr, alignments=alignments)
            if amr.alignments is None:
                out_gold = None
            else:
                out_gold = amr_to_string(amr, alignments=amr.alignments)

            yield out_pred, out_gold


class FormatAlignmentsPretty(object):
    """
    Print alignments in desired format.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def format(self, batch_map, model_output, batch_indices, use_jamr=False):
        alignment_info_ = AlignmentDecoder().batch_decode(batch_map, model_output)

        for idx, ainfo in zip(batch_indices, alignment_info_):

            posterior = ainfo['posterior']

            #

            amr = self.dataset.corpus[idx]
            text_tokens = amr.tokens

            node_ids = list(sorted(amr.nodes.keys()))
            node_names = [amr.nodes[x] for x in node_ids]

            #
            s = ''

            #
            s += '{}\n'.format(idx)
            s += ' '.join(node_names) + '\n'
            s += ' '.join(node_ids) + '\n'
            s += ' '.join(text_tokens) + '\n'

            #
            for i in range(len(node_ids)):
                for j in range(len(text_tokens)):
                    a = posterior[i, j].item()
                    s += '{} {} {}\n'.format(i, j, a)

            s += '\n'

            yield amr, s
