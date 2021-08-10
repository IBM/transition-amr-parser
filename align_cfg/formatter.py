import datetime

import torch

import penman
from penman import layout
from penman._format import _format_node

from amr_utils import get_node_ids

from alignment_decoder import AlignmentDecoder


class AMRStringHelper(object):
    def f(self):
        pass


def amr_to_string(amr, alignments=None):
    if alignments is None:
        alignments = amr.alignments


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
            text_tokens = amr.tokens
            node_ids = get_node_ids(amr)
            alignments = {node_ids[node_id]: a for node_id, a in ainfo['node_alignments']}

            def s_alignment(alignments):
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

            def fmt(alignments):
                node_TO_align = {}
                for node_id, a in alignments.items():
                    assert isinstance(a, list)
                    start = a[0]
                    end = a[-1] + 1
                    assert start >= 0
                    assert end >= 0
                    node_TO_align[node_id] = '{}-{}'.format(start, end)

                # tok
                tok = '# ::tok ' + ' '.join(amr.tokens)

                # nodes
                nodes = []
                for node_id, a in node_TO_align.items():
                    name = amr.nodes[node_id]
                    nodes.append('# ::node\t{}\t{}\t{}'.format(node_id, name, a))

                # root
                node_id = amr.root
                a = node_TO_align.get(node_id, None)
                name = amr.nodes[node_id]
                if a is not None:
                    nodes.append('# ::root\t{}\t{}\t{}'.format(node_id, name, a))
                else:
                    nodes.append('# ::root\t{}\t{}'.format(node_id, name))

                # edges
                edges = []
                for edge_in, label, edge_out in amr.edges:
                    name_in = amr.nodes[edge_in]
                    name_out = amr.nodes[edge_out]
                    label = label[1:]
                    row = [name_in, label, name_out, edge_in, edge_out]
                    edges.append('# ::edge\t' + '\t'.join(row))

                # get alignments
                alignments_str = s_alignment(alignments)

                # amr
                if amr.penman is None:
                    pretty_amr = amr.__str__().split('\t')[-1].strip()
                else:
                    pretty_amr = _format_node(layout.configure(amr.penman).node, -1, 0, [])

                # new output
                out = ''
                out += tok + '\n'
                if len(nodes) > 0:
                    out += alignments_str + '\n'
                    out += '\n'.join(nodes) + '\n'
                if len(edges) > 0:
                    out += '\n'.join(edges) + '\n'
                out += pretty_amr + '\n'

                return out

            out_pred = fmt(alignments)
            if amr.alignments is None:
                out_gold = None
            else:
                out_gold = fmt(amr.alignments)

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
