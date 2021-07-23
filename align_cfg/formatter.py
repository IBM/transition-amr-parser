import datetime

import torch

from vocab import *

import penman
from penman import layout
from penman._format import _format_node


class FormatAlignments(object):
    """
    Print alignments in desired format.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def decode_alignments(self, batch_map, model_output):
        """
        For each node, find most probable alignments.
        """
        x_t = batch_map['text_tokens']

        batch_size, len_t = x_t.shape
        device = x_t.device

        y_a = model_output['labels']
        y_a_mask = model_output['labels_mask']
        y_a_node_ids = model_output['label_node_ids']

        for i_b in range(batch_size):
            indexa = torch.arange(y_a.shape[1])
            m_ = y_a_mask[i_b].view(-1)
            n = m_.sum().item()
            indexa_ = indexa[m_]

            indext = torch.arange(len_t)
            t = x_t[i_b]
            mt = t != PADDING_IDX
            nt = mt.sum().item()
            indext_ = indext[mt]

            align = model_output['batch_align'][i_b]
            assert align.shape == (n, nt, 1)

            argmax = align.squeeze(2).argmax(1)
            assert argmax.shape == (n,)

            node_alignments = []
            for j in range(n):
                node_id = y_a_node_ids[i_b, indexa_[j]].item()
                idx_txt = indext_[argmax[j].item()].item()
                node_alignments.append((node_id, idx_txt))

            yield node_alignments

    def format(self, batch_map, model_output, batch_indices, use_jamr=False):
        for idx, node_alignments in zip(batch_indices, self.decode_alignments(batch_map, model_output)):
            amr = self.dataset.corpus[idx]
            text_tokens = amr.tokens

            node_ids = list(sorted(amr.nodes.keys()))
            amr_tokens, _ = self.dataset.amr_tokenizer.dfs(amr)

            def convert(node_alignments):
                output = []

                for node_id, idx_txt in node_alignments:
                    node_id = node_ids[node_id]

                    # indexing should account for padding
                    # TODO: Redo the text indexing to be more tidy...
                    t = text_tokens[idx_txt - 1]
                    a = amr.nodes[node_id]
                    a_id = node_id

                    assert a_id is not None

                    output.append((a, a_id, t))

                return output

            def convert_standard(node_alignments):
                output = {}

                for node_id, idx_txt in node_alignments:
                    # indexing should account for padding
                    a_id = node_ids[node_id]
                    output[a_id] = [idx_txt]

                return output

            def s_alignment(alignments):
                dt_string = datetime.datetime.isoformat(datetime.datetime.now())
                prefix = '# ::alignments '
                suffix = '::annotator neural ibm model 1 v.01 ::date {}'.format(dt_string)

                body = ''
                for i, (node_id, idx_txt) in enumerate(alignments):
                    if i > 0:
                        body += ' '
                    a_id = node_ids[node_id]
                    if isinstance(idx_txt, list):
                        start = idx_txt[0]
                        end = idx_txt[-1] + 1
                        assert start >= 0
                        assert end >= 0
                        body += '{}-{}|{}'.format(start, end, node_id)
                    else:
                        body += '{}-{}|{}'.format(idx_txt - 1, idx_txt, node_id)
                body += ' '

                return prefix + body + suffix

            def fmt(alignments):
                if use_jamr:
                    return amr.__str__()

                node_TO_align = {}
                for node_id, idx_txt in alignments:
                    node_id = node_ids[node_id]
                    if isinstance(idx_txt, list):
                        start = idx_txt[0]
                        end = idx_txt[-1] + 1
                        assert start >= 0
                        assert end >= 0
                        node_TO_align[node_id] = '{}-{}'.format(start, end)
                    else:
                        node_TO_align[node_id] = '{}-{}'.format(idx_txt - 1, idx_txt)

                # tok
                tok = '# ::tok ' + ' '.join(amr.tokens)

                # nodes
                nodes = []
                for node_id, name in amr.nodes.items():
                    a = node_TO_align[node_id]
                    nodes.append('# ::node\t{}\t{}\t{}'.format(node_id, name, a))

                # root
                node_id = amr.root
                a = node_TO_align[node_id]
                name = amr.nodes[node_id]
                nodes.append('# ::root\t{}\t{}\t{}'.format(node_id, name, a))

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
                assert amr.penman is None
                pretty_amr = amr.__str__().split('\t')[-1].strip()
                # pretty_amr = _format_node(layout.configure(amr.penman).node, -1, 0, [])

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

            alignments = convert(node_alignments)

            def convert_gold_alignments():
                new_alignments = []

                for node_id, a in amr.alignments.items():
                    new_node_id = node_ids.index(node_id)
                    assert isinstance(a, list)
                    # assert len(a) == 1, a

                    new_alignments.append((new_node_id, a))
                return new_alignments


            out_pred = fmt(node_alignments)
            out_gold = fmt(convert_gold_alignments())
            out_standard = convert_standard(node_alignments)

            yield out_pred, out_gold, out_standard


class FormatAlignmentsPretty(object):
    """
    Print alignments in desired format.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def decode_alignments(self, batch_map, model_output):
        """
        For each node, find most probable alignments.
        """
        x_t = batch_map['text_tokens']

        batch_size, len_t = x_t.shape
        device = x_t.device

        y_a = model_output['labels']
        y_a_mask = model_output['labels_mask']
        y_a_node_ids = model_output['label_node_ids']

        for i_b in range(batch_size):
            mask_amr = y_a_mask[i_b].view(-1)
            index_amr = torch.arange(y_a.shape[1])[mask_amr]
            n_amr = mask_amr.sum().item()

            txt = x_t[i_b]
            mask_txt = txt != PADDING_IDX
            index_txt = torch.arange(len_t)[mask_txt]
            n_txt = mask_txt.sum().item()

            posterior = model_output['batch_align'][i_b]
            assert posterior.shape == (n_amr, n_txt, 1)

            argmax = posterior.squeeze(2).argmax(1)
            assert argmax.shape == (n_amr,)

            node_idx = y_a_node_ids[i_b][index_amr]
            txt = txt[index_txt]

            alignment_info = {}
            alignment_info['posterior'] = posterior
            alignment_info['argmax'] = argmax
            # alignment_info['node_idx'] = node_idx
            # alignment_info['txt'] = txt

            yield alignment_info

    def format(self, batch_map, model_output, batch_indices, use_jamr=False):
        for idx, alignment_info in zip(batch_indices, self.decode_alignments(batch_map, model_output)):
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
                    a = alignment_info['posterior'][i, j].item()
                    s += '{} {} {}\n'.format(i, j, a)

            s += '\n'

            yield amr, s
