import torch
import numpy as np

from vocab import PADDING_IDX


class AlignmentDecoder(object):

    def batch_decode(self, batch_map, model_output):
        """
        For each node, find most probable alignments.
        """
        x_t = batch_map['text_tokens']
        y_a = model_output['labels']
        y_a_mask = model_output['labels_mask']
        y_a_node_ids = model_output['label_node_ids']
        align = model_output['batch_align']

        batch_size, len_t = x_t.shape
        len_a = y_a.shape[-1]
        device = x_t.device

        for i_b in range(batch_size):

            # variables

            indexa = torch.arange(len_a).to(device)
            indext = torch.arange(len_t).to(device)

            # select

            b_x_t = x_t[i_b]
            b_y_a_mask = y_a_mask[i_b].view(-1)
            b_align = align[i_b]

            # mask

            b_x_t_mask = b_x_t != PADDING_IDX
            b_indexa = indexa[b_y_a_mask]
            b_indext = indext[b_x_t_mask]

            n = b_y_a_mask.sum().item()
            nt = b_x_t_mask.sum().item()

            assert b_align.shape == (n, nt, 1)

            # decode

            argmax = b_align.squeeze(2).argmax(1)

            assert argmax.shape == (n,)

            # node alignments

            node_alignments = []
            for j in range(n):
                node_id = y_a_node_ids[i_b, b_indexa[j]].item()
                idx_txt = argmax[j].item()
                node_alignments.append((node_id, [idx_txt]))

            # fix order

            node_id_list = [x[0] for x in node_alignments]
            order = np.argsort(node_id_list)

            node_alignments = [node_alignments[idx] for idx in order]
            b_align = b_align[order]
            argmax = argmax[order]

            # result

            info = {}
            info['node_alignments'] = node_alignments
            info['posterior'] = b_align
            info['argmax'] = argmax

            yield info
