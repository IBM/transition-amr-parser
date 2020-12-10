"""Decoder self-attention masks, e.g. graph structure for message passing."""
from packaging import version

import torch

from .graph_attention_masks import modify_mask_pre_post_softmax


def get_graph_self_attn_mask(tgt_actedge_masks,
                             tgt_actedge_1stnode_masks,
                             tgt_actedge_indexes,
                             tgt_actedge_cur_node_indexes,
                             tgt_actedge_cur_1stnode_indexes,
                             tgt_actedge_pre_node_indexes,
                             tgt_actedge_directions,
                             # graph structure to connect with all previous nodes
                             tgt_actedge_allpre_indexes,
                             tgt_actedge_allpre_pre_node_indexes,
                             tgt_actedge_allpre_directions,
                             # mask generation control
                             mask_num_heads,
                             num_heads,
                             tgt_graph_mask='1prev'
                             ):
    """Create the self-attention mask to embed the graph structure in the decoder self-attention. This embeds an GNN
    within the Transformer decoder self-attention. The mask could contain multiple heads.

    Args:
        tgt_actedge_masks (torch.Tensor): target input actions edge mask, size (batch_size, tgt_max_len).
            Note: padded with 0.
        tgt_actedge_1stnode_masks (torch.Tensor): target input actions 1st node mask, size (batch_size, tgt_max_len).
            Note: padded with 0.
        tgt_actedge_indexes (torch.Tensor): target input actions edge indexes, size (#edge_actions,) in the batch.
            Values are tied with the batch_size * tgt_max_len dimension.
        tgt_actedge_cur_node_indexes (torch.Tensor): target input actions edge --> current node indexes, size
            (#edge_actions,) in the batch. Values are tied with the tgt_max_len dimension.
        tgt_actedge_cur_1stnode_indexes (torch.Tensor): target input actions edge --> current node 1st position indexes,
            size (#edge_actions,) in the batch. Values are tied with the tgt_max_len dimension.
        tgt_actedge_pre_node_indexes (torch.Tensor): target input actions edge --> previous node indexes, size
            (#edge_actions,) in the batch. Values are tied with the tgt_max_len dimension.
        tgt_actedge_directions (torch.Tensor): target input actions edge --> directions, size (#edge_actions,) in the
            batch. 1: RA, 0: LA, -1: LA(root).
        tgt_actedge_allpre_indexes (torch.Tensor): target input actions to encode all the previous edges so far.
            Mask indexes for the batch_size * tgt_max_len dimension.
        tgt_actedge_allpre_pre_node_indexes (torch.Tensor): target input actions to encode all previous edges so far.
            Mask indexes for the tgt_max_len dimension.
        tgt_actedge_allpre_directions (torch.Tensor): target input actions to encode all previous edges so far.
            Directions of these edges. 1: RA, 0: LA, -1: LA(root).
        mask_num_heads (int): number of attention heads that are used for graph structure masking.
        num_heads (int): total number of attention heads.
        tgt_graph_mask (str, optional):
    """
    assert tgt_graph_mask in ['1prev', 'allprev', '1prev_in', 'allprev_in', '1prev_1in1out', 'allprev_1in1out']
    if tgt_graph_mask in ['1prev_1in1out', 'allprev_1in1out']:
        assert mask_num_heads == 2

    # node diagonal mask, size (batch_size, tgt_max_len, tgt_max_len)
    mask = torch.diag_embed(tgt_actedge_1stnode_masks)

    # encode the current node into the edge position; for input swap edges for nodes, this is just diagonal
    mask.view(-1, mask.size(-1))[tgt_actedge_indexes, tgt_actedge_cur_node_indexes] = 1

    if tgt_graph_mask == '1prev':
        # encode the 1 previous node based on the current edge
        mask.view(-1, mask.size(-1))[tgt_actedge_indexes, tgt_actedge_pre_node_indexes] = 1
    elif tgt_graph_mask == 'allprev':
        # encode all previous nodes based on all the edges so far for the current node
        mask.view(-1, mask.size(-1))[tgt_actedge_allpre_indexes, tgt_actedge_allpre_pre_node_indexes] = 1
    elif tgt_graph_mask == '1prev_in':
        # encode the 1 previous node based on the current edge, when the edge direction is incoming (RA)
        mask.view(-1, mask.size(-1))[tgt_actedge_indexes[tgt_actedge_directions == 1],
                                     tgt_actedge_pre_node_indexes[tgt_actedge_directions == 1]] = 1
    elif tgt_graph_mask == 'allprev_in':
        # encode all previous nodes based on all the edges so far for the current node, when edge directions are in (RA)
        mask.view(-1, mask.size(-1))[tgt_actedge_allpre_indexes[tgt_actedge_allpre_directions == 1],
                                     tgt_actedge_allpre_pre_node_indexes[tgt_actedge_allpre_directions == 1]] = 1
    elif tgt_graph_mask == '1prev_1in1out':
        # encode the 1 previous node based on the current edge, 1 head for incoming edges (RA)
        # and 1 head for outgoing edges (LA and LA(root))
        mask_in = mask.clone()
        mask_in.view(-1, mask_in.size(-1))[
            tgt_actedge_indexes[tgt_actedge_directions == 1],
            tgt_actedge_pre_node_indexes[tgt_actedge_directions == 1]
            ] = 1

        mask_out = mask.clone()
        mask_out.view(-1, mask_out.size(-1))[
            tgt_actedge_indexes[tgt_actedge_directions != 1],
            tgt_actedge_pre_node_indexes[tgt_actedge_directions != 1]
            ] = 1

        mask = torch.stack([mask_in, mask_out], dim=1)

    elif tgt_graph_mask == 'allprev_1in1out':
        # encode all previous nodes based on all the edges so far for the current node, 1 head for incoming edges (RA)
        # and 1 head for outgoing edges (LA and LA(root))
        mask_in = mask.clone()
        mask_in.view(-1, mask_in.size(-1))[
            tgt_actedge_allpre_indexes[tgt_actedge_allpre_directions == 1],
            tgt_actedge_allpre_pre_node_indexes[tgt_actedge_allpre_directions == 1]
            ] = 1

        mask_out = mask.clone()
        mask_out.view(-1, mask_out.size(-1))[
            tgt_actedge_allpre_indexes[tgt_actedge_allpre_directions != 1],
            tgt_actedge_allpre_pre_node_indexes[tgt_actedge_allpre_directions != 1]
            ] = 1

        mask = torch.stack([mask_in, mask_out], dim=1)

        """
        # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans
        # across two contiguous subspaces). Use .reshape(...) instead.
        # HOWEVER, .reshape(...) in this case does not change the original values.
        mask = mask.unsqueeze(1).repeat(1, 2, 1, 1)
        mask[:, 0, :, :].view(-1, mask.size(-1))[
            tgt_actedge_allpre_indexes[tgt_actedge_allpre_directions == 1],
            tgt_actedge_allpre_pre_node_indexes[tgt_actedge_allpre_directions == 1]
            ] = 1
        mask[:, 1, :, :].view(-1, mask.size(-1))[
            tgt_actedge_allpre_indexes[tgt_actedge_allpre_directions != 1],
            tgt_actedge_allpre_pre_node_indexes[tgt_actedge_allpre_directions != 1]
            ] = 1
        """
    else:
        raise NotImplementedError

    # put the mask into heads
    bsz_head_mask = tgt_actedge_masks.new_ones(tgt_actedge_masks.size(0), num_heads,
                                               tgt_actedge_masks.size(1), tgt_actedge_masks.size(1),
                                               dtype=torch.uint8)
    bsz_head_mask[:, :mask_num_heads, :, :] = mask.unsqueeze(1) if mask.dim() == 3 else mask    # else mask.dim() == 4
    bsz_head_mask = bsz_head_mask.reshape(-1, tgt_actedge_masks.size(1), tgt_actedge_masks.size(1))
    # size (batch_size * num_heads, tgt_max_len, tgt_max_len)

    # make the mask causal
    bsz_head_mask = torch.tril(bsz_head_mask)

    # modify the mask to prevent NAN
    bsz_head_mask, bsz_head_mask_post_softmax = modify_mask_pre_post_softmax(bsz_head_mask)

    # NOTE after modification, the `bsz_head_mask` contains all 1 rows which are not causal; but this is fine since
    #      the post mask will 0 out whatever those rows generate as distributions to combine values

    # for compatibility of PyTorch 1.1
    if version.parse(torch.__version__) >= version.parse('1.2.0'):
        bsz_head_mask = bsz_head_mask.bool()

    return bsz_head_mask, bsz_head_mask_post_softmax
