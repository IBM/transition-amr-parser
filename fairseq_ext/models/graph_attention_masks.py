"""Decoder self-attention masks, e.g. graph structure."""
from packaging import version

import torch


def modify_mask_pre_post_softmax(mask):
    """Modify the attention mask to a pre-softmax mask and a post-softmax mask.
    The issue is that when one row out of bsz * num_heads (tgt_max_len, src_max_len) masks is full zeros,
    after softmax the distribution will be all "nan"s, which will cause problem when calculating gradients.

    Thus, we mask out these rows after softmax by multiplying them with 0;
    and we need to modify the pre-softmax as well, since after we get nan, multiplying by 0 is still nan.

    Args:
        mask (torch.Tensor): binary attention mask of size (batch_size * num_heads, tgt_max_len, src_max_len).

    Returns:
        mask_pre_softmax (torch.BoolTensor): binary attention mask of the same size, used pre-softmax.
        mask_post_softmax (torch.FloatTensor): binary attention mask of size (batch_size * num_heads, tgt_max_len),
            to multiple after the softmax.
    """
    mask_post_softmax = mask.new_ones(*mask.size()[:2], 1, dtype=torch.float)
    mask_post_softmax[mask.sum(dim=2) == 0] = 0
    # modify the pre-softmax mask
    mask[(mask.sum(dim=2, keepdim=True) == 0).repeat(1, 1, mask.size(-1))] = 1
    return mask.to(torch.bool), mask_post_softmax


def get_graph_self_attn_mask(tgt_actedge_masks, tgt_actedge_cur_nodes,
                             tgt_actedge_pre_nodes, tgt_actedge_directions,
                             tgt_actnode_masks_shift,
                             mask_num_heads, num_heads,
                             tgt_graph_mask='e1c1p1'):
    """Create the mask to encode graph structure for decoder self-attention. The mask could contain multiple heads.

    Args:
        tgt_actedge_masks (torch.Tensor): target input actions edge mask, size (batch_size, tgt_max_len).
            Note: padded with 0.
        tgt_actedge_cur_nodes (torch.Tensor): target input actions edge current node, size (batch_size, tgt_max_len).
            Note: padded with -1, root node denoted as -2 (which is not in the action sequence).
        tgt_actedge_pre_nodes (torch.Tensor): target input actions edge previous node, size (batch_size, tgt_max_len).
            Note: padded with -1.
        tgt_actedge_directions (torch.Tensor): target input actions edge directions, size (batch_size, tgt_max_len).
            Note: padded with 0, and 1 for pre -> cur ('RA') and -1 for pre <- cur ('LA').
        tgt_actnode_masks_shift (torch.Tensor): target input actions node mask, size (batch_size, tgt_max_len).
            Note: padded with 0.
        mask_num_heads (int): number of attention heads that are used for graph structure masking.
        num_heads (int): total number of attention heads.
        tgt_graph_mask (str): masking strategy for the graph edge positions. It takes the form 'e-c-p-' where 'e' is for
            the edge, 'c' is for the current node on the edge, 'p' is for the previous node on the edge, and '-' can be
            either '0' or '1', indicating the masking value.
    """
    assert len(tgt_graph_mask) == 6
    assert tgt_graph_mask[0] == 'e' and tgt_graph_mask[2] == 'c' and tgt_graph_mask[4] == 'p'

    # node diagonal mask, size (batch_size, tgt_max_len, tgt_max_len)
    mask_node = torch.diag_embed(tgt_actnode_masks_shift)

    # edge diagonal mask, size (batch_size, tgt_max_len, tgt_max_len)
    if tgt_graph_mask[1] == '1':
        mask_edge = torch.diag_embed(tgt_actedge_masks)
    else:
        mask_edge = torch.diag_embed(torch.zeros_like(tgt_actedge_masks))

    # make the mask: at edge position, attend to the edge and two nodes
    if tgt_graph_mask[3] == '1':
        # indexes of the edges in a flattened vector of size (batch_size * tgt_max_len,)
        # NOTE root node is not added by any action, the cur_node is denoted as -2
        edge_idx_flattened_noroot = torch.nonzero(tgt_actedge_cur_nodes.flatten() >= 0).squeeze()
        mask_edge.view(-1, mask_edge.size(-1))[edge_idx_flattened_noroot,
                                               tgt_actedge_cur_nodes.flatten()[edge_idx_flattened_noroot]] = 1
    if tgt_graph_mask[5] == '1':
        # indexes of the edges in a flattened vector of size (batch_size * tgt_max_len,)
        edge_idx_flattened = torch.nonzero(tgt_actedge_masks.flatten()).squeeze()
        mask_edge.view(-1, mask_edge.size(-1))[edge_idx_flattened,
                                               tgt_actedge_pre_nodes.flatten()[edge_idx_flattened]] = 1

    mask = mask_edge + mask_node    # size (batch_size, tgt_max_len, tgt_max_len)

    # put the mask into heads
    bsz_head_mask = tgt_actedge_masks.new_ones(tgt_actedge_masks.size(0), num_heads,
                                               tgt_actedge_masks.size(1), tgt_actedge_masks.size(1),
                                               dtype=torch.uint8)
    bsz_head_mask[:, :mask_num_heads, :, :] = mask.unsqueeze(1)
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
