import torch


def make_bsz_tgt_src_align_mask(tgt_src_cursors, src_max_len, src_pad_mask=None):
    """Get the batched target-source alignment mask.

    NOTE
        The source batch is **left padded**... should be very careful...
    """
    bsz, tgt_max_len = tgt_src_cursors.size()
    if src_pad_mask is not None:
        src_num_pads = src_pad_mask.sum(dim=1).unsqueeze(1)
        tgt_src_cursors = tgt_src_cursors + src_num_pads    # NOTE this is key to left padding!

    tgt_src_align_mask = tgt_src_cursors.new_zeros(bsz, tgt_max_len, src_max_len)

    tgt_seq_batch_idx = torch.arange(tgt_max_len).repeat(bsz, 1) + (torch.arange(bsz) * tgt_max_len).view(-1, 1)
    tgt_seq_batch_idx = tgt_seq_batch_idx.to(tgt_src_cursors)    # size (bsz, tgt_max_len)

    tgt_src_align_mask.view(-1, src_max_len)[tgt_seq_batch_idx.view(-1), tgt_src_cursors.view(-1)] = 1

    return tgt_src_align_mask


def get_cross_attention_mask(tgt_src_cursors, src_max_len, src_pad_mask, tgt_src_align_focus, num_heads):
    """Create the cross attention mask for decoder source attention.

    The padding target positions are not taken care of. The target padding mask should be taken care outside.
    """
    assert tgt_src_align_focus[0] == 'p' and tgt_src_align_focus[2] == 'n'
    prv = int(tgt_src_align_focus[1])    # number of previous alignment positions to include
    nxt = int(tgt_src_align_focus[3])    # number of next alignment positions to include

    tgt_src_align_mask = make_bsz_tgt_src_align_mask(tgt_src_cursors, src_max_len, src_pad_mask)

    for p in range(prv):
        tgt_src_cursors_prv = torch.zeros_like(tgt_src_cursors)
        # NOTE here we need to be very careful about left padding
        tgt_src_cursors_prv[tgt_src_cursors >= p + 1] = tgt_src_cursors[tgt_src_cursors >= p + 1] - (p + 1)
        tgt_src_align_mask_prv = make_bsz_tgt_src_align_mask(tgt_src_cursors_prv, src_max_len, src_pad_mask)
        tgt_src_align_mask += tgt_src_align_mask_prv

    for n in range(nxt):
        tgt_src_cursors_nxt = torch.zeros_like(tgt_src_cursors)
        # NOTE here we need to be very careful about left padding
        if src_pad_mask is not None:
            src_lens = (src_pad_mask == 0).sum(dim=1)
            tgt_src_cursors_nxt[tgt_src_cursors <= src_lens.view(-1, 1) - n - 2] = \
                tgt_src_cursors[tgt_src_cursors <= src_lens.view(-1, 1) - n - 2] + n + 1
        else:
            tgt_src_cursors_nxt[tgt_src_cursors <= src_max_len - n - 2] = \
                tgt_src_cursors[tgt_src_cursors <= src_max_len - n - 2] + n + 1
        tgt_src_align_mask_nxt = make_bsz_tgt_src_align_mask(tgt_src_cursors_nxt, src_max_len, src_pad_mask)
        tgt_src_align_mask += tgt_src_align_mask_nxt

    if prv or nxt:
        tgt_src_align_mask[tgt_src_align_mask >= 1] = 1

    bsz_head_mask = tgt_src_cursors.new_zeros(tgt_src_cursors.size(0), num_heads, tgt_src_cursors.size(1), src_max_len,
                                              dtype=torch.uint8)
    bsz_head_mask[:, 0, :, :] = tgt_src_align_mask    # first head use the mask
    bsz_head_mask[:, 1:, :, :] = 1    # other heads keep all

    bsz_head_mask = bsz_head_mask.reshape(-1, tgt_src_cursors.size(1), src_max_len)

    return bsz_head_mask
