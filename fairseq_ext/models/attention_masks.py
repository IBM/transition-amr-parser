from packaging import version

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
    # assert tgt_src_align_focus[0] == 'p' and tgt_src_align_focus[2] == 'n'
    # prv = int(tgt_src_align_focus[1])    # number of previous alignment positions to include
    # nxt = int(tgt_src_align_focus[3])    # number of next alignment positions to include

    prv, nxt = parse_align_focus(tgt_src_align_focus)

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


def parse_align_focus(tgt_src_align_focus):
    """Parse the input 'tgt_src_align_focus' string list."""
    if isinstance(tgt_src_align_focus, str):
        # for compatibility reason from legacy models
        assert tgt_src_align_focus[0] == 'p' and tgt_src_align_focus[2] == 'n'
        prv = int(tgt_src_align_focus[1])    # number of previous alignment positions to include
        nxt = int(tgt_src_align_focus[3])    # number of next alignment positions to include
        return prv, nxt

    # assert isinstance(tgt_src_align_focus, list)

    num_focus = len(tgt_src_align_focus)
    if num_focus == 1:
        tgt_src_align_focus = tgt_src_align_focus[0]

        if len(tgt_src_align_focus) == 4:
            # for compatibility reason from legacy input format
            assert tgt_src_align_focus[0] == 'p' and tgt_src_align_focus[2] == 'n'
            prv = int(tgt_src_align_focus[1])    # number of previous alignment positions to include
            nxt = int(tgt_src_align_focus[3])    # number of next alignment positions to include
            return prv, nxt

        assert tgt_src_align_focus[0] == 'p' and tgt_src_align_focus[2] == 'c' and tgt_src_align_focus[4] == 'n'
        prv = int(tgt_src_align_focus[1])    # number of previous alignment positions to include
        nxt = int(tgt_src_align_focus[5])    # number of next alignment positions to include
        return prv, nxt
    else:
        raise NotImplementedError


def get_cross_attention_mask_heads(tgt_src_cursors, src_max_len, src_pad_mask, tgt_src_align_focus,
                                   mask_num_heads, num_heads):
    """Create the cross attention mask for decoder source attention. The mask contains multiple heads, where each head
    could have different focus.

    Args:
        tgt_src_cursors (torch.Tensor): target-source 1-1 alignments. Size (batch_size, tgt_max_len).
        src_max_len (int): source max length in the batch.
        src_pad_mask (torch.Tensor or None): source padding mask or None.
        tgt_src_align_focus (List[str]): masking focus for different heads, centering around the singular alignment
            position. Each focus takes the form 'p-c-n-' where 'p' is for 'previous', 'c' is for 'current' (alignment),
            and 'n' is for 'next', and '-' could be either a number or '*' which means 'all'.
        mask_num_heads (int): number of attention heads that are used for alignment masking.
        num_heads (int): total number of attention heads.

    NOTE
        The source batch is **left padded**... should be very careful...
    """
    assert len(tgt_src_align_focus) == mask_num_heads or len(tgt_src_align_focus) == 1

    bsz, tgt_max_len = tgt_src_cursors.size()
    if src_pad_mask is not None:
        src_num_pads = src_pad_mask.sum(dim=1).unsqueeze(1)
        tgt_src_cursors = tgt_src_cursors + src_num_pads    # NOTE this is key to left padding!

    align_mask_idx = torch.arange(src_max_len).view(1, 1, -1).to(tgt_src_cursors)

    # size (bsz, num_heads, tgt_max_len, src_max_len)
    bsz_head_mask = tgt_src_cursors.new_zeros(tgt_src_cursors.size(0), num_heads, tgt_src_cursors.size(1), src_max_len,
                                              dtype=torch.uint8)

    for i, focus in enumerate(tgt_src_align_focus):
        prv = focus[1]
        cur = focus[3]
        nxt = focus[5]

        if prv == '0' or prv == '-':
            mask_prv = 0
        elif prv == '*':
            mask_prv = align_mask_idx < tgt_src_cursors.unsqueeze(-1)
        else:
            mask_prv = 0
            for p in range(int(prv)):
                mask_prv += align_mask_idx == (tgt_src_cursors.unsqueeze(-1) - p)

        if cur == '0' or cur == '-':
            mask_cur = 0
        else:
            mask_cur = align_mask_idx == tgt_src_cursors.unsqueeze(-1)

        if nxt == '0' or nxt == '-':
            mask_nxt = 0
        elif nxt == '*':
            mask_nxt = align_mask_idx > tgt_src_cursors.unsqueeze(-1)
        else:
            mask_nxt = 0
            for n in range(int(nxt)):
                mask_nxt += align_mask_idx == (tgt_src_cursors.unsqueeze(-1) + n)

        # size (bsz, tgt_max_len, src_max_len)
        align_mask = mask_prv + mask_cur + mask_nxt

        bsz_head_mask[:, i, :, :] = align_mask

    # when length is 1, all masking heads use the same mask
    if len(tgt_src_align_focus) == 1 and mask_num_heads > 1:
        bsz_head_mask[:, 1:mask_num_heads, :, :] = align_mask.unsqueeze(1)

    # the remaining heads: keep all
    bsz_head_mask[:, mask_num_heads:, :, :] = 1

    bsz_head_mask = bsz_head_mask.reshape(-1, tgt_src_cursors.size(1), src_max_len)

    # NOTE when one row out of bsz * num_heads (tgt_max_len, src_max_len) masks is full zeros, after softmax the
    # distribution will be all "nan"s, which will cause problem when calculating gradients.
    # Thus, we mask these positions after softmax
    bsz_head_mask_post_softmax = bsz_head_mask.new_ones(*bsz_head_mask.size()[:2], 1, dtype=torch.float)
    bsz_head_mask_post_softmax[bsz_head_mask.sum(dim=2) == 0] = 0
    # we need to modify the pre-softmax as well, since after we get nan, multiplying by 0 is still nan
    bsz_head_mask[(bsz_head_mask.sum(dim=2, keepdim=True) == 0).repeat(1, 1, src_max_len)] = 1

    # NOTE must use torch.bool for mask for PyTorch >= 1.2, otherwise there will be problems around ~mask
    # for compatibility of PyTorch 1.1
    if version.parse(torch.__version__) < version.parse('1.2.0'):
        return bsz_head_mask, bsz_head_mask_post_softmax
    return bsz_head_mask.to(torch.bool), bsz_head_mask_post_softmax
