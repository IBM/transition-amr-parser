# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data import FairseqDataset

from fairseq_ext.data import data_utils
from fairseq_ext.data.data_utils import (
    collate_embeddings,
    collate_target_masks,
    collate_wp_idx,
    collate_masks
)


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, state_machine=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_tgt_pos(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            -2, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we will need for sanity check furtehr down
        no_sorted_target = target.clone()
        target = target.index_select(0, sort_order)
        # needed for sanity checks
        tgt_legths = torch.LongTensor([len(s['target']) for s in samples])
        ntokens = tgt_legths.sum().item()
        tgt_legths = tgt_legths.index_select(0, sort_order)

        tgt_pos = merge_tgt_pos('tgt_pos', left_pad=left_pad_target)
        tgt_pos = tgt_pos.index_select(0, sort_order)
        # NOTE we do not need to shift target actions pointer since it is associated with the out sequence

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # Pre-trained embeddings
    def merge_embeddings(key, left_pad_source, move_eos_to_beginning):
        return collate_embeddings(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad_source, move_eos_to_beginning
        )

    source_fix_emb = merge_embeddings(
        'source_fix_emb',
        left_pad_source,
        False
        #left_pad_target
    )
    source_fix_emb = source_fix_emb.index_select(0, sort_order)

    # Word-pieces
    src_wordpieces = merge('src_wordpieces', left_pad=left_pad_source)
    src_wordpieces = src_wordpieces.index_select(0, sort_order)

    def merge_wp_idx(key, left_pad, move_eos_to_beginning=False):
        return collate_wp_idx(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            reverse=True
        )

    # Wordpiece to word mapping
    src_wp2w = merge_wp_idx('src_wp2w', left_pad=left_pad_source)
    src_wp2w = src_wp2w.index_select(0, sort_order)

#     # DEBUG: Inline RoBERTa
#     from torch_scatter import scatter_mean
#     # extract roberta from collated
#     roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
#     roberta.eval()
#     last_layer = roberta.extract_features(src_wordpieces)
#     # remove sentence start
#     bsize, max_len, emb_size = last_layer.shape
#     mask = (src_wordpieces != 0).unsqueeze(2).expand(last_layer.shape)
#     last_layer = last_layer[mask].view((bsize, max_len - 1, emb_size))
#     # remove sentence end
#     last_layer = last_layer[:, :-1, :]
#     # apply scatter, flip before to have left-side padding
#     source_fix_emb2 = scatter_mean(last_layer, src_wp2w.unsqueeze(2), dim=1)
#     source_fix_emb2 = source_fix_emb2.flip(1)
#     # Remove extra padding
#     source_fix_emb2 = source_fix_emb2[:, -src_tokens.shape[1]:, :]
#     abs(source_fix_emb2 - source_fix_emb).max()
#     # DEBUG: Inline RoBERTa

    # source masks
    def merge_masks(key, left_pad_source, left_pad_target):
        return collate_masks(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad_source, left_pad_target
        )

    # target masks
    # get sub-set of active logits for this batch and mask for each individual
    # sentence and target time step
    def merge_target_masks(left_pad_target):
        return collate_target_masks(
            [(s['target_masks'], s['active_logits'], len(s['target'])) for s in samples],
            pad_idx, eos_idx, left_pad_target=left_pad_target, move_eos_to_beginning=False,
            target=no_sorted_target
        )

    # TODO change later
    state_machine = False

    if state_machine:

        # stack info
        memory = merge_masks('memory', left_pad_source, left_pad_target)
        memory = memory.index_select(0, sort_order)
        memory_pos = merge_masks('memory_pos', left_pad_source, left_pad_target)
        memory_pos = memory_pos.index_select(0, sort_order)
        # active logits
        logits_mask, logits_indices = merge_target_masks(left_pad_target)
        logits_mask = logits_mask.index_select(0, sort_order)

    else:

        memory = None
        memory_pos = None
        logits_indices = None
        logits_mask = None


    def merge_target_vocab_masks():
        # default right padding
        # TODO organize the code here
        masks = [s['tgt_vocab_masks'] for s in samples]
        max_len = max([len(m) for m in masks])
        merged = masks[0].new(len(masks), max_len, masks[0].size(1)).fill_(pad_idx)
        for i, v in enumerate(masks):
            merged[i, :v.size(0), :] = v
        return merged

    tgt_vocab_masks = merge_target_vocab_masks()
    tgt_vocab_masks = tgt_vocab_masks.index_select(0, sort_order)
    tgt_actnode_masks = merge('tgt_actnode_masks', left_pad=left_pad_target)
    tgt_actnode_masks = tgt_actnode_masks.index_select(0, sort_order)
    tgt_srctok_cursors = merge('tgt_srctok_cursors', left_pad=left_pad_target)
    tgt_srctok_cursors = tgt_srctok_cursors.index_select(0, sort_order)

    # batch variables
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'source_fix_emb': source_fix_emb,
            'src_wordpieces': src_wordpieces,
            'src_wp2w': src_wp2w,
            'memory': memory,
            'memory_pos': memory_pos,
            'logits_mask': logits_mask,
            'logits_indices': logits_indices,
            'tgt_vocab_masks': tgt_vocab_masks,
            'tgt_actnode_masks': tgt_actnode_masks,
            'tgt_srctok_cursors': tgt_srctok_cursors
        },
        'target': target,
        'tgt_pos': tgt_pos
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # sanity check batch
    # from fairseq.debug_tools import sanity_check_collated_batch
    # sanity_check_collated_batch(batch, pad_idx, left_pad_source, left_pad_target, tgt_legths)

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        src_fix_emb, src_fix_emb_sizes,
        src_wordpieces, src_wordpieces_sizes,
        src_wp2w, src_wp2w_sizes,
        tgt, tgt_sizes, tgt_dict,
        tgt_pos, tgt_pos_sizes,
        # memory, memory_sizes,
        # mem_pos, mem_pos_sizes,
        # target_masks, target_masks_sizes,
        # active_logits, active_logits_sizes,
        target_vocab_masks, target_action_node_masks, target_src_token_cursor,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        state_machine=True
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tgt_pos = tgt_pos
        self.tgt_pos_sizes = tgt_pos_sizes
        # dataset variables
        self.src_fix_emb = src_fix_emb
        self.src_fix_emb_sizes = src_fix_emb_sizes
        self.src_wordpieces = src_wordpieces
        self.src_wordpieces_sizes = src_wordpieces_sizes
        self.src_wp2w = src_wp2w
        self.src_wp2w_sizes = src_wp2w_sizes
        # self.memory = memory
        # self.mem_pos = mem_pos
        # self.memory_sizes = np.array(memory_sizes)
        # self.mem_pos_sizes = np.array(mem_pos_sizes)
        # self.target_masks = target_masks
        # self.target_masks_sizes = np.array(target_masks_sizes)
        # self.active_logits = active_logits
        # self.active_logits_sizes = np.array(active_logits_sizes)
        # amr actions state information
        self.target_vocab_masks = target_vocab_masks
        self.target_action_node_masks = target_action_node_masks
        self.target_src_token_cursor = target_src_token_cursor
        # other
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        # compute or not state of state machine
        self.state_machine = state_machine

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        tgt_pos_item = self.tgt_pos[index]

        # Deduce pretrained embeddings size
        pretrained_embed_dim = self.src_fix_emb[index].shape[0] // src_item.shape[0]
        shape_factor = (self.src_fix_emb[index].shape[0] // pretrained_embed_dim, pretrained_embed_dim)
        src_fix_emb_item = self.src_fix_emb[index].view(*shape_factor)
        src_wordpieces_item = self.src_wordpieces[index]
        src_wp2w_item = self.src_wp2w[index]
        shape_factor = (tgt_item.shape[0], src_item.shape[0])
        # memory_item = self.memory[index].view(*shape_factor).transpose(0, 1)
        # memory_pos_item = self.mem_pos[index].view(*shape_factor).transpose(0, 1)
        # target_masks = self.target_masks[index]
        # active_logits = self.active_logits[index]

        # Cast to float to simplify mask manipulation
        # memory_item = memory_item.type(src_fix_emb_item.type())
        # memory_pos_item = memory_pos_item.type(src_fix_emb_item.type())
        # target_masks = target_masks.type(src_fix_emb_item.type())
        # active_logits = active_logits.type(src_fix_emb_item.type())

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'source_fix_emb': src_fix_emb_item,
            'src_wordpieces': src_wordpieces_item,
            'src_wp2w': src_wp2w_item,
            'target': tgt_item,
            'tgt_pos': tgt_pos_item,
            # 'memory': memory_item,
            # 'memory_pos': memory_pos_item,
            # 'target_masks': target_masks,
            # 'active_logits': active_logits
            'tgt_vocab_masks': self.target_vocab_masks[index],
            'tgt_actnode_masks': self.target_action_node_masks[index],
            'tgt_srctok_cursors': self.target_src_token_cursor[index]
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            state_machine=self.state_machine
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.src_fix_emb, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
            # and getattr(self.memory, 'supports_prefetch', False)
            # and getattr(self.mem_pos, 'supports_prefetch', False)
            and getattr(self.tgt_pos, 'supports_prefetch', False)

        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
            self.tgt_pos.prefetch(indices)
        self.src_fix_emb.prefetch(indices)
        # self.memory.prefetch(indices)
        # self.mem_pos.prefetch(indices)
