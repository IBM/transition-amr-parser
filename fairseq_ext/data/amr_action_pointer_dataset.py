import numpy as np
import torch
from fairseq.data import FairseqDataset

from fairseq_ext.data.data_utils import (
    collate_embeddings,
    collate_wp_idx,
    # collate_target_masks,
    # collate_masks
)
from fairseq_ext.data import data_utils


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
    collate_tgt_states=False,
    collate_tgt_states_graph=False,
    pad_tgt_actedge_cur_nodes=-1,
    pad_tgt_actedge_pre_nodes=-1,
    pad_tgt_actedge_directions=0
):
    assert not left_pad_target

    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_idx=pad_idx):
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

    if isinstance(samples[0]['source'], torch.Tensor):
        # source tokens are numerical values encoded from a src dictionary
        src_tokens = merge('source', left_pad=left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        # sort by descending source length
        src_lengths, sort_order = src_lengths.sort(descending=True)
        src_tokens = src_tokens.index_select(0, sort_order)
    else:
        # source tokens are original string form, which is only a place holder since the embeddings are directly used
        # assert samples[0].get('source_fix_emb', None) is not None
        src_tokens = [s['source'] for s in samples]
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])
        # sort by descending source length
        src_lengths, sort_order = src_lengths.sort(descending=True)
        src_tokens = [src_tokens[i] for i in sort_order]

    id = id.index_select(0, sort_order)

    src_sents = [s['src_tokens'] for s in samples]
    if src_sents[0] is None:
        src_sents = None
    else:
        src_sents = [src_sents[i] for i in sort_order]

    prev_output_tokens = None
    target = None
    tgt_pos = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we will need for sanity check further down
        no_sorted_target = target.clone()
        target = target.index_select(0, sort_order)
        # needed for sanity checks
        tgt_legths = torch.LongTensor([len(s['target']) for s in samples])
        ntokens = tgt_legths.sum().item()
        tgt_legths = tgt_legths.index_select(0, sort_order)

        tgt_pos = merge_tgt_pos('tgt_pos', left_pad=left_pad_target)
        tgt_pos = tgt_pos.index_select(0, sort_order)
        # NOTE we do not need to shift target actions pointer since it is associated with the out sequence

        if samples[0].get('tgt_in', None) is not None:
            # NOTE we do not shift here, as it is already shifter 1 position to the right in dataset .__getitem__
            tgt_in = merge('tgt_in', left_pad=left_pad_target)
            prev_output_tokens = tgt_in = tgt_in.index_select(0, sort_order)

        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        else:
            raise ValueError
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # Pre-trained embeddings
    def merge_embeddings(key, left_pad_source, move_eos_to_beginning=False):
        return collate_embeddings(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad_source, move_eos_to_beginning
        )

    if samples[0].get('source_fix_emb', None) is not None:
        source_fix_emb = merge_embeddings('source_fix_emb', left_pad_source)
        source_fix_emb = source_fix_emb.index_select(0, sort_order)
    else:
        source_fix_emb = None

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

    # # source masks
    # def merge_masks(key, left_pad_source, left_pad_target):
    #     return collate_masks(
    #         [s[key] for s in samples],
    #         pad_idx, eos_idx, left_pad_source, left_pad_target
    #     )

    # # target masks
    # # get sub-set of active logits for this batch and mask for each individual
    # # sentence and target time step
    # def merge_target_masks(left_pad_target):
    #     return collate_target_masks(
    #         [(s['target_masks'], s['active_logits'], len(s['target'])) for s in samples],
    #         pad_idx, eos_idx, left_pad_target=left_pad_target, move_eos_to_beginning=False,
    #         target=no_sorted_target
    #     )

    # # TODO change later
    # state_machine = False

    # if state_machine:

    #     # stack info
    #     memory = merge_masks('memory', left_pad_source, left_pad_target)
    #     memory = memory.index_select(0, sort_order)
    #     memory_pos = merge_masks('memory_pos', left_pad_source, left_pad_target)
    #     memory_pos = memory_pos.index_select(0, sort_order)
    #     # active logits
    #     logits_mask, logits_indices = merge_target_masks(left_pad_target)
    #     logits_mask = logits_mask.index_select(0, sort_order)

    # else:

    #     memory = None
    #     memory_pos = None
    #     logits_indices = None
    #     logits_mask = None

    # TODO legacy from stack-transformer to make sure the model runs; remove later
    memory = None
    memory_pos = None
    logits_indices = None
    logits_mask = None

    # TODO write a function to collate 2-D matrices, similar to the collate_tokens function
    def merge_tgt_vocab_masks():
        # default right padding: left_pad_target should be False
        # TODO organize the code here
        masks = [s['tgt_vocab_masks'] for s in samples]
        max_len = max([len(m) for m in masks])
        merged = masks[0].new(len(masks), max_len, masks[0].size(1)).fill_(pad_idx)
        for i, v in enumerate(masks):
            merged[i, :v.size(0), :] = v
        return merged

    if collate_tgt_states:
        tgt_vocab_masks = merge_tgt_vocab_masks()
        tgt_vocab_masks = tgt_vocab_masks.index_select(0, sort_order)
        tgt_actnode_masks = merge('tgt_actnode_masks', left_pad=left_pad_target, pad_idx=0)
        tgt_actnode_masks = tgt_actnode_masks.index_select(0, sort_order)
        tgt_src_cursors = merge('tgt_src_cursors', left_pad=left_pad_target)
        tgt_src_cursors = tgt_src_cursors.index_select(0, sort_order)
    else:
        tgt_vocab_masks = None
        tgt_actnode_masks = None
        tgt_src_cursors = None

    if collate_tgt_states_graph:
        # graph structure (NOTE the pad_idx is fixed at some special values)
        tgt_actedge_masks = merge('tgt_actedge_masks', left_pad=left_pad_target, pad_idx=0)
        tgt_actedge_masks = tgt_actedge_masks.index_select(0, sort_order)
        tgt_actedge_cur_nodes = merge('tgt_actedge_cur_nodes', left_pad=left_pad_target,
                                      pad_idx=pad_tgt_actedge_cur_nodes)
        tgt_actedge_cur_nodes = tgt_actedge_cur_nodes.index_select(0, sort_order)
        tgt_actedge_pre_nodes = merge('tgt_actedge_pre_nodes', left_pad=left_pad_target,
                                      pad_idx=pad_tgt_actedge_pre_nodes)
        tgt_actedge_pre_nodes = tgt_actedge_pre_nodes.index_select(0, sort_order)
        tgt_actedge_directions = merge('tgt_actedge_directions', left_pad=left_pad_target,
                                       pad_idx=pad_tgt_actedge_directions)
        tgt_actedge_directions = tgt_actedge_directions.index_select(0, sort_order)
        tgt_actnode_masks_shift = merge('tgt_actnode_masks_shift', left_pad=left_pad_target, pad_idx=0)
        tgt_actnode_masks_shift = tgt_actnode_masks_shift.index_select(0, sort_order)
    else:
        # graph structure
        tgt_actedge_masks = None
        tgt_actedge_cur_nodes = None
        tgt_actedge_pre_nodes = None
        tgt_actedge_directions = None
        tgt_actnode_masks_shift = None

    force_actions = None
    if 'force_actions' in samples[0]:
        f_actions = [s['force_actions'] for s in samples]
        force_actions = [f_actions[i] for i in sort_order]
        
    # batch variables
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,    # target side number of tokens in the batch
        'src_sents': src_sents,    # original source sentences
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_fix_emb': source_fix_emb,
            'src_wordpieces': src_wordpieces,
            'src_wp2w': src_wp2w,
            # AMR actions states
            'tgt_vocab_masks': tgt_vocab_masks,
            'tgt_actnode_masks': tgt_actnode_masks,
            'tgt_src_cursors': tgt_src_cursors,
            # graph structure
            'tgt_actedge_masks': tgt_actedge_masks,
            'tgt_actedge_cur_nodes': tgt_actedge_cur_nodes,
            'tgt_actedge_pre_nodes': tgt_actedge_pre_nodes,
            'tgt_actedge_directions': tgt_actedge_directions,
            'tgt_actnode_masks_shift': tgt_actnode_masks_shift
        },
        'target': target,
        'force_actions': force_actions,
        'tgt_pos': tgt_pos
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # sanity check batch
    # from fairseq.debug_tools import sanity_check_collated_batch
    # sanity_check_collated_batch(batch, pad_idx, left_pad_source, left_pad_target, tgt_legths)

    return batch


class AMRActionPointerDataset(FairseqDataset):
    """Dataset for AMR transition-pointer parsing: source is English sentences, and target is action sequences, along
    with pointer values for arc actions where the pointers are on the action sequence.

    Args:
        FairseqDataset ([type]): [description]

    Note:
        - when we are using RoBERTa embeddings for the source, there is no need for "src_dict".
    """
    def __init__(self, *,
                 # src
                 src_tokens=None,
                 src=None, src_sizes=None, src_dict=None,
                 src_fix_emb=None, src_fix_emb_sizes=None, src_fix_emb_use=False,
                 src_wordpieces=None, src_wordpieces_sizes=None,
                 src_wp2w=None, src_wp2w_sizes=None,
                 # tgt
                 tgt=None,
                 tgt_force_actions=None,
                 tgt_sizes=None,
                 tgt_in=None,
                 tgt_in_sizes=None,
                 tgt_dict=None,
                 tgt_pos=None,
                 tgt_pos_sizes=None,
                 # core state info
                 tgt_vocab_masks=None,
                 tgt_actnode_masks=None,    # for the valid pointer positions
                 tgt_src_cursors=None,
                 # side state info for graph
                 tgt_actedge_masks=None,
                 tgt_actedge_cur_nodes=None,
                 tgt_actedge_pre_nodes=None,
                 tgt_actedge_directions=None,
                 # batching
                 left_pad_source=True, left_pad_target=False,
                 max_source_positions=1024, max_target_positions=1024,
                 shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
                 collate_tgt_states=True,
                 collate_tgt_states_graph=False,
                 ):

        if tgt_dict is not None and src_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        assert src is not None
        assert tgt is not None
        assert tgt_pos is not None
        assert src_sizes is not None

        assert not left_pad_target    # this should be fixed as in collate function it is by default for tgt_vocab_masks

        # core dataset variables
        self.src_tokens = src_tokens
        self.src = src
        self.tgt = tgt
        self.tgt_force_actions = tgt_force_actions
        self.tgt_in = tgt_in
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.tgt_in_sizes = np.array(tgt_in_sizes) if tgt_in_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tgt_pos = tgt_pos
        self.tgt_pos_sizes = tgt_pos_sizes

        # additional dataset variables

        # src RoBERTa embeddings
        self.src_fix_emb = src_fix_emb
        self.src_fix_emb_sizes = src_fix_emb_sizes

        self.src_fix_emb_use = src_fix_emb_use    # whether to use fix pretrained embeddings for src

        self.src_wordpieces = src_wordpieces
        self.src_wordpieces_sizes = src_wordpieces_sizes
        self.src_wp2w = src_wp2w
        self.src_wp2w_sizes = src_wp2w_sizes

        # AMR actions state information for each tgt step
        self.tgt_vocab_masks = tgt_vocab_masks
        self.tgt_actnode_masks = tgt_actnode_masks
        self.tgt_src_cursors = tgt_src_cursors

        # AMR graph structure information for each tgt step
        self.tgt_actedge_masks = tgt_actedge_masks
        self.tgt_actedge_cur_nodes = tgt_actedge_cur_nodes
        self.tgt_actedge_pre_nodes = tgt_actedge_pre_nodes
        self.tgt_actedge_directions = tgt_actedge_directions

        # others for collating examples to a batch
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

        # whether to include target actions states information during batching
        self.collate_tgt_states = collate_tgt_states
        self.collate_tgt_states_graph = collate_tgt_states_graph
        if self.collate_tgt_states:
            assert self.tgt_vocab_masks is not None
            assert self.tgt_actnode_masks is not None
            assert self.tgt_src_cursors is not None

        if self.collate_tgt_states_graph:
            # NOTE graph structure is also required here
            assert self.tgt_actedge_masks is not None
            assert self.tgt_actedge_cur_nodes is not None
            assert self.tgt_actedge_pre_nodes is not None
            assert self.tgt_actedge_directions is not None

            # get the padding values for the graph structure vectors
            # since they are not all 0s, e.g. for node action ids 0 is an legit id
            for i in range(len(self.tgt_actedge_masks[0])):
                if self.tgt_actedge_masks[0][i] == 0:
                    self.pad_tgt_actedge_cur_nodes = self.tgt_actedge_cur_nodes[0][i]
                    self.pad_tgt_actedge_pre_nodes = self.tgt_actedge_pre_nodes[0][i]
                    self.pad_tgt_actedge_directions = self.tgt_actedge_directions[0][i]
                    break

            # TODO lift this; associated with AMRStateMachine:apply_canonical_action()
            assert self.pad_tgt_actedge_cur_nodes != -2, 'currently -2 is fixed to denote root node for LA(root)'
        else:
            self.pad_tgt_actedge_cur_nodes = None
            self.pad_tgt_actedge_pre_nodes = None
            self.pad_tgt_actedge_directions = None

    def __getitem__(self, index):
        src_tokens_item = self.src_tokens[index] if self.src_tokens is not None else None
        src_item = self.src[index] if self.src is not None else None
        tgt_item = self.tgt[index]
        force_actions_item = self.tgt_force_actions[index] if self.tgt_force_actions else None
        tgt_in_item = self.tgt_in[index] if self.tgt_in is not None else None
        tgt_pos_item = self.tgt_pos[index]

        src_wordpieces_item = self.src_wordpieces[index].type(torch.long)    # TODO type conversion here is needed
        src_wp2w_item = self.src_wp2w[index].type(torch.long)
        # TODO type conversion above is needed; change in preprocessing while saving data

        # Deduce pretrained embeddings size; the src_fix_emb is NOT averaged to words
        if self.src_fix_emb_use:
            pretrained_embed_dim = self.src_fix_emb[index].shape[0] // len(src_wordpieces_item)
            shape_factor = (self.src_fix_emb[index].shape[0] // pretrained_embed_dim, pretrained_embed_dim)
            src_fix_emb_item = self.src_fix_emb[index].view(*shape_factor)
        else:
            src_fix_emb_item = None

        # shape_factor = (tgt_item.shape[0], src_item.shape[0])
        # memory_item = self.memory[index].view(*shape_factor).transpose(0, 1)
        # memory_pos_item = self.mem_pos[index].view(*shape_factor).transpose(0, 1)
        # target_masks = self.target_masks[index]
        # active_logits = self.active_logits[index]

        # Cast to float to simplify mask manipulation
        # memory_item = memory_item.type(src_fix_emb_item.type())
        # memory_pos_item = memory_pos_item.type(src_fix_emb_item.type())
        # target_masks = target_masks.type(src_fix_emb_item.type())
        # active_logits = active_logits.type(src_fix_emb_item.type())

        # reshape the target vocabulary mask to be 2-D
        tgt_vocab_mask_item = self.tgt_vocab_masks[index].view(-1, len(self.tgt_dict)) \
            if self.tgt_vocab_masks is not None else None
        tgt_actnode_mask_item = self.tgt_actnode_masks[index] if self.tgt_actnode_masks is not None else None
        tgt_src_cursor_item = self.tgt_src_cursors[index] if self.tgt_src_cursors is not None else None
        # graph structure
        tgt_actedge_masks_item = self.tgt_actedge_masks[index] \
            if self.tgt_actedge_masks is not None else None
        tgt_actedge_cur_nodes_item = self.tgt_actedge_cur_nodes[index] \
            if self.tgt_actedge_cur_nodes is not None else None
        tgt_actedge_pre_nodes_item = self.tgt_actedge_pre_nodes[index] \
            if self.tgt_actedge_pre_nodes is not None else None
        tgt_actedge_directions_item = self.tgt_actedge_directions[index] \
            if self.tgt_actedge_directions is not None else None
        tgt_actnode_mask_shift_item = None

        # ========== shift the tgt input vector to be fed into model ==========
        # NOTE we must clone first -- modification in their original place is not allowed
        # if not using .clone(), there is an error:
        #      RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to
        #      a single memory location. Please clone() the tensor before performing the operation.
        if tgt_in_item is not None:
            tgt_in_item = tgt_in_item.clone()
            tgt_in_item[1:] = tgt_in_item[:-1].clone()
            # for the <s> position at the beginning
            tgt_in_item[0] = self.tgt_dict.eos()

        # =====================================================================

        # ========== tgt graph structure information should be tied with the tgt input ==========
        if self.collate_tgt_states_graph:
            # a) the vectors should be shifted to the right
            tgt_actedge_masks_item = tgt_actedge_masks_item.clone()
            tgt_actedge_cur_nodes_item = tgt_actedge_cur_nodes_item.clone()
            tgt_actedge_pre_nodes_item = tgt_actedge_pre_nodes_item.clone()
            tgt_actedge_directions_item = tgt_actedge_directions_item.clone()
            tgt_actnode_mask_shift_item = tgt_actnode_mask_item.clone()    # node mask on the tgt input side
            # NOTE we must clone first -- modification in their original place is not allowed
            # if not using .clone(), there is an error:
            #      RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to
            #      a single memory location. Please clone() the tensor before performing the operation.
            for tgt_actedge_x in (tgt_actedge_masks_item, tgt_actedge_cur_nodes_item,
                                tgt_actedge_pre_nodes_item, tgt_actedge_directions_item,
                                tgt_actnode_mask_shift_item):
                tgt_actedge_x[1:] = tgt_actedge_x[:-1].clone()

            # b) the values referring to node positions should be shifted 1 to the right as well
            tgt_actedge_cur_nodes_item[tgt_actedge_cur_nodes_item >= 0] += 1
            tgt_actedge_pre_nodes_item[tgt_actedge_pre_nodes_item >= 0] += 1

            # c) for the <s> position at the beginning
            tgt_actedge_masks_item[0] = 0
            tgt_actedge_cur_nodes_item[0] = self.pad_tgt_actedge_cur_nodes
            tgt_actedge_pre_nodes_item[0] = self.pad_tgt_actedge_pre_nodes
            tgt_actedge_directions_item[0] = self.pad_tgt_actedge_directions
            tgt_actnode_mask_shift_item[0] = 0

        # ========================================================================================

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        else:
            # tgt action state information includes the CLOSE action at last step, which is mapped to the eos token
            tgt_vocab_mask_item = tgt_vocab_mask_item[:-1] if tgt_vocab_mask_item is not None else None
            tgt_actnode_mask_item = tgt_actnode_mask_item[:-1] if tgt_actnode_mask_item is not None else None
            tgt_src_cursor_item = tgt_src_cursor_item[:-1] if tgt_src_cursor_item is not None else None
            # tgt graph structure
            tgt_actedge_masks_item = tgt_actedge_masks_item[:-1] \
                if tgt_actedge_masks_item is not None else None
            tgt_actedge_cur_nodes_item = tgt_actedge_cur_nodes_item[:-1] \
                if tgt_actedge_cur_nodes_item is not None else None
            tgt_actedge_pre_nodes_item = tgt_actedge_pre_nodes_item[:-1] \
                if tgt_actedge_pre_nodes_item is not None else None
            tgt_actedge_directions_item = tgt_actedge_directions_item[:-1] \
                if tgt_actedge_directions_item is not None else None
            tgt_actnode_mask_shift_item = tgt_actnode_mask_shift_item[:-1] \
                if tgt_actnode_mask_shift_item is not None else None

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'src_tokens': src_tokens_item,
            'source': src_item,
            'source_fix_emb': src_fix_emb_item,
            'src_wordpieces': src_wordpieces_item,
            'src_wp2w': src_wp2w_item,
            'target': tgt_item,
            'force_actions': force_actions_item,
            'tgt_in': tgt_in_item,
            'tgt_pos': tgt_pos_item,
            # AMR actions states (tied with the tgt output side)
            'tgt_vocab_masks': tgt_vocab_mask_item,
            'tgt_actnode_masks': tgt_actnode_mask_item,
            'tgt_src_cursors': tgt_src_cursor_item,
            # graph structure (tied with the tgt input side)
            'tgt_actedge_masks': tgt_actedge_masks_item,
            'tgt_actedge_cur_nodes': tgt_actedge_cur_nodes_item,
            'tgt_actedge_pre_nodes': tgt_actedge_pre_nodes_item,
            'tgt_actedge_directions': tgt_actedge_directions_item,
            'tgt_actnode_masks_shift': tgt_actnode_mask_shift_item
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
            samples, pad_idx=self.tgt_dict.pad(), eos_idx=self.tgt_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            collate_tgt_states=self.collate_tgt_states,
            collate_tgt_states_graph=self.collate_tgt_states_graph,
            pad_tgt_actedge_cur_nodes=self.pad_tgt_actedge_cur_nodes,
            pad_tgt_actedge_pre_nodes=self.pad_tgt_actedge_pre_nodes,
            pad_tgt_actedge_directions=self.pad_tgt_actedge_directions
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
        if self.src_fix_emb is not None:
            self.src_fix_emb.prefetch(indices)
        # self.memory.prefetch(indices)
        # self.mem_pos.prefetch(indices)
