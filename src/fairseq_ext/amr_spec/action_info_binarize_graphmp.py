import os
import itertools
from multiprocessing import Pool
import time
from collections import Counter

import torch
import numpy as np

from transition_amr_parser.action_pointer.o8_state_machine import AMRStateMachine
from .action_info_graphmp import get_actions_states
from ..tokenizer import tokenize_line_tab
from ..binarize import make_builder    # TODO move this to data folder
from ..data.data_utils import load_indexed_dataset
from ..utils import time_since


# names for all the action states; tensor names and file names MUST be paired in order
# tensor names should be the same as those returned by `get_actions_states`, except that
# 'allowed_cano_actions' -> 'vocab_mask'
actions_states_tensor_names = [
    # training data
    'actions_nopos_in',
    'actions_nopos_out',
    'actions_pos',
    # general states
    'vocab_mask',
    'token_cursors',
    'actions_nodemask',
    # graph structure
    'actions_edge_mask',
    'actions_edge_1stnode_mask',
    'actions_edge_index',
    'actions_edge_cur_node_index',
    'actions_edge_cur_1stnode_index',
    'actions_edge_pre_node_index',
    'actions_edge_direction',
    # graph structure: edge to include all previous nodes
    'actions_edge_allpre_index',
    'actions_edge_allpre_pre_node_index',
    'actions_edge_allpre_direction'
    ]
actions_states_file_names = [
    # training data
    'nopos_in',
    'nopos_out',
    'pos',
    # general states
    'vocab_masks',
    'src_cursors',
    'actnode_masks',
    # graph structure
    'actedge_masks',
    'actedge_1stnode_masks',
    'actedge_indexes',
    'actedge_cur_node_indexes',
    'actedge_cur_1stnode_indexes',
    'actedge_pre_node_indexes',
    'actedge_directions',
    # graph structure: edge to include all previous nodes
    'actedge_allpre_indexes',
    'actedge_allpre_pre_node_indexes',
    'actedge_allpre_directions'
]
assert len(actions_states_tensor_names) == len(actions_states_file_names)


# reference: fairseq binarizer.py
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


# a class implementation to allow for multiprocessing
class ActionStatesBinarizer:
    def __init__(self, actions_dict):
        self.actions_dict = actions_dict
        self.canonical_actions = AMRStateMachine.canonical_actions
        self.canonical_act_ids = AMRStateMachine.canonical_action_to_dict(actions_dict)

    def binarize(self, en_file, actions_file, consumer, tokenize=tokenize_line_tab,
                 en_offset=0, en_end=-1,
                 actions_offset=0, actions_end=-1):
        # NOTE here we scan two files, and their sizes are different, thus we need to have different offsets for them
        # (and make sure the file positions match) when we process part of files in multiprocessing
        if tokenize is None:
            tokenize = tokenize_line_tab

        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == self.actions_dict.unk_index and word != self.actions_dict.unk_word:
                replaced.update([word])

        with open(en_file, 'r', encoding='utf-8') as f, open(actions_file, 'r', encoding='utf-8') as g:
            f.seek(en_offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            g.seek(actions_offset)
            actions = safe_readline(g)
            count = 0
            start = time.time()
            while line:
                if en_end > 0 and f.tell() > en_end:
                    assert actions_end > 0 and g.tell() > actions_end
                    break

                actions_states = get_actions_states(tokens=tokenize(line), actions=tokenize(actions))

                # construct actions states tensors
                actions_states_tensors = {}    # to store the action states converted to tensors

                allowed_cano_actions = actions_states['allowed_cano_actions']
                del actions_states['allowed_cano_actions']
                vocab_mask = torch.zeros(len(allowed_cano_actions), len(self.actions_dict), dtype=torch.uint8)
                for i, act_allowed in enumerate(allowed_cano_actions):
                    # vocab_ids_allowed = list(set().union(*[set(canonical_act_ids[act]) for act in act_allowed]))
                    # this is a bit faster than above
                    vocab_ids_allowed = list(
                        itertools.chain.from_iterable(
                            [self.canonical_act_ids[act] for act in act_allowed]
                        )
                    )
                    vocab_mask[i][vocab_ids_allowed] = 1

                # convert state vectors to tensors
                actions_states_tensors['vocab_mask'] = vocab_mask
                for k, v in actions_states.items():
                    if 'mask' in k:
                        actions_states_tensors[k] = torch.tensor(v, dtype=torch.uint8)
                    elif 'nopos_in' in k:
                        # input sequence
                        actions_states_tensors[k] = self.actions_dict.encode_line(
                            line=[act if act != 'CLOSE' else self.actions_dict.eos_word for act in v],
                            line_tokenizer=lambda x: x,    # already tokenized
                            add_if_not_exist=False,
                            consumer=None,
                            append_eos=False,
                            reverse_order=False
                        )
                    elif 'nopos_out' in k:
                        # output sequence
                        actions_states_tensors[k] = self.actions_dict.encode_line(
                            line=[act if act != 'CLOSE' else self.actions_dict.eos_word for act in v],
                            line_tokenizer=lambda x: x,    # already tokenized
                            add_if_not_exist=False,
                            consumer=replaced_consumer,
                            append_eos=False,
                            reverse_order=False
                        )
                        nseq += 1
                        ntok += len(actions_states_tensors[k])
                    else:
                        actions_states_tensors[k] = torch.tensor(v)    # int64

                consumer(actions_states_tensors)

                count += 1
                if count % 1000 == 0:
                    print(f'\r processed {count} en-actions pairs (time: {time_since(start)})', end='')

                line = f.readline()
                actions = g.readline()
            print('')
        # return any useful statistics
        return {'nseq': nseq,
                'ntok': ntok,
                # do not count 'CLOSE' as <unk>, as it is equivalent to <eos> or </s>
                'nunk': sum(replaced.values()) - replaced['CLOSE'],
                'replaced': replaced}

    @staticmethod
    def find_offsets_paired(en_filename, actions_filename, num_chunks):
        """Find the file offsets of two files which are paired line by line; the offsets must match."""
        with open(en_filename, 'r', encoding='utf-8') as f, open(actions_filename, 'r', encoding='utf-8') as g:
            nlines_en = sum(1 for line in f)
            nlines_actions = sum(1 for line in g)
            assert nlines_en == nlines_actions
            chunk_nlines = nlines_en // num_chunks
            en_offsets = [0 for _ in range(num_chunks + 1)]
            actions_offsets = [0 for _ in range(num_chunks + 1)]
            f.seek(0)
            g.seek(0)
            for i in range(1, num_chunks):
                for j in range(chunk_nlines):
                    safe_readline(f)
                    safe_readline(g)
                en_offsets[i] = f.tell()
                actions_offsets[i] = g.tell()
            return en_offsets, actions_offsets

    @staticmethod
    def find_offsets(filename, num_chunks):
        """Only works for a single file; chunks are divided by file size (#bytes)."""
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets


def binarize_actstates_tofile(en_file, actions_file, out_file_pref,
                              actions_dict=None,
                              impl='mmap',
                              tokenize=tokenize_line_tab,
                              action_state_binarizer=None,
                              en_offset=0, en_end=-1,
                              actions_offset=0, actions_end=-1):
    """Get the action states and save to binary files."""

    """
    out_file_tgt_vocab_masks = out_file_pref + '.vocab_masks' + '.bin'
    index_file_tgt_vocab_masks = out_file_pref + '.vocab_masks' + '.idx'

    out_file_tgt_actnode_masks = out_file_pref + '.actnode_masks' + '.bin'
    index_file_tgt_actnode_masks = out_file_pref + '.actnode_masks' + '.idx'

    out_file_tgt_src_cursors = out_file_pref + '.src_cursors' + '.bin'
    index_file_tgt_src_cursors = out_file_pref + '.src_cursors' + '.idx'

    # graph structure
    out_file_tgt_actedge_masks = out_file_pref + '.actedge_masks' + '.bin'
    index_file_tgt_actedge_masks = out_file_pref + '.actedge_masks' + '.idx'

    out_file_tgt_actedge_cur_nodes = out_file_pref + '.actedge_cur_nodes' + '.bin'
    index_file_tgt_actedge_cur_nodes = out_file_pref + '.actedge_cur_nodes' + '.idx'

    out_file_tgt_actedge_pre_nodes = out_file_pref + '.actedge_pre_nodes' + '.bin'
    index_file_tgt_actedge_pre_nodes = out_file_pref + '.actedge_pre_nodes' + '.idx'

    out_file_tgt_actedge_directions = out_file_pref + '.actedge_directions' + '.bin'
    index_file_tgt_actedge_directions = out_file_pref + '.actedge_directions' + '.idx'

    ds_tgt_vocab_masks = make_builder(out_file_tgt_vocab_masks, impl=impl, dtype=np.uint8)
    ds_tgt_actnode_masks = make_builder(out_file_tgt_actnode_masks, impl=impl, dtype=np.uint8)
    ds_tgt_src_cursors = make_builder(out_file_tgt_src_cursors, impl=impl, dtype=np.int64)
    # graph structure
    ds_tgt_actedge_masks = make_builder(out_file_tgt_actedge_masks, impl=impl, dtype=np.uint8)
    ds_tgt_actedge_cur_nodes = make_builder(out_file_tgt_actedge_cur_nodes, impl=impl, dtype=np.int64)
    ds_tgt_actedge_pre_nodes = make_builder(out_file_tgt_actedge_pre_nodes, impl=impl, dtype=np.int64)
    ds_tgt_actedge_directions = make_builder(out_file_tgt_actedge_directions, impl=impl, dtype=np.int64)
    """

    out_file_tgt_list = []
    index_file_tgt_list = []
    ds_tgt_list = []

    for name in actions_states_file_names:
        out_file_tgt_list.append(out_file_pref + '.' + name + '.bin')
        index_file_tgt_list.append(out_file_pref + '.' + name + '.idx')
        if 'mask' in name:
            ds_tgt_list.append(make_builder(out_file_pref + '.' + name + '.bin', impl=impl, dtype=np.uint8))
        else:
            ds_tgt_list.append(make_builder(out_file_pref + '.' + name + '.bin', impl=impl, dtype=np.int64))

    def consumer(actions_states_tensors):
        for i, name in enumerate(actions_states_tensor_names):
            if name == 'vocab_mask':
                # NOTE here we flatten the 2-D tensor
                ds_tgt_list[i].add_item(actions_states_tensors['vocab_mask'].view(-1))
            else:
                ds_tgt_list[i].add_item(actions_states_tensors[name])
        return

    if action_state_binarizer is None:
        assert actions_dict is not None
        action_state_binarizer = ActionStatesBinarizer(actions_dict)
    res = action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                          en_offset=en_offset, en_end=en_end,
                                          actions_offset=actions_offset, actions_end=actions_end)

    for ds, index_file in zip(ds_tgt_list, index_file_tgt_list):
        ds.finalize(index_file)

    return res


def binarize_actstates_tofile_workers(en_file, actions_file, out_file_pref,
                                      actions_dict=None,
                                      action_state_binarizer=None,
                                      impl='mmap',
                                      tokenize=tokenize_line_tab,
                                      num_workers=1):
    """Get the action states and save to binary files, allowing multiprocessing to speed up."""
    print('-' * 100)
    print(f'Generate and process action states information (number of workers: {num_workers}):')
    print(f'[English sentence file: {en_file}]')
    print(f'[AMR actions file: {actions_file}]')
    print('processing ...', end=' ')
    start = time.time()

    n_seq_tok = [0, 0]
    replaced = Counter()

    def merge_result(worker_result):
        replaced.update(worker_result["replaced"])
        n_seq_tok[0] += worker_result["nseq"]
        n_seq_tok[1] += worker_result["ntok"]

    en_offsets, actions_offsets = ActionStatesBinarizer.find_offsets_paired(en_file, actions_file, num_workers)
    if action_state_binarizer is None:
        assert actions_dict is not None
        action_state_binarizer = ActionStatesBinarizer(actions_dict)

    pool = None
    # multiprocessing
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            out_file_pref_temp = out_file_pref + f'{worker_id}'
            pool.apply_async(
                binarize_actstates_tofile,
                (
                    en_file,
                    actions_file,
                    out_file_pref_temp,
                    actions_dict,
                    impl,
                    tokenize,
                    action_state_binarizer,
                    en_offsets[worker_id],
                    en_offsets[worker_id + 1],
                    actions_offsets[worker_id],
                    actions_offsets[worker_id + 1]
                ),
                callback=merge_result
            )
        pool.close()

    # main process


    # out_file_tgt_vocab_masks = out_file_pref + '.vocab_masks' + '.bin'
    # index_file_tgt_vocab_masks = out_file_pref + '.vocab_masks' + '.idx'

    # out_file_tgt_actnode_masks = out_file_pref + '.actnode_masks' + '.bin'
    # index_file_tgt_actnode_masks = out_file_pref + '.actnode_masks' + '.idx'

    # out_file_tgt_src_cursors = out_file_pref + '.src_cursors' + '.bin'
    # index_file_tgt_src_cursors = out_file_pref + '.src_cursors' + '.idx'

    # # graph structure
    # out_file_tgt_actedge_masks = out_file_pref + '.actedge_masks' + '.bin'
    # index_file_tgt_actedge_masks = out_file_pref + '.actedge_masks' + '.idx'

    # out_file_tgt_actedge_cur_nodes = out_file_pref + '.actedge_cur_nodes' + '.bin'
    # index_file_tgt_actedge_cur_nodes = out_file_pref + '.actedge_cur_nodes' + '.idx'

    # out_file_tgt_actedge_pre_nodes = out_file_pref + '.actedge_pre_nodes' + '.bin'
    # index_file_tgt_actedge_pre_nodes = out_file_pref + '.actedge_pre_nodes' + '.idx'

    # out_file_tgt_actedge_directions = out_file_pref + '.actedge_directions' + '.bin'
    # index_file_tgt_actedge_directions = out_file_pref + '.actedge_directions' + '.idx'

    # ds_tgt_vocab_masks = make_builder(out_file_tgt_vocab_masks, impl=impl, dtype=np.uint8)
    # ds_tgt_actnode_masks = make_builder(out_file_tgt_actnode_masks, impl=impl, dtype=np.uint8)
    # ds_tgt_src_cursors = make_builder(out_file_tgt_src_cursors, impl=impl, dtype=np.int64)
    # # graph structure
    # ds_tgt_actedge_masks = make_builder(out_file_tgt_actedge_masks, impl=impl, dtype=np.uint8)
    # ds_tgt_actedge_cur_nodes = make_builder(out_file_tgt_actedge_cur_nodes, impl=impl, dtype=np.int64)
    # ds_tgt_actedge_pre_nodes = make_builder(out_file_tgt_actedge_pre_nodes, impl=impl, dtype=np.int64)
    # ds_tgt_actedge_directions = make_builder(out_file_tgt_actedge_directions, impl=impl, dtype=np.int64)

    # def consumer(vocab_mask, actions_nodemask, token_cursors,
    #              actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction):
    #     ds_tgt_vocab_masks.add_item(vocab_mask.view(-1))    # NOTE here we flatten the 2-D tensor
    #     ds_tgt_actnode_masks.add_item(actions_nodemask)
    #     ds_tgt_src_cursors.add_item(token_cursors)
    #     # graph structure
    #     ds_tgt_actedge_masks.add_item(actions_edge_mask)
    #     ds_tgt_actedge_cur_nodes.add_item(actions_edge_cur_node)
    #     ds_tgt_actedge_pre_nodes.add_item(actions_edge_pre_node)
    #     ds_tgt_actedge_directions.add_item(actions_edge_direction)
    #     return

    out_file_tgt_list = []
    index_file_tgt_list = []
    ds_tgt_list = []

    for name in actions_states_file_names:
        out_file_tgt_list.append(out_file_pref + '.' + name + '.bin')
        index_file_tgt_list.append(out_file_pref + '.' + name + '.idx')
        if 'mask' in name:
            ds_tgt_list.append(make_builder(out_file_pref + '.' + name + '.bin', impl=impl, dtype=np.uint8))
        else:
            ds_tgt_list.append(make_builder(out_file_pref + '.' + name + '.bin', impl=impl, dtype=np.int64))

    def consumer(actions_states_tensors):
        for i, name in enumerate(actions_states_tensor_names):
            if name == 'vocab_mask':
                # NOTE here we flatten the 2-D tensor
                ds_tgt_list[i].add_item(actions_states_tensors['vocab_mask'].view(-1))
            else:
                ds_tgt_list[i].add_item(actions_states_tensors[name])
        return

    merge_result(
        action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                        en_offset=0, en_end=en_offsets[1],
                                        actions_offset=0, actions_end=actions_offsets[1])
        )

    # merge the files from multiple workers to the main process
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            out_file_pref_temp = out_file_pref + f'{worker_id}'

            for ds, name in zip(ds_tgt_list, actions_states_file_names):
                ds.merge_file_(out_file_pref_temp + '.' + name)
                os.remove(out_file_pref_temp + '.' + name + '.bin')
                os.remove(out_file_pref_temp + '.' + name + '.idx')


            # ds_tgt_vocab_masks.merge_file_(out_file_pref_temp + '.vocab_masks')
            # ds_tgt_actnode_masks.merge_file_(out_file_pref_temp + '.actnode_masks')
            # ds_tgt_src_cursors.merge_file_(out_file_pref_temp + '.src_cursors')
            # # graph structure
            # ds_tgt_actedge_masks.merge_file_(out_file_pref_temp + '.actedge_masks')
            # ds_tgt_actedge_cur_nodes.merge_file_(out_file_pref_temp + '.actedge_cur_nodes')
            # ds_tgt_actedge_pre_nodes.merge_file_(out_file_pref_temp + '.actedge_pre_nodes')
            # ds_tgt_actedge_directions.merge_file_(out_file_pref_temp + '.actedge_directions')

            # os.remove(out_file_pref_temp + '.vocab_masks' + '.bin')
            # os.remove(out_file_pref_temp + '.vocab_masks' + '.idx')
            # os.remove(out_file_pref_temp + '.actnode_masks' + '.bin')
            # os.remove(out_file_pref_temp + '.actnode_masks' + '.idx')
            # os.remove(out_file_pref_temp + '.src_cursors' + '.bin')
            # os.remove(out_file_pref_temp + '.src_cursors' + '.idx')
            # # graph structure
            # os.remove(out_file_pref_temp + '.actedge_masks' + '.bin')
            # os.remove(out_file_pref_temp + '.actedge_masks' + '.idx')
            # os.remove(out_file_pref_temp + '.actedge_cur_nodes' + '.bin')
            # os.remove(out_file_pref_temp + '.actedge_cur_nodes' + '.idx')
            # os.remove(out_file_pref_temp + '.actedge_pre_nodes' + '.bin')
            # os.remove(out_file_pref_temp + '.actedge_pre_nodes' + '.idx')
            # os.remove(out_file_pref_temp + '.actedge_directions' + '.bin')
            # os.remove(out_file_pref_temp + '.actedge_directions' + '.idx')

    # finalize to save the dtype and size and index info
    for ds, index_file in zip(ds_tgt_list, index_file_tgt_list):
        ds.finalize(index_file)

    print('finished !')
    print(f'Processed data saved to path with prefix: {out_file_pref}')
    print(f'Total time elapsed: {time_since(start)}')
    print('-' * 100)

    return {'nseq': n_seq_tok[0],
            'ntok': n_seq_tok[1],
            # do not count 'CLOSE' as <unk>, as it is equivalent to <eos> or </s>
            'nunk': sum(replaced.values()) - replaced['CLOSE'],
            'replaced': replaced}


def load_actstates_fromfile(file_pref, actions_dict, impl='mmap'):
    """Load the action states from binary files"""
    tgt_actstates = {}
    for name in actions_states_file_names:
        tgt_name = 'tgt_' + name
        if name == 'vocab_masks':
            tgt_actstates[tgt_name] = load_indexed_dataset(file_pref + '.' + name, actions_dict, impl)
        else:
            tgt_actstates[tgt_name] = load_indexed_dataset(file_pref + '.' + name, None, impl)

    return tgt_actstates
