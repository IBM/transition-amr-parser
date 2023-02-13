import os
import itertools
from multiprocessing import Pool
import time

import torch
import numpy as np
from tqdm import tqdm

from transition_amr_parser.action_pointer.o8_state_machine import AMRStateMachine
from .action_info import get_actions_states
from ..tokenizer import tokenize_line_tab
from ..binarize import make_builder    # TODO move this to data folder
from ..data.data_utils import load_indexed_dataset
from ..utils import time_since


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
                allowed_cano_actions = actions_states['allowed_cano_actions']
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
                actions_nodemask = torch.tensor(actions_states['actions_nodemask'], dtype=torch.uint8)
                token_cursors = torch.tensor(actions_states['token_cursors'])

                # graph structure information
                actions_edge_mask = torch.tensor(actions_states['actions_edge_mask'], dtype=torch.uint8)
                actions_edge_cur_node = torch.tensor(actions_states['actions_edge_cur_node'])
                actions_edge_pre_node = torch.tensor(actions_states['actions_edge_pre_node'])
                actions_edge_direction = torch.tensor(actions_states['actions_edge_direction'])

                consumer(vocab_mask, actions_nodemask, token_cursors,
                         actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction)

                count += 1
                if count % 1000 == 0:
                    print(f'\r processed {count} en-actions pairs (time: {time_since(start)})', end='')

                line = f.readline()
                actions = g.readline()
            print('')
        # return any useful statistics
        return {}

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


# same functionality as above, but without design for multiprocessing
def get_actions_states_file(en_file, actions_file, actions_dict, consumer=None, canonical_act_ids=None, tokenize=None):
    """Build actions information such as masks for restricted action space, masks for node generating actions,
    token cursor positions, etc.

    Save them into binary files.

    Args:
        en_file (str): English sentence file path.
        actions_file (str): actions file path.
        actions_dict (fairseq.data.Dict): action word dictionary.
        canonical_act_ids (dict): a dictionary mapping from canonical action names to ids in the action word vocabulary.
        tokenize (callable): a function to tokenize a line of string to a list of tokens.

    Note:
        - the states for each action sequence include the last CLOSE action, which will be removed if eos token is not
          appended at the end of each sequence.
    """
    tgt_vocab_masks = []    # a list of 2-D tensors of size (seq_len, tgt_vocab_size)
    tgt_actnode_masks = []    # a list of 1-D tensors of size (seq_len,)
    tgt_src_cursors = []    # a list of 1-D tensors of size (seq_len,)
    # graph structure
    tgt_actedge_masks = []        # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_cur_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_pre_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_directions = []    # a list of 1-D tensors of size (seq_len,)

    if canonical_act_ids is None:
        assert actions_dict is not None
        canonical_act_ids = AMRStateMachine.canonical_action_to_dict(actions_dict)

    if tokenize is None:
        def tokenize(line):
            return line.strip().split('\t')

    with open(en_file, 'r') as f, open(actions_file, 'r') as g:
        for tokens, actions in tqdm(zip(f, g)):
            if tokens.strip():
                tokens = tokenize(tokens)
                actions = tokenize(actions)
                assert tokens[-1] == '<ROOT>'
                actions_states = get_actions_states(tokens=tokens, actions=actions)
                # construct actions states tensors
                allowed_cano_actions = actions_states['allowed_cano_actions']
                vocab_mask = torch.zeros(len(allowed_cano_actions), len(actions_dict), dtype=torch.uint8)
                for i, act_allowed in enumerate(allowed_cano_actions):
                    # vocab_ids_allowed = list(set().union(*[set(canonical_act_ids[act]) for act in act_allowed]))
                    # this is a bit faster than above
                    vocab_ids_allowed = list(
                        itertools.chain.from_iterable(
                            [canonical_act_ids[act] for act in act_allowed]
                        )
                    )
                    vocab_mask[i][vocab_ids_allowed] = 1

                tgt_vocab_masks.append(vocab_mask)

                actions_nodemask = torch.tensor(actions_states['actions_nodemask'], dtype=torch.uint8)
                tgt_actnode_masks.append(actions_nodemask)

                token_cursors = torch.tensor(actions_states['token_cursors'])
                tgt_src_cursors.append(token_cursors)

                # graph structure information
                actions_edge_mask = torch.tensor(actions_states['actions_edge_mask'], dtype=torch.uint8)
                actions_edge_cur_node = torch.tensor(actions_states['actions_edge_cur_node'])
                actions_edge_pre_node = torch.tensor(actions_states['actions_edge_pre_node'])
                actions_edge_direction = torch.tensor(actions_states['actions_edge_direction'])

                tgt_actedge_masks.append(actions_edge_mask)
                tgt_actedge_cur_nodes.append(actions_edge_cur_node)
                tgt_actedge_pre_nodes.append(actions_edge_pre_node)
                tgt_actedge_directions.append(actions_edge_direction)

                if consumer is not None:
                    consumer(vocab_mask, actions_nodemask, token_cursors,
                             actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction)

    return tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions


def binarize_actstates_tolist(en_file, actions_file, actions_dict=None, tokenize=tokenize_line_tab,
                              action_state_binarizer=None,
                              en_offset=0, en_end=-1,
                              actions_offset=0, actions_end=-1):
    """Get the action states and save to lists."""
    tgt_vocab_masks = []    # a list of 2-D tensors of size (seq_len, tgt_vocab_size)
    tgt_actnode_masks = []    # a list of 1-D tensors of size (seq_len,)
    tgt_src_cursors = []    # a list of 1-D tensors of size (seq_len,)
    # graph structure
    tgt_actedge_masks = []        # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_cur_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_pre_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_directions = []    # a list of 1-D tensors of size (seq_len,)

    def consumer(vocab_mask, actions_nodemask, token_cursors,
                 actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction):
        tgt_vocab_masks.append(vocab_mask)
        tgt_actnode_masks.append(actions_nodemask)
        tgt_src_cursors.append(token_cursors)
        # graph structure
        tgt_actedge_masks.append(actions_edge_mask)
        tgt_actedge_cur_nodes.append(actions_edge_cur_node)
        tgt_actedge_pre_nodes.append(actions_edge_pre_node)
        tgt_actedge_directions.append(actions_edge_direction)
        return

    if action_state_binarizer is None:
        assert actions_dict is not None
        action_state_binarizer = ActionStatesBinarizer(actions_dict)
    action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                    en_offset=en_offset, en_end=en_end,
                                    actions_offset=actions_offset, actions_end=actions_end)

    # OR (for the same results, but should not be used when the function is called for multiprocessing)
    # _ = get_actions_states_file(en_file, actions_file, actions_dict, consumer=consumer, tokenize=tokenize)

    return tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions


# TODO not working for num_workers > 1: need to figure out how to properly return tensor values from Pool
def binarize_actstates_tolist_workers(en_file, actions_file, actions_dict=None,
                                      action_state_binarizer=None,
                                      tokenize=tokenize_line_tab, num_workers=1):
    """Get the action states and save to binary files, allowing multiprocessing to speed up."""
    print('-' * 100)
    print(f'Generate and process action states information (number of workers: {num_workers}):')
    print(f'[English sentence file: {en_file}]')
    print(f'[AMR actions file: {actions_file}]')
    print('processing ...', end=' ')
    start = time.time()

    en_offsets, actions_offsets = ActionStatesBinarizer.find_offsets_paired(en_file, actions_file, num_workers)
    if action_state_binarizer is None:
        assert actions_dict is not None
        action_state_binarizer = ActionStatesBinarizer(actions_dict)

    pool = None
    # multiprocessing
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        res = []
        for worker_id in range(1, num_workers):
            res.append(pool.apply_async(
                binarize_actstates_tolist,
                (
                    en_file,
                    actions_file,
                    actions_dict,
                    tokenize,
                    action_state_binarizer,
                    en_offsets[worker_id],
                    en_offsets[worker_id + 1],
                    actions_offsets[worker_id],
                    actions_offsets[worker_id + 1]
                ),
            ))
        pool.close()

    # main process
    tgt_vocab_masks = []    # a list of 2-D tensors of size (seq_len, tgt_vocab_size)
    tgt_actnode_masks = []    # a list of 1-D tensors of size (seq_len,)
    tgt_src_cursors = []    # a list of 1-D tensors of size (seq_len,)
    # graph structure
    tgt_actedge_masks = []        # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_cur_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_pre_nodes = []    # a list of 1-D tensors of size (seq_len,)
    tgt_actedge_directions = []    # a list of 1-D tensors of size (seq_len,)

    def consumer(vocab_mask, actions_nodemask, token_cursors,
                 actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction):
        tgt_vocab_masks.append(vocab_mask)
        tgt_actnode_masks.append(actions_nodemask)
        tgt_src_cursors.append(token_cursors)
        # graph structure
        tgt_actedge_masks.append(actions_edge_mask)
        tgt_actedge_cur_nodes.append(actions_edge_cur_node)
        tgt_actedge_pre_nodes.append(actions_edge_pre_node)
        tgt_actedge_directions.append(actions_edge_direction)
        return

    action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                    en_offset=0, en_end=en_offsets[1],
                                    actions_offset=0, actions_end=actions_offsets[1])

    # merge the files from multiple workers to the main process
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            import pdb
            pdb.set_trace()
            # tgt_vocab_masks += res[worker_id - 1].get()[0]
            # tgt_actnode_masks += res[worker_id - 1].get()[1]
            # tgt_src_cursors += res[worker_id - 1].get()[2]
            # tgt_actedge_masks += res[worker_id - 1].get()[3]
            # tgt_actedge_cur_nodes += res[worker_id - 1].get()[4]
            # tgt_actedge_pre_nodes += res[worker_id - 1].get()[5]
            # tgt_actedge_directions += res[worker_id - 1].get()[6]

    print('finished !')
    print(f'Processed data saved to lists.')
    print(f'Total time elapsed: {time_since(start)}')
    print('-' * 100)

    return tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions


def binarize_actstates_tofile(en_file, actions_file, out_file_pref,
                              actions_dict=None,
                              impl='mmap',
                              tokenize=tokenize_line_tab,
                              action_state_binarizer=None,
                              en_offset=0, en_end=-1,
                              actions_offset=0, actions_end=-1):
    """Get the action states and save to binary files."""
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

    def consumer(vocab_mask, actions_nodemask, token_cursors,
                 actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction):
        ds_tgt_vocab_masks.add_item(vocab_mask.view(-1))    # NOTE here we flatten the 2-D tensor
        ds_tgt_actnode_masks.add_item(actions_nodemask)
        ds_tgt_src_cursors.add_item(token_cursors)
        # graph structure
        ds_tgt_actedge_masks.add_item(actions_edge_mask)
        ds_tgt_actedge_cur_nodes.add_item(actions_edge_cur_node)
        ds_tgt_actedge_pre_nodes.add_item(actions_edge_pre_node)
        ds_tgt_actedge_directions.add_item(actions_edge_direction)
        return

    if action_state_binarizer is None:
        assert actions_dict is not None
        action_state_binarizer = ActionStatesBinarizer(actions_dict)
    action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                    en_offset=en_offset, en_end=en_end,
                                    actions_offset=actions_offset, actions_end=actions_end)

    # OR (for the same results, but should not be used when the function is called for multiprocessing)
    # _ = get_actions_states_file(en_file, actions_file, actions_dict, consumer=consumer, tokenize=tokenize)

    ds_tgt_vocab_masks.finalize(index_file_tgt_vocab_masks)
    ds_tgt_actnode_masks.finalize(index_file_tgt_actnode_masks)
    ds_tgt_src_cursors.finalize(index_file_tgt_src_cursors)
    # graph structure
    ds_tgt_actedge_masks.finalize(index_file_tgt_actedge_masks)
    ds_tgt_actedge_cur_nodes.finalize(index_file_tgt_actedge_cur_nodes)
    ds_tgt_actedge_pre_nodes.finalize(index_file_tgt_actedge_pre_nodes)
    ds_tgt_actedge_directions.finalize(index_file_tgt_actedge_directions)

    return


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
                callback=None
            )
        pool.close()

    # main process
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

    def consumer(vocab_mask, actions_nodemask, token_cursors,
                 actions_edge_mask, actions_edge_cur_node, actions_edge_pre_node, actions_edge_direction):
        ds_tgt_vocab_masks.add_item(vocab_mask.view(-1))    # NOTE here we flatten the 2-D tensor
        ds_tgt_actnode_masks.add_item(actions_nodemask)
        ds_tgt_src_cursors.add_item(token_cursors)
        # graph structure
        ds_tgt_actedge_masks.add_item(actions_edge_mask)
        ds_tgt_actedge_cur_nodes.add_item(actions_edge_cur_node)
        ds_tgt_actedge_pre_nodes.add_item(actions_edge_pre_node)
        ds_tgt_actedge_directions.add_item(actions_edge_direction)
        return

    action_state_binarizer.binarize(en_file, actions_file, consumer, tokenize=tokenize,
                                    en_offset=0, en_end=en_offsets[1],
                                    actions_offset=0, actions_end=actions_offsets[1])

    # merge the files from multiple workers to the main process
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            out_file_pref_temp = out_file_pref + f'{worker_id}'
            ds_tgt_vocab_masks.merge_file_(out_file_pref_temp + '.vocab_masks')
            ds_tgt_actnode_masks.merge_file_(out_file_pref_temp + '.actnode_masks')
            ds_tgt_src_cursors.merge_file_(out_file_pref_temp + '.src_cursors')
            # graph structure
            ds_tgt_actedge_masks.merge_file_(out_file_pref_temp + '.actedge_masks')
            ds_tgt_actedge_cur_nodes.merge_file_(out_file_pref_temp + '.actedge_cur_nodes')
            ds_tgt_actedge_pre_nodes.merge_file_(out_file_pref_temp + '.actedge_pre_nodes')
            ds_tgt_actedge_directions.merge_file_(out_file_pref_temp + '.actedge_directions')

            os.remove(out_file_pref_temp + '.vocab_masks' + '.bin')
            os.remove(out_file_pref_temp + '.vocab_masks' + '.idx')
            os.remove(out_file_pref_temp + '.actnode_masks' + '.bin')
            os.remove(out_file_pref_temp + '.actnode_masks' + '.idx')
            os.remove(out_file_pref_temp + '.src_cursors' + '.bin')
            os.remove(out_file_pref_temp + '.src_cursors' + '.idx')
            # graph structure
            os.remove(out_file_pref_temp + '.actedge_masks' + '.bin')
            os.remove(out_file_pref_temp + '.actedge_masks' + '.idx')
            os.remove(out_file_pref_temp + '.actedge_cur_nodes' + '.bin')
            os.remove(out_file_pref_temp + '.actedge_cur_nodes' + '.idx')
            os.remove(out_file_pref_temp + '.actedge_pre_nodes' + '.bin')
            os.remove(out_file_pref_temp + '.actedge_pre_nodes' + '.idx')
            os.remove(out_file_pref_temp + '.actedge_directions' + '.bin')
            os.remove(out_file_pref_temp + '.actedge_directions' + '.idx')

    # finalize to save the dtype and size and index info
    ds_tgt_vocab_masks.finalize(index_file_tgt_vocab_masks)
    ds_tgt_actnode_masks.finalize(index_file_tgt_actnode_masks)
    ds_tgt_src_cursors.finalize(index_file_tgt_src_cursors)
    # graph structure
    ds_tgt_actedge_masks.finalize(index_file_tgt_actedge_masks)
    ds_tgt_actedge_cur_nodes.finalize(index_file_tgt_actedge_cur_nodes)
    ds_tgt_actedge_pre_nodes.finalize(index_file_tgt_actedge_pre_nodes)
    ds_tgt_actedge_directions.finalize(index_file_tgt_actedge_directions)

    print('finished !')
    print(f'Processed data saved to path with prefix: {out_file_pref}')
    print(f'Total time elapsed: {time_since(start)}')
    print('-' * 100)

    return


def load_actstates_fromfile(file_pref, actions_dict, impl='mmap'):
    """Load the action states from binary files"""
    file_pref_tgt_vocab_masks = file_pref + '.vocab_masks'
    file_pref_tgt_actnode_masks = file_pref + '.actnode_masks'
    file_pref_tgt_src_cursors = file_pref + '.src_cursors'
    # graph structure
    file_pref_tgt_actedge_masks = file_pref + '.actedge_masks'
    file_pref_tgt_actedge_cur_nodes = file_pref + '.actedge_cur_nodes'
    file_pref_tgt_actedge_pre_nodes = file_pref + '.actedge_pre_nodes'
    file_pref_tgt_actedge_directions = file_pref + '.actedge_directions'

    tgt_vocab_masks = load_indexed_dataset(file_pref_tgt_vocab_masks, actions_dict, impl)
    tgt_actnode_masks = load_indexed_dataset(file_pref_tgt_actnode_masks, None, impl)
    tgt_src_cursors = load_indexed_dataset(file_pref_tgt_src_cursors, None, impl)
    # graph structure
    tgt_actedge_masks = load_indexed_dataset(file_pref_tgt_actedge_masks, None, impl)
    tgt_actedge_cur_nodes = load_indexed_dataset(file_pref_tgt_actedge_cur_nodes, None, impl)
    tgt_actedge_pre_nodes = load_indexed_dataset(file_pref_tgt_actedge_pre_nodes, None, impl)
    tgt_actedge_directions = load_indexed_dataset(file_pref_tgt_actedge_directions, None, impl)

    return tgt_vocab_masks, tgt_actnode_masks, tgt_src_cursors, \
        tgt_actedge_masks, tgt_actedge_cur_nodes, tgt_actedge_pre_nodes, tgt_actedge_directions
