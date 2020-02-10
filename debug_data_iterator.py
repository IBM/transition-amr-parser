#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

import os
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.tokenizer import tokenize_line, tab_tokenize
from fairseq.data import data_utils, FairseqDataset

from transition_amr_parser.data_oracle import writer


def get_batch_iterator(
    dataset, max_tokens=None, max_sentences=None, max_positions=None,
    ignore_invalid_inputs=False, required_batch_size_multiple=1,
    seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    large_sent_first=False
):
    """
    Get an iterator that yields batches of data from the given dataset.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset to batch
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        max_positions (optional): max sentence length supported by the
            model (default: None).
        ignore_invalid_inputs (bool, optional): don't raise Exception for
            sentences that are too long (default: False).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 0).

    Returns:
        ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
            given dataset split
    """
    assert isinstance(dataset, FairseqDataset)

    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()
        # invert order to start by bigger ones
        if large_sent_first:
            indices = indices[::-1]

    # filter examples that are too large
    if max_positions is not None:
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, 
            raise_exception=(not ignore_invalid_inputs),
        )

    # create mini-batches with given size constraints
    batch_sampler = data_utils.batch_by_size(
        indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    return batch_sampler


def main(args):

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    # Note: states are not needed since they will be provided by the state
    # machine
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # max positions
    max_positions = None

#     # Load dataset (possibly sharded)
#     import ipdb; ipdb.set_trace(context=30)
#     itr = task.get_batch_iterator(
#         dataset=task.dataset(args.gen_subset),
#         max_tokens=args.max_tokens,
#         max_sentences=args.max_sentences,
#         max_positions=max_positions,
#         ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
#         required_batch_size_multiple=args.required_batch_size_multiple,
#         num_shards=args.num_shards,
#         shard_id=args.shard_id,
#         num_workers=args.num_workers,
#         large_sent_first=False
#     ).next_epoch_itr(shuffle=False)

    dataset=task.dataset(args.gen_subset)

    # Load dataset (possibly sharded)
    batch_index_iterator = get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        large_sent_first=False
    )

    # Normally Wrapped with EpochBatchIterator and CountingIterator
    data_iterator = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_index_iterator,
        num_workers=1,
    )

    import ipdb; ipdb.set_trace(context=30)

    num_sentences = 0
    has_target = True

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample


def cli_main():
    parser = options.get_generation_parser(default_task='parsing')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
