"""
Test data iterator with e.g.

. set_environment.sh

arguments="
    DATA/amr/features/o3+Word100_RoBERTa-base/
    --gen-subset train
    --batch-size 128
"

# do not use @profile
#python tests/fairseq_data_iterator.py $arguments

# Use @profile
kernprof -l tests/fairseq_data_iterator.py $arguments
python -m line_profiler fairseq_data_iterator.py.lprof
"""

from fairseq import tasks, utils
from fairseq_ext import options
from fairseq_ext.utils_import import import_user_module
from fairseq.data import data_utils, FairseqDataset
from tqdm import tqdm


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
        indices, dataset.num_tokens, max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    return batch_sampler


def main(args):

    # Load dataset
    import_user_module(args)
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)

    # Get iterator over batches
    batch_index_iterator = get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=None,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        large_sent_first=False
    )

    # collate batch of sentences into single tensor for all data
    for batch_ids in tqdm(batch_index_iterator):
        samples = [dataset[i] for i in batch_ids]
        dataset.collater(samples)


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
