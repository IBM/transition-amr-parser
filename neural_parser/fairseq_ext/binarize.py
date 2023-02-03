import numpy as np
import torch
from fairseq.data.indexed_dataset import __best_fitting_dtype, MMapIndexedDatasetBuilder, IndexedDatasetBuilder
from fairseq.tokenizer import tokenize_line


# TODO move this file into data folder
def make_builder(out_file, impl, vocab_size=None, dtype=None):
    if impl == 'mmap':
        if dtype is None:
            dtype = __best_fitting_dtype(vocab_size)
        return MMapIndexedDatasetBuilder(out_file, dtype=dtype)
    else:
        return IndexedDatasetBuilder(out_file)


def binarize_file(input_file, out_file_pref, impl, dtype=np.int64, tokenize=tokenize_line):
    out_file = out_file_pref + '.bin'
    index_file = out_file_pref + '.idx'
    ds = make_builder(out_file, impl=impl, dtype=dtype)
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                line = tokenize_line(line)
                line = list(map(int, line))
                line = torch.tensor(line)
                ds.add_item(line)
            else:
                raise Exception('empty line')

    ds.finalize(index_file)

    return
