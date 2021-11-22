import argparse
import copy
import hashlib
import os
import numpy as np

from transition_amr_parser.io import read_amr2
from amr_utils import get_node_ids
from formatter import amr_to_pretty_format, amr_to_string


def hash_corpus(corpus):
    m = hashlib.md5()
    for amr in corpus:
        m.update(' '.join(amr.tokens).encode())
        m.update(' '.join(sorted(amr.nodes.keys())).encode())
    return m.hexdigest()


def save_align_dist(path, corpus, dist_list):
    corpus_id = hash_corpus(corpus)
    print('writing to {} with corpus_id {}'.format(os.path.abspath(path), corpus_id))

    assert isinstance(dist_list, list)

    sizes = list(map(lambda x: len(x.tokens) * len(x.nodes), corpus))
    assert all([size > 0 for size in sizes])
    total_size = sum(sizes)
    offsets = np.zeros(len(corpus), dtype=np.int)
    offsets[1:] = np.cumsum(sizes[:-1])

    align_dist = np.zeros((total_size, 1), dtype=np.float32)

    for idx, dist in enumerate(dist_list):
        amr = corpus[idx]
        offset = offsets[idx]
        size = sizes[idx]
        align_dist[offset:offset + size] = dist.reshape(size, 1)

    np_align_dist = np.memmap(path, dtype=np.float32, shape=(total_size, 1), mode='w+')
    np_align_dist[:] = align_dist

    return align_dist, corpus_id


def load_align_dist(path, corpus):
    corpus_id = hash_corpus(corpus)
    print('reading from {} and current corpus has corpus_id {}'.format(os.path.abspath(path), corpus_id))

    sizes = list(map(lambda x: len(x.tokens) * len(x.nodes), corpus))
    assert all([size > 0 for size in sizes])
    total_size = sum(sizes)
    offsets = np.zeros(len(corpus), dtype=np.int)
    offsets[1:] = np.cumsum(sizes[:-1])

    np_align_dist = np.memmap(path, dtype=np.float32, shape=(total_size, 1), mode='r')
    align_dist = np.zeros((total_size, 1), dtype=np.float32)
    align_dist[:] = np_align_dist[:]

    dist_list = []
    for idx, amr in enumerate(corpus):
        offset = offsets[idx]
        size = sizes[idx]
        assert size == len(amr.tokens) * len(amr.nodes)

        dist = align_dist[offset:offset+size].reshape(len(amr.nodes), len(amr.tokens))
        dist_list.append(dist)

    return align_dist, dist_list, corpus_id

def write_align(corpus, dataset, path_gold, path_pred, write_gold=True):
    net.eval()

    indices = np.arange(len(corpus))

    predictions = collections.defaultdict(list)

    with torch.no_grad():
        for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
            end = min(start + batch_size, len(corpus))
            batch_indices = indices[start:end]
            items = [dataset[idx] for idx in batch_indices]
            batch_map = batchify(items, cuda=args.cuda)

            # forward pass
            model_output = shared_validation_step(net, batch_indices, batch_map)

            # save alignments for eval.
            for idx, ainfo in zip(batch_indices, AlignmentDecoder().batch_decode(batch_map, model_output)):
                amr = corpus[idx]
                node_ids = get_node_ids(amr)
                alignments = {node_ids[node_id]: a for node_id, a in ainfo['node_alignments']}

                predictions['amr'].append(amr)
                predictions['alignments'].append(alignments)# write alignments

    # write pred
    with open(path_pred, 'w') as f_pred:
        for amr, alignments in zip(predictions['amr'], predictions['alignments']):
            f_pred.write(amr_to_string(amr, alignments).strip() + '\n\n')

    # write gold
    if write_gold:
        with open(path_gold, 'w') as f_gold:
            for amr, alignments in zip(predictions['amr'], predictions['alignments']):
                   f_gold.write(amr_to_string(amr).strip() + '\n\n')


def save_amr_align_argmax(path, corpus, dist_list, path_gold=None, write_gold=False):
    with open(path, 'w') as f:
        for amr, dist in zip(corpus, dist_list):
            n = len(amr.nodes)
            nt = len(amr.tokens)
            assert dist.shape == (n, nt)

            nodes = get_node_ids(amr)
            argmax = dist.argmax(-1).reshape(-1).tolist()
            assert len(nodes) == len(argmax)

            alignments = {k: [v] for k, v in zip(nodes, argmax)}

            f.write(amr_to_string(amr, alignments).strip() + '\n\n')


def main(args):
    corpus = read_amr2(args.in_amr, ibm_format=False, tokenize=False)
    align_dist, dist_list, corpus_id = load_align_dist(args.in_amr_align_dist, corpus)
    save_amr_align_argmax(args.out_amr_aligned, corpus, dist_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-amr', default=None, type=str)
    parser.add_argument('--in-amr-align-dist', default=None, type=str)
    parser.add_argument('--out-amr-aligned', default=None, type=str)
    args = parser.parse_args()
    main(args)

