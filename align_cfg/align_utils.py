import argparse
import hashlib
import os
import numpy as np
from tqdm import tqdm

from transition_amr_parser.io import read_amr
from amr_utils import get_node_ids


def hash_corpus(corpus):
    m = hashlib.md5()
    for amr in corpus:
        m.update(' '.join(amr.tokens).encode())
        m.update(' '.join(sorted(amr.nodes.keys())).encode())
    return m.hexdigest()


def save_align_dist(path, corpus, dist_list):
    corpus_id = hash_corpus(corpus)
    print(
        'writing to {} with corpus_id {}'.format(
            os.path.abspath(path),
            corpus_id
        )
    )

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

    np_align_dist = np.memmap(
        path, dtype=np.float32, shape=(total_size, 1), mode='w+'
    )
    np_align_dist[:] = align_dist

    return align_dist, corpus_id


def load_align_dist(path, corpus):
    corpus_id = hash_corpus(corpus)
    print(
        'reading from {} and current corpus has corpus_id {}'.format(
            os.path.abspath(path),
            corpus_id
        )
    )

    sizes = list(map(lambda x: len(x.tokens) * len(x.nodes), corpus))
    assert all([size > 0 for size in sizes])
    total_size = sum(sizes)
    offsets = np.zeros(len(corpus), dtype=np.int)
    offsets[1:] = np.cumsum(sizes[:-1])

    np_align_dist = np.memmap(path, dtype=np.float32, shape=(total_size, 1), mode='r')
    align_dist = np.zeros((total_size, 1), dtype=np.float32)
    align_dist[:] = np_align_dist[:]

    dist_list = []
    for idx, amr in tqdm(enumerate(corpus), desc='read-align-dist'):
        offset = offsets[idx]
        size = sizes[idx]
        assert size == len(amr.tokens) * len(amr.nodes)

        dist = align_dist[offset:offset+size].reshape(len(amr.nodes), len(amr.tokens))
        dist_list.append(dist)

    return align_dist, dist_list, corpus_id


def save_amr_align_argmax(path, corpus, dist_list, path_gold=None, write_gold=False):
    with open(path, 'w') as f:
        for amr, dist in zip(corpus, dist_list):
            n = len(amr.nodes)
            nt = len(amr.tokens)
            assert dist.shape == (n, nt)

            nodes = get_node_ids(amr)
            argmax = dist.argmax(-1).reshape(-1).tolist()
            assert len(nodes) == len(argmax)

            amr.alignments = {k: [v] for k, v in zip(nodes, argmax)}

            f.write(f'{amr.__str__()}\n')


def load_align_dist_pretty(path, corpus):

    state = 0

    results = {}

    with open(path) as f:
        for line in tqdm(f, desc='read-align-dist-pretty'):
            line = line.strip()

            if not line:
                state = 0
                continue

            if state == 0:
                idx = int(line)
                amr = corpus[idx]
                n, nt = len(amr.nodes), len(amr.tokens)
                results[idx] = {}

                key = 'example_id'
                val = idx
                results[idx][key] = val

                key = 'amr'
                val = amr
                results[idx][key] = val

                key = 'dist'
                val = np.zeros((n, nt), dtype=np.float32)
                results[idx][key] = val

                key = 'total'
                val = 0
                results[idx][key] = val

            elif state == 1: # node names
                pass

            elif state == 2: # node ids
                # TODO: Verify order of node ids.
                pass

            elif state == 3: # tokens
                pass

            else: # dist
                i, j, x = line.split()
                i = int(i)
                j = int(j)
                x = float(x)

                results[idx]['dist'][i, j] = x
                results[idx]['total'] += 1

            state += 1

    # check
    for idx in range(len(corpus)):
        amr = corpus[idx]
        res = results[idx]
        n, nt = len(amr.nodes), len(amr.tokens)
        assert n * nt == res['total']

    # result
    dist_list = [results[idx]['dist'] for idx in range(len(corpus))]

    return dist_list


def main(args):
    if args.mode == 'write_argmax':
        """
        Read alignment distributions and write argmax alignments to file.
        """
        corpus = read_amr(args.in_amr, jamr=args.jamr)
        if args.in_amr_align_dist is not None:
            align_dist, dist_list, corpus_id = load_align_dist(args.in_amr_align_dist, corpus)
        else:
            dist_list = load_align_dist(args.in_amr_align_dist_pretty, corpus)
        save_amr_align_argmax(args.out_amr_aligned, corpus, dist_list)

    elif args.mode == 'compare_dist':
        """
        Verify two alignment distributions are the same.
        """
        corpus = read_amr(args.in_amr, jamr=args.jamr)
        dist_list_pretty = load_align_dist_pretty(args.in_amr_align_dist_pretty, corpus)
        align_dist, dist_list, corpus_id = load_align_dist(args.in_amr_align_dist, corpus)

        # check argmax
        assert len(dist_list) == len(dist_list_pretty), (len(dist_list), len(dist_list_pretty))

        def compare_argmax(d1, d2):
            return np.all(d1.argmax(-1) == d2.argmax(-1)).item()

        for idx, (d1, d2) in enumerate(zip(dist_list, dist_list_pretty)):
            assert compare_argmax(d1, d2)

        # check exact vals

        def compare_vals(d1, d2):
            return np.allclose(d1, d2)

        for idx, (d1, d2) in enumerate(zip(dist_list, dist_list_pretty)):
            assert compare_vals(d1, d2)

        print('OKAY')

    elif args.mode == 'compare_argmax':
        """
        Verify two alignment distributions are the same.
        """
        # corpus = read_amr(args.in_amr, jamr=True) ?
        corpus = read_amr(args.in_amr, jamr=args.jamr)
        if args.in_amr_align_dist_pretty is not None:
            dist_list = load_align_dist_pretty(args.in_amr_align_dist_pretty, corpus)
        else:
            align_dist, dist_list, corpus_id = load_align_dist(args.in_amr_align_dist, corpus)

        assert len(corpus) == len(dist_list)

        def compare_(amr, dist):
            argmax = dist.argmax(-1).reshape(-1).tolist()

            for i, k in enumerate(sorted(amr.nodes.keys())):
                a = amr.alignments[k][0]
                if a != argmax[i]:
                    return False

            return True

        for idx, (dist, amr) in enumerate(zip(dist_list, corpus)):
            assert compare_(amr, dist)

        print('OKAY')

    elif args.mode == 'verify_corpus_id':
        """
        Verify corpus matches already saved corpus id. Corpus ID is a hash
        generated by tokens and node ids.
        """
        corpus = read_amr(args.in_amr, jamr=args.jamr)
        corpus_id = hash_corpus(corpus)
        with open(args.corpus_id) as f:
            check_corpus_id = f.read().strip()

        print('INPUT: {}'.format(corpus_id))
        print('CHECK: {}'.format(check_corpus_id))

        assert corpus_id == check_corpus_id, (corpus_id, check_corpus_id)

        print('OKAY')

    elif args.mode == 'read':
        """
        Verify corpus matches already saved corpus id. Corpus ID is a hash
        generated by tokens and node ids.
        """
        corpus = read_amr(args.in_amr, jamr=args.jamr)
        corpus_id = hash_corpus(corpus)

        print(args.in_amr, len(corpus), corpus_id)

        print('OKAY')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=(
            'write_argmax', 'compare_dist', 'compare_argmax',
            'verify_corpus_id', 'read'
        ), help="See main() for mode descriptions."
    )
    parser.add_argument('--in-amr', default=None, type=str, help="Path to input amr file.")
    parser.add_argument('--in-amr-align-dist', default=None, type=str, help="Path to input alignment distribution with np.memmap.")
    parser.add_argument('--in-amr-align-dist-pretty', default=None, type=str, help="Path to input alignment distribution with pretty format.")
    parser.add_argument('--out-amr-aligned', default=None, type=str, help="Path to output amr with alignments.")
    parser.add_argument('--corpus-id', default=None, type=str, help="Path to .corpus_hash file.")
    parser.add_argument('--jamr', action='store_true')
    args = parser.parse_args()
    main(args)
