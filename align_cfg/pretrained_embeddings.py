import hashlib
import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm

from align_cfg.vocab_definitions import BOS_TOK, EOS_TOK, special_tokens
from align_cfg.standalone_elmo import batch_to_ids, ElmoCharacterEncoder, remove_sentence_boundaries


# files for original elmo model
weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'


def maybe_download(remote_url, cache_dir):
    path = os.path.join(cache_dir, os.path.basename(remote_url))
    if not os.path.exists(path):
        os.system(f'curl {remote_url} -o {path} -L')
    return path


def hash_string_list(string_list):
    m = hashlib.sha256()
    for s in string_list:
        m.update(str.encode(s))
    return m.hexdigest()[:8]


def read_text_vocab_file(path):
    output = []
    with open(path) as f:
        for line in f:
            output.append(line.rstrip())
    return output


def read_amr_vocab_file(path):
    output = []
    with open(path) as f:
        for line in f:
            output.append(line.rstrip())
    return output


def get_character_embeddings_from_elmo(tokens, cache_dir, cuda=False):
    assert len(special_tokens) == 3
    assert tokens[1] == BOS_TOK and tokens[2] == EOS_TOK

    # Remove special tokens.
    vocab_to_cache = tokens[3:]

    size = 512
    batch_size = 1024

    char_embedder = ElmoCharacterEncoder(
        options_file=maybe_download(options_file, cache_dir=cache_dir),
        weight_file=maybe_download(weights_file, cache_dir=cache_dir),
        requires_grad=False)
    if cuda:
        char_embedder.cuda()

    all_vocab_to_cache = [BOS_TOK, EOS_TOK] + vocab_to_cache

    shape = (1 + len(all_vocab_to_cache), size)
    embeddings = np.zeros(shape, dtype=np.float32)

    for start in tqdm(range(0, len(all_vocab_to_cache), batch_size), desc='embed'):
        end = min(start + batch_size, len(all_vocab_to_cache))
        batch = all_vocab_to_cache[start:end]
        batch_ids = batch_to_ids([[x] for x in batch])
        output = char_embedder(batch_ids)
        vec = remove_sentence_boundaries(output['token_embedding'], output['mask'])[0].squeeze(1)

        embeddings[1 + start:1 + end] = vec

    return embeddings


def read_embeddings(tokens, path=None, cache_dir=None):
    if path is None:
        token_hash = hash_string_list(tokens)
        if cache_dir:
            path = '{}/elmo.{}.npy'.format(cache_dir, token_hash)
        else:
            path = 'elmo.{}.npy'.format(token_hash)
        assert os.path.exists(path), path
    embeddings = np.load(path)
    assert embeddings.shape[0] == len(tokens)
    return embeddings


def write_embeddings(path, embeddings):
    np.save(path, embeddings)

    with open(path + '.shape', 'w') as f:
        f.write(json.dumps(embeddings.shape))


def main(arg):

    tokens = read_text_vocab_file(args.vocab_text)
    token_hash = hash_string_list(tokens)

    print('found {} tokens with hash = {}'.format(len(tokens), token_hash))

    embeddings = get_character_embeddings_from_elmo(tokens, args.cache_dir, args.cuda)

    path = f'{args.cache_dir}/elmo.{token_hash}.npy'
    print(f'writing to {path}')
    write_embeddings(path, embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-text", type=str, help="Vocab file.",
                        required=True)
    parser.add_argument('--cuda', action='store_true',
                        help='If true, then use GPU.')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Folder to save elmo weights and embeddings.')
    args = parser.parse_args()

    main(args)
