import hashlib
import json
import numpy as np
import os
import torch

try:
    import allennlp.modules.elmo as elmo
except:
    print('warning: No allennlp installed.')

from tqdm import tqdm

from transition_amr_parser.io import read_amr

from vocab import *


# files for original elmo model
weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'


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


def read_tokens_from_amr(files):
    tokens = set()

    for path in files:
        path = os.path.expanduser(path)
        for amr in tqdm(read_amr(path).amrs, desc='read'):
            tokens.update(amr.tokens)

    tokens = special_tokens + sorted(tokens)

    return tokens


def get_character_embeddings_from_elmo(tokens, cuda=False):
    assert len(special_tokens) == 3
    assert tokens[1] == BOS_TOK and tokens[2] == EOS_TOK

    # Remove special tokens.
    vocab_to_cache = tokens[3:]

    model = elmo.Elmo(options_file=options_file, weight_file=weights_file, requires_grad=False, num_output_representations=1)
    model = model._elmo_lstm
    if cuda:
        model.cuda()

    with torch.no_grad():
        model.create_cached_cnn_embeddings(vocab_to_cache)

    size = model._word_embedding.weight.shape[1]
    shape = (len(tokens), size)
    embeddings = np.zeros(shape, dtype=np.float32)
    # embeddings[0] = 0 # padding token
    embeddings[1] = model._bos_embedding.cpu().numpy()
    embeddings[2] = model._eos_embedding.cpu().numpy()
    embeddings[3:] = model._word_embedding.weight.cpu().numpy()

    return embeddings


def read_embeddings(tokens, path=None):
    if path is None:
        token_hash = hash_string_list(tokens)
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

    embeddings = get_character_embeddings_from_elmo(tokens, args.cuda)

    path = 'elmo.{}.npy'.format(token_hash)
    print('writing to {}'.format(path))
    write_embeddings(path, embeddings)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-text", type=str, default='./align_cfg/vocab.text.2021-06-28.txt',
                        help="Vocab file.")
    parser.add_argument('--cuda', action='store_true',
                        help='If true, then use GPU.')
    args = parser.parse_args()

    main(args)
