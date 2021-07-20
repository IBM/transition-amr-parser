import hashlib
import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm

try:
    import allennlp.modules.elmo as elmo
except ImportError:
    print('warning: No allennlp installed.')

from align_cfg.vocab_definitions import BOS_TOK, EOS_TOK, special_tokens


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


def get_character_embeddings_from_elmo(tokens, cuda=False):
    assert len(special_tokens) == 3
    assert tokens[1] == BOS_TOK and tokens[2] == EOS_TOK

    # Remove special tokens.
    vocab_to_cache = tokens[3:]

    # OLD

    model = elmo.Elmo(options_file=options_file, weight_file=weights_file,
                      requires_grad=False, num_output_representations=1)
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

    # NEW

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

    embeddings = get_character_embeddings_from_elmo(tokens, args.cuda)

    path = f'{args.cache_dir}/elmo.{token_hash}.npy'
    print(f'writing to {path}')
    write_embeddings(path, embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-text", type=str, help="Vocab file.",
                        required=True)
    parser.add_argument('--cuda', action='store_true',
                        help='If true, then use GPU.')
    parser.add_argument('--cache-dir', required=True,
                        help='Folder where to store e,ebddings')
    args = parser.parse_args()

    main(args)
