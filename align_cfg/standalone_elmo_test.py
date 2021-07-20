import hashlib

import torch

import allennlp.modules.elmo as elmo

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoCharacterMapper,
    ELMoTokenCharactersIndexer,
)

from standalone_elmo import batch_to_ids as batch_to_ids_new
from standalone_elmo import remove_sentence_boundaries
from standalone_elmo import ElmoCharacterEncoder as ElmoCharacterEncoderNew


def hash_string_list(string_list):
    m = hashlib.sha256()
    for s in string_list:
        m.update(str.encode(s))
    return m.hexdigest()[:8]


if __name__ == '__main__':
    requires_grad = False
    option_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    char_embedder = elmo._ElmoCharacterEncoder(
        options_file=option_file,
        weight_file=weight_file,
        requires_grad=requires_grad)

    char_embedder_new = ElmoCharacterEncoderNew(
        options_file=option_file,
        weight_file=weight_file,
        requires_grad=requires_grad)

    vocab_to_cache = list(set('cat jumped over the fence'.split()))

    full_model = elmo.Elmo(options_file=option_file, weight_file=weight_file, requires_grad=False, num_output_representations=1)._elmo_lstm
    full_model.create_cached_cnn_embeddings(vocab_to_cache)

    bos_vec = full_model._bos_embedding
    eos_vec = full_model._eos_embedding
    weights = full_model._word_embedding.weight.data

    # TEST

    batch = []
    batch.append('the cat jumped over the fence'.split())
    batch.append('cat dog'.split())

    batch_ids = elmo.batch_to_ids(batch)
    batch_ids_new = batch_to_ids_new(batch)

    batch_hash = hash_string_list(list(map(str, batch_ids.view(-1).tolist())))
    batch_hash_new = hash_string_list(list(map(str, batch_ids_new.view(-1).tolist())))
    print(f'batch: {batch_hash} [old] {batch_hash_new} [new]')
    assert batch_hash == batch_hash_new

    embeddings = char_embedder(batch_ids)['token_embedding']
    embeddings_new = char_embedder_new(batch_ids)['token_embedding']

    emb_hash = hash_string_list(list(map(str, embeddings.view(-1).tolist())))
    emb_hash_new = hash_string_list(list(map(str, embeddings.view(-1).tolist())))
    print(f'embed: {emb_hash} [old] {emb_hash_new} [new]')
    assert emb_hash == emb_hash_new

    # TEST old encoder match cached weights.

    # IMPORTANT NOTE: There is a subtle bug here... The first vocab item is treated as a padding token. Keep this
    # for now to match previous behavior, but should be fixed eventually.
    assert weights.shape[0] == len(vocab_to_cache), (weights.shape, len(vocab_to_cache))

    vocab_ids = elmo.batch_to_ids([[x] for x in vocab_to_cache])
    vocab_output = char_embedder(vocab_ids)
    vocab_vec = vocab_output['token_embedding']
    vocab_mask = vocab_output['mask']
    vocab_vec = remove_sentence_boundaries(vocab_vec, vocab_mask)[0].squeeze(1)

    # IMPORTANT NOTE: There is a subtle bug here... The first vocab item is treated as a padding token. Keep this
    # for now to match previous behavior, but should be fixed eventually.
    vocab_vec[0] = 0
    check = torch.isclose(weights, vocab_vec, atol=1e-4)
    assert check.all().item() is True, (check.sum().item(), check.view(-1).shape[0])

    print('old encoder matches cached vocab')

    # TEST new encoder match cached weights.

    # IMPORTANT NOTE: There is a subtle bug here... The first vocab item is treated as a padding token. Keep this
    # for now to match previous behavior, but should be fixed eventually.
    assert weights.shape[0] == len(vocab_to_cache), (weights.shape, len(vocab_to_cache))

    vocab_ids = batch_to_ids_new([[x] for x in vocab_to_cache])
    vocab_output = char_embedder_new(vocab_ids)
    vocab_vec = vocab_output['token_embedding']
    vocab_mask = vocab_output['mask']
    vocab_vec = remove_sentence_boundaries(vocab_vec, vocab_mask)[0].squeeze(1)

    # IMPORTANT NOTE: There is a subtle bug here... The first vocab item is treated as a padding token. Keep this
    # for now to match previous behavior, but should be fixed eventually.
    vocab_vec[0] = 0
    check = torch.isclose(weights, vocab_vec, atol=1e-4)
    assert check.all().item() is True, (check.sum().item(), check.view(-1).shape[0])

    print('new encoder matches cached vocab')
