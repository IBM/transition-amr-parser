import copy

import torch

from ..data.data_utils import collate_tokens
from ..utils_font import yellow_font


def get_average_embeddings(final_layer, word2piece):

    # Average worpiece representations to get word representations
    num_words = len(word2piece)
    batch_dim, num_wordpieces, hidden_size = final_layer.shape
    assert batch_dim == 1, "batch_size must be 1"
    if num_words < num_wordpieces:
        word_features = torch.zeros(
            (1, num_words, hidden_size)
        ).to(final_layer.device)
        for word_idx, wordpiece_idx in enumerate(word2piece):
            # column of features for all involved worpieces
            column = final_layer[0:1, wordpiece_idx, :]
            if isinstance(wordpiece_idx, list):
                column = column.mean(1, keepdim=True)
            word_features[0:1, word_idx, :] = column
    else:
        word_features = final_layer

    return word_features


def get_wordpiece_to_word_map(sentence, roberta_bpe):

    # Get word and wordpiece tokens according to RoBERTa
    word_tokens = sentence.split()
    wordpiece_tokens = [
        roberta_bpe.decode(wordpiece)
        for wordpiece in roberta_bpe.encode(sentence).split()
    ]

    assert len(word_tokens) <= len(wordpiece_tokens)
    assert isinstance(word_tokens, list)
    assert isinstance(wordpiece_tokens, list)

    w_index = 0
    word_to_wordpiece = []
    subword_sequence = []
    for wp_index in range(len(wordpiece_tokens)):
        word = word_tokens[w_index]
        if word == wordpiece_tokens[wp_index]:
            word_to_wordpiece.append(wp_index)
            w_index += 1
        else:
            subword_sequence.append(wp_index)
            word_from_pieces = "".join([
                # NOTE: Facebooks BPE signals SOW with whitesplace
                wordpiece_tokens[i].lstrip()
                for i in subword_sequence
            ])
            if word == word_from_pieces:
                word_to_wordpiece.append(subword_sequence)
                w_index += 1
                subword_sequence = []

            assert word_from_pieces in word, \
                "wordpiece must be at least a segment of current word"

    return word_to_wordpiece


def get_scatter_indices(word2piece, reverse=False):
    if reverse:
        indices = range(len(word2piece))[::-1]
    else:
        indices = range(len(word2piece))
    # we will need as well the wordpiece to word indices
    wp_indices = [
        [index] * (len(span) if isinstance(span, list) else 1)
        for index, span in zip(indices, word2piece)
    ]
    # flatten the list
    wp_indices = [x for span in wp_indices for x in span]
    return torch.tensor(wp_indices)


class SentenceEncodingBART:
    def __init__(self, name):
        # bart model name
        self.name = name

        if name in ['bart.base', 'bart.large']:
            self.model = torch.hub.load('pytorch/fairseq', name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
                print(f'Using {name} extraction in GPU')
            else:
                print(f'Using {name} extraction in cpu (slow, wont OOM)')
        else:
            raise Exception(f'Unknown pretrained model name or path {name}')

    def encode_sentence(self, sentence_string):
        """BPE tokenization and numerical encoding based on model vocabulary.

        Args:
            sentence_string (str): sentence string, not tokenized.

        Raises:
            Exception: [description]
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        # get numerical encoding of the sentence
        # NOTE bpe token ids include BOS `<s>` and EOS `</s>`
        wordpiece_ids = self.model.encode(sentence_string)

        # get word to word piece mapping
        # NOTE the mapping index does not consider BOS `<s>` and EOS `</s>`
        word2piece = get_wordpiece_to_word_map(
            sentence_string,
            self.model.bpe
        )

        return wordpiece_ids, word2piece
