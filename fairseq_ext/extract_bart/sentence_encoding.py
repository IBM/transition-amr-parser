import copy

import torch

from ..data.data_utils import collate_tokens
from ..utils_font import yellow_font
from fairseq_ext.roberta.pretrained_embeddings import PretrainedEmbeddings


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
    word_to_wordpiece_count = 0
    subword_sequence = []
    for wp_index in range(len(wordpiece_tokens)):
        word = word_tokens[w_index]

        # debug
        # if subword_sequence and word == wordpiece_tokens[wp_index]:
        #     # e.g. when the subword is ' ', and the next subword is
        #     # an exact match with the current word with no leading white space
        #     print('-' * 10, 'corner case when a subword would be skipped')
        #     print('subword_sequence index:', subword_sequence)
        #     print('subword_sequence string', "".join([wordpiece_tokens[i] for i in subword_sequence]))
        #     print('-' * 10)

        if not subword_sequence and word == wordpiece_tokens[wp_index]:
            # NOTE when subword_sequence is not empty, we should not enter here;
            #      otherwise it will case a subword skipped (e.g. when the subword is ' ', and the next subword is
            #      an exact match with the current word with no leading white space)
            word_to_wordpiece.append(wp_index)
            w_index += 1
            word_to_wordpiece_count += 1
        else:
            subword_sequence.append(wp_index)
            word_from_pieces = "".join([
                wordpiece_tokens[i]
                for i in subword_sequence
            ])
            # NOTE: Facebooks BPE signals SOW with whitesplace
            word_from_pieces = word_from_pieces.lstrip()
            if word == word_from_pieces:
                word_to_wordpiece.append(subword_sequence)
                w_index += 1
                word_to_wordpiece_count += len(subword_sequence)
                subword_sequence = []

            assert word_from_pieces in word, \
                "wordpiece must be at least a segment of current word"

    assert word_to_wordpiece_count == len(wordpiece_tokens), 'every subword token must be mapped to a word'

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


class SentenceEmbeddingRoberta(PretrainedEmbeddings):
    def __init__(self, name, bert_layers, model=None, remove_be=False, avg_word=False):
        super().__init__(name, bert_layers, model)
        self.remove_be = remove_be    # whether to remove <s> and </s>
        self.avg_word = avg_word      # whether to average the embeddings from subtokens to words

    def extract(self, sentence_string):
        """
        sentence_string (not tokenized)
        """

        # get words, wordpieces and mapping
        # FIXME: PTB oracle already tokenized
        word2piece = get_wordpiece_to_word_map(
            sentence_string,
            self.roberta.bpe
        )

        # NOTE: We need to re-extract BPE inside roberta. Token indices
        # will also be different. BOS/EOS added
        wordpieces_roberta = self.roberta.encode(sentence_string)

        # Extract roberta, remove BOS/EOS
        if torch.cuda.is_available():

            # Hotfix for sequences above 512
            if wordpieces_roberta.shape[0] > 512:
                excess = wordpieces_roberta.shape[0] - 512
                # first 512 tokens
                last_layer = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)[:512]
                )
                # last 512 tokens
                last_layer2 = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)[excess:]
                )
                # concatenate
                shape = (last_layer, last_layer2[:, -excess:, :])
                last_layer = torch.cat(shape, 1)

                assert wordpieces_roberta.shape[0] == last_layer.shape[1]

                # warn user about this
                string = '\nMAX_POS overflow!! {wordpieces_roberta.shape[0]}'
                print(yellow_font(string))

            else:

                # Normal extraction
                last_layer = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)
                )

        else:

            # Copy code above
            raise NotImplementedError()
            last_layer = self.roberta.extract_features(
                wordpieces_roberta
            )

        # FIXME: this should not bee needed using roberta.eval()
        last_layer = last_layer.detach()

        # Ignore start and end symbols
        if self.remove_be:
            last_layer = last_layer[0:1, 1:-1, :]

        # average over wordpieces of same word
        if self.avg_word:
            assert self.remove_be
            word_features = get_average_embeddings(
                last_layer,
                word2piece
            )
        else:
            word_features = last_layer

#        # sanity check differentiable and non differentiable averaging
#        match
#        from torch_scatter import scatter_mean
#        word_features2 = scatter_mean(
#            last_layer[0, :, :],
#            get_scatter_indices(word2piece).to(roberta.device),
#            dim=0
#        )
#        # This works
#        assert np.allclose(word_features.cpu(), word_features2.cpu())

        return word_features, wordpieces_roberta, word2piece
