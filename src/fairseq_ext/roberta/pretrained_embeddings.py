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

    # Get word and wordpiece tokens according to GPT2BPE (used by RoBERTa/BART)
    word_tokens = sentence.split()
    # NOTE this only returns the surface form of each byte encoding, which will not match as a subsequence of some
    #      characters such as '\x91' and chinese symbols -> we need to dynamically recover the chars from utf8 bytes
    # NOTE this is NOT used for matching
    # wordpiece_tokens = [
    #     roberta_bpe.decode(wordpiece)
    #     for wordpiece in roberta_bpe.encode(sentence).split()
    # ]
    # NOTE we need to use lower level bpe encodings to handle all characters such as chinese and u'\x91'
    #      the lower level bye bytes are used for matching
    wordpiece_bpe_ids = roberta_bpe.bpe.encode(sentence)    # List[int] corresponding to bpe vocab

    assert len(word_tokens) <= len(wordpiece_bpe_ids)
    assert isinstance(word_tokens, list)
    assert isinstance(wordpiece_bpe_ids, list)
    # assert isinstance(wordpiece_tokens, list)
    # assert len(wordpiece_tokens) == len(wordpiece_bpe_ids)

    w_index = 0
    word_to_wordpiece = []    # List[List[int]]
    subword_sequence = []
    bpe_id_sequence = []

    for wp_index, bpe_id in enumerate(wordpiece_bpe_ids):
        word = word_tokens[w_index]
        # only the initial word doesn't need whitespace at the beginning to be matched
        if w_index > 0:
            word = ' ' + word

        subword_sequence.append(wp_index)
        bpe_id_sequence.append(bpe_id)
        word_from_pieces = roberta_bpe.bpe.decode(bpe_id_sequence)    # this recovers any original characters
        if word == word_from_pieces:
            word_to_wordpiece.append(subword_sequence)
            w_index += 1
            subword_sequence = []
            bpe_id_sequence = []

    assert len(word_tokens) == len(word_to_wordpiece), 'word_to_wordpiece must be of the same size of the word_tokens'
    assert word_to_wordpiece[0][0] == 0 and word_to_wordpiece[-1][-1] == len(wordpiece_bpe_ids) - 1, \
        'word_to_wordpiece mapping must cover all wordpieces, from the beginning towards the end'

    # # # debug: AMR3.0 training data '\x91'
    # if any(['\x91' in w for w in word_tokens]):
    #     breakpoint()

    # # # debug: AMR3.0 training data chinese characters
    # # if sentence.startswith('< b > Suining County < / b > ( simplified Chinese'):
    # # OR equivalently below
    # if len(word_tokens) == 53 and len(wordpiece_bpe_ids) == 80:
    #     breakpoint()

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
    wp_indices = [x for span in wp_indices for x in span]
    return torch.tensor(wp_indices)


class PretrainedEmbeddings():

    def __init__(self, name, bert_layers, model=None):

        # embedding type name
        self.name = name
        # select some layers for averaging
        self.bert_layers = bert_layers

        if model is None:
            if name in ['roberta.base', 'roberta.large']:

                # Extract
                self.roberta = torch.hub.load('pytorch/fairseq', name)
                self.roberta.eval()
                if torch.cuda.is_available():
                    self.roberta.cuda()
                    print(f'Using {name} extraction in GPU')
                else:
                    print(f'Using {name} extraction in cpu (slow, wont OOM)')

            else:
                raise Exception(
                    f'Unknown --pretrained-embed {name}'
                )
        else:
            self.roberta = model

    def extract_features(self, worpieces):
        """Extract features from wordpieces"""

        if self.bert_layers is None:
            # normal RoBERTa
            return self.roberta.extract_features(worpieces)
        else:
            # layer average RoBERTa
            features = self.roberta.extract_features(
                worpieces,
                return_all_hiddens=True
            )
            # sum layers
            feature_layers = []
            for layer_index in self.bert_layers:
                feature_layers.append(features[layer_index])
            feature_layers = sum(feature_layers)
            return torch.div(feature_layers, len(self.bert_layers))

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
        worpieces_roberta = self.roberta.encode(sentence_string)

        # Extract roberta, remove BOS/EOS
        if torch.cuda.is_available():

            # Hotfix for sequences above 512
            if worpieces_roberta.shape[0] > 512:
                excess = worpieces_roberta.shape[0] - 512
                # first 512 tokens
                last_layer = self.extract_features(
                    worpieces_roberta.to(self.roberta.device)[:512]
                )
                # last 512 tokens
                last_layer2 = self.extract_features(
                    worpieces_roberta.to(self.roberta.device)[excess:]
                )
                # concatenate
                shape = (last_layer, last_layer2[:, -excess:, :])
                last_layer = torch.cat(shape, 1)

                assert worpieces_roberta.shape[0] == last_layer.shape[1]

                # warn user about this
                string = '\nMAX_POS overflow!! {worpieces_roberta.shape[0]}'
                print(yellow_font(string))

            else:

                # Normal extraction
                last_layer = self.extract_features(
                    worpieces_roberta.to(self.roberta.device)
                )

        else:

            # Copy code above
            raise NotImplementedError()
            last_layer = self.roberta.extract_features(
                worpieces_roberta
            )

        # FIXME: this should not bee needed using roberta.eval()
        last_layer = last_layer.detach()

        # Ignore start and end symbols
        last_layer = last_layer[0:1, 1:-1, :]

        # average over wordpieces of same word
        word_features = get_average_embeddings(
            last_layer,
            word2piece
        )

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

        return word_features, worpieces_roberta, word2piece

    def extract_batch(self, sentence_string_batch):
        bert_data = {}
        bert_data["word_features"] = []
        bert_data["wordpieces_roberta"] = []
        bert_data["word2piece_scattered_indices"] = []
        src_wordpieces = []
        src_word2piece = []
        for sentence in sentence_string_batch:
            word2piece = get_wordpiece_to_word_map(sentence, self.roberta.bpe)
            wordpieces_roberta = self.roberta.encode(sentence)
            wordpieces_roberta = wordpieces_roberta[:512]
            src_wordpieces.append(copy.deepcopy(wordpieces_roberta))
            src_word2piece.append(copy.deepcopy(word2piece))

        src_wordpieces_collated = collate_tokens(src_wordpieces, pad_idx=1)
        roberta_batch_features = self.extract_features(src_wordpieces_collated)
        roberta_batch_features = roberta_batch_features.detach().cpu()
        for index,(word2piece, wordpieces_roberta) in enumerate(zip(src_word2piece, src_wordpieces)):
            roberta_features = roberta_batch_features[index]
            roberta_features = roberta_features[1:len(wordpieces_roberta)-1]
            word_features = get_average_embeddings(roberta_features.unsqueeze(0), word2piece)
            word2piece_scattered_indices = get_scatter_indices(word2piece, reverse=True)
            bert_data["word_features"].append(word_features[0])
            bert_data["wordpieces_roberta"].append(wordpieces_roberta)
            bert_data["word2piece_scattered_indices"].append(word2piece_scattered_indices)

        return bert_data
