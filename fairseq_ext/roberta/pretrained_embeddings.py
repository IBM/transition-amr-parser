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

    # Get word and worpiece tokens according to RoBERTa
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
                elif word_from_pieces not in word:
                    word_to_wordpiece.append(subword_sequence)
                    w_index += 1
                    subword_sequence = []
                    bad_unicode_flag = 1
                # assert word_from_pieces in word, \
                #    "wordpiece must be at least a segment of current word"
    if bad_unicode_flag==0:
        #assert len(word_tokens) == len(word_to_wordpiece)
        if len(word_tokens) != len(word_to_wordpiece):
            print("sentence: ", sentence)
            print("wordpiecetokens: ", wordpiece_tokens)
            print("word token count: ", len(word_tokens))
            print("word_to_wordpiece count: ", len(word_to_wordpiece))
        return word_to_wordpiece
    else:
        # remove extra bad token 'fffd' for oov 2-byte characters
        wptok = []
        i = 0
        while(i < len(wordpiece_tokens)):
            x = wordpiece_tokens[i]
            if ord(x[-1:]) != 65533:
                wptok.append(x)
                i += 1
            else:
                print("X: ", x)
                nx = wordpiece_tokens[i+1]
                if ord(x[-1:])==65533:
                    if ord(nx[-1:])==65533:
                        wptok.append(x)
                        i += 2
                    else:
                        wptok.append(x)
                        i += 1
        ### reimplementation of word_to_wordpiece with the modified roberta wordpieces
        w_index = 0
        word_to_wordpiece = []
        subword_sequence = []
        bad_match_flag = 0
        for wp_index, wp in enumerate(wptok):
            if w_index in range(len(word_tokens)):
                word = word_tokens[w_index]
                if word == wptok[wp_index]:
                    word_to_wordpiece.append(wp_index)
                    w_index += 1
                else:
                    subword_sequence.append(wp_index)
                    word_from_pieces = "".join([
                        wptok[i].lstrip()
                        for i in subword_sequence
                    ])
                    if word == word_from_pieces:
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                    elif word_from_pieces not in word:
                        # compare the length instead of strings since there are offending
                        # characters in roberta wordpieces
                        if len(word) == len(word_from_pieces):
                            word_to_wordpiece.append(subword_sequence)
                            w_index += 1
                            subword_sequence = []

        # assert len(word_tokens) == len(word_to_wordpiece)
        if len(word_tokens) != len(word_to_wordpiece):
            print("SENTENCE: ",sentence)
            print("WPTOK: ", wptok, " ", len(wptok))
            print("WORDPIECE: ", wordpiece_tokens, " ", len(wordpiece_tokens))
            print("WORD token count: ",len(word_tokens))
            print("WORD_to_wordpiece: ", word_to_wordpiece, " ", len(word_to_wordpiece))
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
    return  torch.tensor(wp_indices)


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

