import torch

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
                subword_sequence = []

            assert word_from_pieces in word, \
                "wordpiece must be at least a segment of current word"

    return word_to_wordpiece


class PretrainedEmbeddings():

    def __init__(self, name, bert_layers):

        # embedding type name
        self.name = name
        # select some layers for averaging
        self.bert_layers = bert_layers

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
