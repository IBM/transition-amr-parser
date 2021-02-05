import copy

import torch
from transformers import BertTokenizer, BertModel

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


def get_wordpiece_to_word_map(sentence, tokenizer):
    """[summary]

    Args:
        sentence (str): untokenized sentence, white space separated
        tokenizer (transformers Tokenizer): Tokenizer from huggingface transformers library

    Returns:
        List[List or int]: word to wordpiece mappings
    """

    # Get word and worpiece tokens according to BERT
    word_tokens = sentence.split()
    wordpiece_tokens = tokenizer.tokenize(sentence)

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
                # NOTE: transformers BERT use ## to represent BPE
                wordpiece_tokens[i][2:] if wordpiece_tokens[i].startswith('##') else wordpiece_tokens[i]
                for i in subword_sequence
            ])
            if word == word_from_pieces:
                word_to_wordpiece.append(subword_sequence)
                w_index += 1
                subword_sequence = []

            try:
                assert word_from_pieces in word, \
                    "wordpiece must be at least a segment of current word"
            except:
                breakpoint()

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
            if name in ['bert-base-cased', 'bert-large-cased', 'bert-base-uncased', 'bert-large-uncased']:

                # Extract
                self.tokenizer = BertTokenizer.from_pretrained(name)
                self.model = BertModel.from_pretrained(name)
                self.model.eval()
                if torch.cuda.is_available():
                    self.model.cuda()
                    print(f'Using {name} extraction in GPU')
                else:
                    print(f'Using {name} extraction in cpu (slow, wont OOM)')

            else:
                raise Exception(
                    f'Unknown --pretrained-embed {name}'
                )
        else:
            self.model = model

    def extract_features(self, encoded_input):
        """Extract features from wordpieces"""

        if self.bert_layers is None:
            # normal BERT
            with torch.no_grad():
                output = self.model(**encoded_input)
            feature = output.last_hidden_state    # size (1, input_len, hidden_size)
            return feature
        else:
            # layer average BERT
            with torch.no_grad():
                output = self.model(**encoded_input, output_hidden_states=True)
            features = output.hidden_states    # tuple of length 25 for bert-large
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
            self.tokenizer
        )

        # NOTE: We need to re-extract BPE inside bert. Token indices
        # will also be different. [CLS] and [SEP] added
        encoded_input = self.tokenizer(sentence_string, return_tensors='pt')
        worpieces_bert = encoded_input['input_ids'].squeeze(0)

        # Extract bert, remove [CLS]/[SEP]
        if torch.cuda.is_available():

            # Hotfix for sequences above 512
            if worpieces_bert.shape[0] > 512:
                excess = worpieces_bert.shape[0] - 512
                # first 512 tokens
                last_layer = self.extract_features(
                    {k: v.to(self.model.device)[:512] for k, v in encoded_input.items()}
                )
                # last 512 tokens
                last_layer2 = self.extract_features(
                    {k: v.to(self.model.device)[excess:] for k, v in encoded_input.items()}
                )
                # concatenate
                shape = (last_layer, last_layer2[:, -excess:, :])
                last_layer = torch.cat(shape, 1)

                assert worpieces_bert.shape[0] == last_layer.shape[1]

                # warn user about this
                string = '\nMAX_POS overflow!! {worpieces_bert.shape[0]}'
                print(yellow_font(string))

            else:

                # Normal extraction
                last_layer = self.extract_features(
                    {k: v.to(self.model.device) for k, v in encoded_input.items()}
                )

        else:

            # Copy code above
            raise NotImplementedError()
            last_layer = self.extract_features(
                encoded_input
            )

        # FIXME: this should not be needed using with torch.no_grad()
        # last_layer = last_layer.detach()

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

        return word_features, worpieces_bert, word2piece

    def extract_batch(self, sentence_string_batch):
        bert_data = {}
        bert_data["word_features"] = []
        bert_data["wordpieces_roberta"] = []
        bert_data["word2piece_scattered_indices"] = []
        src_wordpieces = []
        src_word2piece = []
        for sentence in sentence_string_batch:
            word2piece = get_wordpiece_to_word_map(sentence, self.tokenizer)
            encoded_input = self.tokenizer(sentence, return_tensors='pt')
            wordpieces_roberta = encoded_input['input_ids'].squeeze(0)
            wordpieces_roberta = wordpieces_roberta[:512]
            src_wordpieces.append(copy.deepcopy(wordpieces_roberta))
            src_word2piece.append(copy.deepcopy(word2piece))

        raise NotImplementedError

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
