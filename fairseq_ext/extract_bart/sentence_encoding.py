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

def get_wordpiece_to_word_map_old(sentence, roberta_bpe):
    # Get word and wordpiece tokens according to RoBERTa
    sentence += " en"
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
    bad_unicode_flag = 0
    overrun_sentence_flag = 0
    for wp_index in range(len(wordpiece_tokens)):
        if w_index in range(len(word_tokens)):
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
                tempword = word_from_pieces
                word_from_pieces = word_from_pieces.lstrip()
                if word == word_from_pieces:
                    word_to_wordpiece.append(subword_sequence)
                    w_index += 1
                    subword_sequence = []

                elif word_from_pieces not in word:
                    if wp_index > 0 and tempword[:1]==' ' and wordpiece_tokens[wp_index+1][:1]==' ':
                        #print("word_from_pieces: ", word_from_pieces)
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                    elif wp_index < len(wordpiece_tokens)-1 and  wordpiece_tokens[wp_index+1][:1]==' ':
                        #print("WORD_FROM_PIECES: ", word_from_pieces)
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                    elif wp_index == len(wordpiece_tokens)-1:
                        #print("FINAL: ", word_from_pieces)
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                        

    if len(word_tokens) != len(word_to_wordpiece):
        print("SENTENCE: ",sentence)
        print("WORDPIECE: ", wordpiece_tokens, " ", len(wordpiece_tokens))
        print("WORD token count: ",len(word_tokens))
        print("WORD_to_wordpiece: ", word_to_wordpiece, " ", len(word_to_wordpiece))

    return word_to_wordpiece

def get_wordpiece_to_word_map(sentence, roberta_bpe):
    # append one additional token to account for language id
    #sentence = sentence.replace('\u200b','').replace('\u2581','')
    sentence = sentence.replace('\u200b','')
    sentence += " en"
    # Get word and wordpiece tokens according to GPT2BPE (used by RoBERTa/BART)
    word_tokens = sentence.split()
    #wordpiece_bpe_ids = roberta_bpe.sp.EncodeAsIds(sentence)    # List[int] corresponding to bpe vocab
    wordpiece_bpe_ids = roberta_bpe.sp.encode(sentence)   
    wordpiece_tokens = roberta_bpe.sp.EncodeAsPieces(sentence)
    if(len(wordpiece_bpe_ids) != len(wordpiece_tokens)):
        print("wordpiece_bpe_ids: ", len(wordpiece_bpe_ids), wordpiece_bpe_ids)
        print("wordpiece_tokens: ", len(wordpiece_tokens), wordpiece_tokens)

    #print("wordpiece_tokens: ", wordpiece_tokens)
    #print("wordpiece_bpe_ids: ", wordpiece_bpe_ids)

    assert len(word_tokens) <= len(wordpiece_bpe_ids)
    assert isinstance(word_tokens, list)
    assert isinstance(wordpiece_bpe_ids, list)

    w_index = 0
    word_to_wordpiece = []    # List[List[int]]
    subword_sequence = []
    bpe_id_sequence = []

    #print("wp: ", roberta_bpe.encode(sentence))
    for wp_index, bpe_id in enumerate(wordpiece_bpe_ids):
        word = word_tokens[w_index]
        #print("wp_index: ", wp_index, " bpe_id: ", bpe_id, " word: ", word)

        subword_sequence.append(wp_index)
        bpe_id_sequence.append(bpe_id)
        #word_from_pieces = roberta_bpe.sp.DecodeIds(bpe_id_sequence)
        word_from_pieces = roberta_bpe.sp.decode(bpe_id_sequence)
        if word == word_from_pieces or len(word)==len(word_from_pieces) or len(word_from_pieces) > len(word):
            #print("word_from_pieces: ", word_from_pieces, word)
            #print("word: ", word)
            word_to_wordpiece.append(subword_sequence)
            w_index += 1
            subword_sequence = []
            bpe_id_sequence = []

    assert len(word_tokens) == len(word_to_wordpiece), 'word_to_wordpiece must be of the same size of the word_tokens'
    assert word_to_wordpiece[0][0] == 0 and word_to_wordpiece[-1][-1] == len(wordpiece_bpe_ids) - 1, \
        'word_to_wordpiece mapping must cover all wordpieces, from the beginning towards the end'

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
    def __init__(self, name, model=None):
        # bart model name
        self.name = name
        if model:
            self.model = model
        elif name in ['bart.base', 'bart.large']:
            self.model = torch.hub.load('pytorch/fairseq', name)
        else:
            raise Exception(f'Unknown pretrained model name or path {name}')
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            print(f'Using {name} extraction in GPU')
        else:
            print(f'Using {name} extraction in cpu (slow, wont OOM)')

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
        # print("sentence_string: ", len(sentence_string.split()), " ", sentence_string)
        # print("WORDPIECE_IDS: ", wordpiece_ids)
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
