import numpy as np
import torch
import shutil
import time

from ..data import indexed_dataset
from ..utils import time_since


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.embdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


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


def make_binary_bert_features(args, input_prefix, output_prefix, tokenize):

    # Load pretrained embeddings extractor
    if args.pretrained_embed.startswith('roberta'):
        from .sentence_encoding import SentenceEmbeddingRoberta

        pretrained_embeddings = SentenceEmbeddingRoberta(
            args.pretrained_embed,
            args.bert_layers,
            remove_be=False,
            avg_word=False
        )
        no_embeddings = False
    elif args.pretrained_embed.startswith('bert'):
        from ..roberta.pretrained_embeddings_bert import PretrainedEmbeddings

        pretrained_embeddings = PretrainedEmbeddings(
            args.pretrained_embed,
            args.bert_layers
        )
        no_embeddings = False
    elif args.pretrained_embed.startswith('bart'):
        from .sentence_encoding import SentenceEncodingBART

        # NOTE only encode token ids in BART bpe vocabulary, not the pretrained embedding features
        pretrained_embeddings = SentenceEncodingBART(
            args.pretrained_embed
        )
        no_embeddings = True
    else:
        raise ValueError('arg.pretrained_embed should be either roberta.* or bert-*')

    # will store pre-extracted BERT layer
    if not no_embeddings:
        indexed_data = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, 'en.bert', "bin"),
            impl=args.dataset_impl,
            dtype=np.float32
        )

    # will store wordpieces and wordpiece to word mapping
    indexed_wordpieces = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, 'en.wordpieces', "bin"),
        impl=args.dataset_impl,
    )

    indexed_wp2w = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, 'en.wp2w', "bin"),
        impl=args.dataset_impl,
    )

    num_sents = 0
    input_file = input_prefix + '.en'

    start = time.time()
    with open(input_file, 'r') as fid:
        for sentence in fid:

            # we only have tokenized data so we feed whitespace separated
            # tokens
            sentence = " ".join(tokenize(str(sentence).rstrip()))

            # extract embeddings, average them per token and return
            # wordpieces anyway
            if not no_embeddings:
                word_features, wordpieces_roberta, word2piece = \
                    pretrained_embeddings.extract(sentence)
            else:
                wordpieces_roberta, word2piece = pretrained_embeddings.encode_sentence(sentence)

            # note that data needs to be stored as a 1d array. Also check
            # that number nof woprds matches with embedding size
            if not no_embeddings:
                assert word_features.shape[1] == len(wordpieces_roberta)    # not average to words and keep BOS/EOS
                # assert word_features.shape[1] == len(sentence.split())    # average to words and remove BOS/EOS
                indexed_data.add_item(word_features.cpu().view(-1))

            # just store the `wordpieces_roberta` indices, including BOS/EOS tokens
            # `word2piece` excluding BOS/EOS tokens
            indexed_wordpieces.add_item(wordpieces_roberta)
            indexed_wp2w.add_item(
                get_scatter_indices(word2piece, reverse=True)
            )

            # udpate number of sents
            num_sents += 1
            if not num_sents % 100:
                print("\r%d sentences (time: %s)" % (num_sents, time_since(start)), end='')
        print("")

    # close indexed data files
    if not no_embeddings:
        indexed_data.finalize(
            dataset_dest_file(args, output_prefix, 'en.bert', "idx")
        )

    indexed_wordpieces.finalize(
        dataset_dest_file(args, output_prefix, 'en.wordpieces', "idx")
    )
    indexed_wp2w.finalize(
        dataset_dest_file(args, output_prefix, 'en.wp2w', "idx")
    )

    # copy the source sentence file to go together with the embeddings
    shutil.copyfile(input_file, dataset_dest_prefix(args, output_prefix, 'en'))


def make_bart_encodings(args, tokenize=None):
    '''
    Makes BART sentence bpe encodings for source words
    '''

    assert tokenize

    if args.trainpref:
        make_binary_bert_features(args, args.trainpref, "train", tokenize)

    if args.validpref:
        if len(args.validpref.split(",")) == 1:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_binary_bert_features(args, validpref, outprefix, tokenize)
        else:
            for k, validpref in enumerate(args.validpref.split(",")):
                #outprefix = "valid_{}".format(k) if k >= 0 else "valid"
                outprefix = "valid_" + validpref.split("/")[-1].split("_")[-1]
                make_binary_bert_features(args, validpref, outprefix, tokenize)

    if args.testpref:
        if len(args.testpref.split(",")) == 1:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_binary_bert_features(args, testpref, outprefix, tokenize)
        else:
            for k, testpref in enumerate(args.testpref.split(",")):
                #outprefix = "test_{}".format(k) if k >= 0 else "test"
                outprefix = "test_" + testpref.split("/")[-1].split("_")[-1]
                make_binary_bert_features(args, testpref, outprefix, tokenize)
            
