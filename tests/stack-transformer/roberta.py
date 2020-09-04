#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import os
import shutil
import torch
from tqdm import tqdm
import argparse
from fairseq.tokenizer import tokenize_line
from fairseq.models.roberta import RobertaModel
from transition_amr_parser.io import read_sentences

def argument_parsing():
    parser = argparse.ArgumentParser(
        description='unit test for roberta unicode handling'
    )
    parser.add_argument(
        '-i', '--in-tokenized-sentences',
        type=str,
        required=True,
        help='File with one __tokenized__ sentence per line'
    )
    parser.add_argument(
        '-p', '--pretrained-embed',
        type=str,
        required=True,
        default="roberta.large",
        help='roberta model to load'
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        help='File to store bad unicode sentences'
    )
    parser.add_argument(
        '--raise-error',
        action='store_true',
        help='Set to force exception if unicode error found'
    )

    return parser.parse_args()

def main():
    args = argument_parsing()

    sentences = read_sentences(args.in_tokenized_sentences)
    split_sentences = []
    for sentence in sentences:
        split_sentences.append(tokenize_line(sentence))
    print(len(split_sentences))

    bad_unicode = open(args.output_file, 'w')

    def load_roberta(name=None, roberta_cache_path=None):
        if not roberta_cache_path:
            roberta = torch.hub.load('pytorch/fairseq', name)
        else:
            roberta = RobertaModel.from_pretrained(roberta_cach_path, checkpoint_file='model.pt')

        roberta.eval()
        if torch.cuda.is_available():
            roberta.cuda()
        return roberta

    def get_wordpiece_to_word_map(sentence, roberta_bpe, raise_error):
        # Get word and worpiece tokens according to RoBERTa                                           
        # sentence = sentence.replace(u'\x91', u' ')
        # sentence = sentence.replace(u'\x96', u' ')
        word_tokens = sentence.split()
        wordpiece_tokens = [
            roberta_bpe.decode(wordpiece)
            for wordpiece in roberta_bpe.encode(sentence).split()
        ]
        #print("wp_tokens: ", wordpiece_tokens)

        assert len(word_tokens) <= len(wordpiece_tokens)
        assert isinstance(word_tokens, list)
        assert isinstance(wordpiece_tokens, list)
        w_index = 0
        word_to_wordpiece = []
        subword_sequence = []
        bad_unicode_flag = 0

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
                    if word == word_from_pieces:
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                    elif word_from_pieces not in word:
                        word_to_wordpiece.append(subword_sequence)
                        w_index += 1
                        subword_sequence = []
                        bad_unicode_flag = 1

        if bad_unicode_flag == 1:
            bad_unicode.write(sentence)
            wp = " ".join(wordpiece_tokens)
            print("\n\nsentence: ", sentence)
            print("wp: ", wp)
            print("\n")
            bad_unicode.write("\n")
            bad_unicode.write(wp)
            bad_unicode.write("\n\n")
            if raise_error:
                raise Exception('Unicode splitting failed')

        return word_to_wordpiece

    def check_wordpiece_to_word_map(input_file, raise_error):
        num_sents = 0
        with open(input_file, 'r') as fid:
            for sentence in tqdm(fid):
                if not sentence:
                    break
                sentence = " ".join(tokenize_line(str(sentence.rstrip())))
                #print("input: ", sentence)
                word2piece = get_wordpiece_to_word_map(
                    sentence,
                    roberta.bpe,
                    raise_error
                )

    roberta = load_roberta(name=args.pretrained_embed)
    check_wordpiece_to_word_map(args.in_tokenized_sentences, args.raise_error)

if __name__ == "__main__":
    main()
