#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import os
import shutil
from collections import Counter

import numpy as np
# from fairseq import options, tasks, utils
from fairseq import tasks
# from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
from multiprocessing import Pool
from fairseq.tokenizer import tokenize_line

from fairseq_ext.utils_import import import_user_module
from fairseq_ext.data import indexed_dataset
from fairseq_ext import options
from fairseq_ext.roberta.binarize_embeddings import make_roberta_embeddings


def main(args):
    import_user_module(args)

    print(args)

    # to control what preprocessing needs to be run (as they take both time and storage so we avoid running repeatedly)
    run_basic = True
    # this includes:
    # src: build src dictionary, copy the raw data to dir; build src binary data (need to refactor later if unneeded)
    # tgt: split target non-pointer actions and pointer values into separate files; build tgt dictionary
    run_act_states = True
    # this includes:
    # run the state machine reformer to get
    # a) training data: input and output, pointer values;
    # b) states information to facilitate modeling;
    # takes about 1 hour and 13G space on CCC
    run_roberta_emb = True
    # this includes:
    # for src sentences, use pre-trained RoBERTa model to extract contextual embeddings for each word;
    # takes about 10min for RoBERTa base and 30 mins for RoBERTa large and 2-3G space;
    # this needs GPU and only needs to run once for the English sentences, which does not change for different oracles;
    # thus the embeddings are stored separately from the oracles.

    if os.path.exists(args.destdir):
        print(f'binarized actions and states directory {args.destdir} already exists; not rerunning.')
        run_basic = False
        run_act_states = False
    if os.path.exists(args.embdir):
        print(f'pre-trained embedding directory {args.embdir} already exists; not rerunning.')
        run_roberta_emb = False

    os.makedirs(args.destdir, exist_ok=True)
    os.makedirs(args.embdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    # preprocess target actions files, to split '.actions' to '.actions_nopos' and '.actions_pos'
    # when building dictionary on the target actions sequences
    # split the action file into two files, one without arc pointer and one with only arc pointer values
    # and the dictionary is only built on the no pointer actions
    if run_basic:
        assert args.target_lang == 'actions', 'target extension must be "actions"'
        actions_files = [f'{pref}.{args.target_lang}' for pref in (args.trainpref, args.validpref, args.testpref)]
        task.split_actions_pointer_files(actions_files)
        args.target_lang_nopos = 'actions_nopos'    # only build dictionary without pointer values
        args.target_lang_pos = 'actions_pos'

    # set tokenizer
    tokenize = task.tokenize if hasattr(task, 'tokenize') else tokenize_line

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt

        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
            # tokenize separator is taken care inside task
        )

    # build dictionary and save

    if run_basic:
        if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
            raise FileExistsError(dict_path(args.source_lang))
        if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
            raise FileExistsError(dict_path(args.target_lang))

        if args.joined_dictionary:
            assert not args.srcdict or not args.tgtdict, \
                "cannot use both --srcdict and --tgtdict with --joined-dictionary"

            if args.srcdict:
                src_dict = task.load_dictionary(args.srcdict)
            elif args.tgtdict:
                src_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
                src_dict = build_dictionary(
                    {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
                )
            tgt_dict = src_dict
        else:
            if args.srcdict:
                src_dict = task.load_dictionary(args.srcdict)
            else:
                assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
                src_dict = build_dictionary([train_path(args.source_lang)], src=True)

            if target:
                if args.tgtdict:
                    tgt_dict = task.load_dictionary(args.tgtdict)
                else:
                    assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                    tgt_dict = build_dictionary([train_path(args.target_lang_nopos)], tgt=True)
            else:
                tgt_dict = None

        src_dict.save(dict_path(args.source_lang))
        if target and tgt_dict is not None:
            tgt_dict.save(dict_path(args.target_lang_nopos))

    # save binarized preprocessed files

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        False,    # note here we shut off append eos
                        tokenize
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab), dtype=np.int64)
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1],
                append_eos=False,
                tokenize=tokenize
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, dataset_impl=args.dataset_impl):
        if dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab, dataset_impl=args.dataset_impl):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers, dataset_impl=dataset_impl)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)

    # NOTE we do not encode the source sentences with dictionary, as the source embeddings are directly provided
    # from RoBERTa, thus the source dictionary here is of no use
    if run_basic:
        make_all(args.source_lang, src_dict, dataset_impl='raw')
        make_all(args.source_lang, src_dict, dataset_impl='mmap')
        # above: just leave for the sake of model to run without too much change
        # NOTE there are <unk> in valid and test set for target actions
        # if target:
        #     make_all(args.target_lang_nopos, tgt_dict)

        # NOTE targets (input, output, pointer values) are now all included in the state generation process

        # binarize pointer values and save to file

        # TODO make naming convention clearer
        # assume one training file, one validation file, and one test file
        # for pos_file, split in [(f'{pref}.actions_pos', split) for pref, split in
        #                         [(args.trainpref, 'train'), (args.validpref, 'valid'), (args.testpref, 'test')]]:
        #     out_pref = os.path.join(args.destdir, split)
        #     task.binarize_actions_pointer_file(pos_file, out_pref)

    # save action states information to assist training with auxiliary info
    # assume one training file, one validation file, and one test file
    if run_act_states:
        task_obj = task(args, tgt_dict=tgt_dict)
        for prefix, split in zip([args.trainpref, args.validpref, args.testpref], ['train', 'valid', 'test']):
            en_file = prefix + '.en'
            actions_file = prefix + '.actions'
            out_file_pref = os.path.join(args.destdir, split)
            task_obj.build_actions_states_info(en_file, actions_file, out_file_pref, num_workers=args.workers)

    # save RoBERTa embeddings
    # TODO refactor this code
    if run_roberta_emb:
        make_roberta_embeddings(args, tokenize=tokenize)

    print("| Wrote preprocessed oracle data to {}".format(args.destdir))
    print("| Wrote preprocessed embedding data to {}".format(args.embdir))


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=False, tokenize=tokenize_line):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab), dtype=np.int64)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end, tokenize=tokenize)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
