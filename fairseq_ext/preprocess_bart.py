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
from fairseq_ext.extract_bart.binarize_encodings import make_bart_encodings


def add_sampling_vocabulary(tgt_dict, trainpref):
    # Fix to avoid unseen symbols whne sampling alignments. Arc directions and
    # what is COPY-ed may change

    # FIXME: ugly and local way
    from transition_amr_parser.io import read_amr
    amr_path = f'{os.path.dirname(trainpref)}/ref_train.amr'
    assert os.path.exists(amr_path)
    for amr in read_amr(amr_path):
        for node_name in amr.nodes.values():
            tgt_dict.add_symbol(node_name.replace('"', ''))

    for action in tgt_dict.symbols:
        # Ensure every right arc has a left arc in case
        # sampling inverses it
        if action.startswith('>LA'):
            tgt_dict.add_symbol('>RA' + action[3:])
        elif action.startswith('>RA'):
            tgt_dict.add_symbol('>LA' + action[3:])


def main(args):
    import_user_module(args)

    print(args)

    # to control what preprocessing needs to be run (as they take both time and storage so we avoid running repeatedly)
    run_basic = True
    # this includes:
    # src: build src dictionary, copy the raw data to dir; build src binary data (need to refactor later if this is not needed)
    # tgt: split target pointer values into a separate file; build tgt dictionary, binarize the actions and pointer values
    run_act_states = True
    # this includes:
    # run the state machine in canonical mode to get states information to facilitate modeling;
    # takes about 1 hour and 13G space
    run_roberta_emb = True
    # this includes:
    # for src sentences, use pre-trained RoBERTa model to extract contextual embeddings for each word;
    # takes about 10min for RoBERTa base and 30 mins for RoBERTa large and 2-3G space;
    # this needs GPU and only needs to run once for the English sentences, which does not change for different oracles;
    # thus the embeddings are stored separately from the oracles.

    if os.path.exists(os.path.join(args.destdir, '.done')):
        print(f'binarized actions and states directory {args.destdir} already exists; not rerunning.')
        run_basic = False
        run_act_states = False
    if os.path.exists(os.path.join(args.embdir, '.done')):
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
        actions_files = [f'{pref}.{args.target_lang}' for pref in [args.trainpref] + args.validpref.split(',') + args.testpref.split(',')]
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
        # if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        #     raise FileExistsError(dict_path(args.source_lang))
        # if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        #     raise FileExistsError(dict_path(args.target_lang))

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
                    if args.task.endswith('dyo'):
                        add_sampling_vocabulary(tgt_dict, args.trainpref)
                    #since there is a common machine for doc and sentence, adding action CLOSE_SENTENCE to tgt vocab
                    tgt_dict.add_symbol('CLOSE_SENTENCE')

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
            if len(args.validpref.split(",")) == 1:
                for k, validpref in enumerate(args.validpref.split(",")):
                    outprefix = "valid{}".format(k) if k > 0 else "valid"
                    make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)
            else:
                for k, validpref in enumerate(args.validpref.split(",")):
                    outprefix = "valid_" + validpref.split("/")[-1].split("_")[-1]
                    make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)
        if args.testpref:
            if len(args.testpref.split(",")) == 1:
                for k, testpref in enumerate(args.testpref.split(",")):
                    outprefix = "test{}".format(k) if k > 0 else "test"
                    make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)
            else:
                for k, testpref in enumerate(args.testpref.split(",")):
                    outprefix = "test_" + testpref.split("/")[-1].split("_")[-1]
                    make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)
                
    # NOTE we do not encode the source sentences with dictionary, as the source embeddings are directly provided
    # from RoBERTa, thus the source dictionary here is of no use
    if run_basic:
        make_all(args.source_lang, src_dict, dataset_impl='raw')
        make_all(args.source_lang, src_dict, dataset_impl='mmap')
        # above: just leave for the sake of model to run without too much change
        # NOTE there are <unk> in valid and test set for target actions
        if target:
            make_all(args.target_lang_nopos, tgt_dict)

        # binarize pointer values and save to file

        # TODO make naming convention clearer
        if len(args.validpref.split(",")) == 1:
            # assume one training file, one validation file, and one test file
            for pos_file, split in [(f'{pref}.actions_pos', split) for pref, split in
                                    [(args.trainpref, 'train'), (args.validpref, 'valid'), (args.testpref, 'test')]]:
                out_pref = os.path.join(args.destdir, split)
                task.binarize_actions_pointer_file(pos_file, out_pref)
        else:
            if args.trainpref:
                out_pref = os.path.join(args.destdir, 'train')
                task.binarize_actions_pointer_file(f'{args.trainpref}.actions_pos', out_pref)
            for (i,validpref) in enumerate(args.validpref.split(",")):
                split = "valid_" + validpref.split("/")[-1].split("_")[-1]
                out_pref = os.path.join(args.destdir, split)
                task.binarize_actions_pointer_file(f'{validpref}.actions_pos', out_pref)
            for (i,testpref) in enumerate(args.testpref.split(",")):
                split = "test_" + testpref.split("/")[-1].split("_")[-1]
                out_pref = os.path.join(args.destdir, split)
                task.binarize_actions_pointer_file(f'{testpref}.actions_pos', out_pref)        

        # for dynamic oracle: copy the gold amr with alignments to the data folder
        if args.task == 'amr_action_pointer_bart_dyo':
            for pref, split in [(args.trainpref, 'train'), (args.validpref, 'valid'), (args.testpref, 'test')]:
                if split == 'valid':
                    split_amr = 'ref_dev.amr'
                else:
                    split_amr = f'ref_{split}.amr'
                shutil.copyfile(
                    os.path.join(os.path.dirname(pref), split_amr),
                    os.path.join(args.destdir, f'{split}.aligned.gold-amr')
                )
                if split == 'train':
                    shutil.copyfile(
                        os.path.join(os.path.dirname(pref), 'alignment.trn.align_dist.npy'),
                        os.path.join(args.destdir, 'alignment.trn.align_dist.npy')
                    )

    # save action states information to assist training with auxiliary info
    # assume one training file, one validation file, and one test file
    if run_act_states:
        task_obj = task(args, tgt_dict=tgt_dict)
        machine_config_file = os.path.join(os.path.dirname(args.trainpref.split(",")[0]), 'machine_config.json')
        if len(args.validpref.split(",")) == 1:
            for prefix, split in zip([args.trainpref, args.validpref, args.testpref], ['train', 'valid', 'test']):
                en_file = prefix + '.en'
                actions_file = prefix + '.actions'
                out_file_pref = os.path.join(args.destdir, split)
                task_obj.build_actions_states_info(en_file, actions_file, machine_config_file, out_file_pref, num_workers=args.workers)
        else:
            #train
            en_file = args.trainpref + '.en'
            actions_file = args.trainpref + '.actions'
            out_file_pref = os.path.join(args.destdir, 'train')
            task_obj.build_actions_states_info(en_file, actions_file, machine_config_file, out_file_pref, num_workers=args.workers)
            #dev/valid
            for (i,validpref) in enumerate(args.validpref.split(",")):
                en_file = validpref + '.en'
                actions_file = validpref + '.actions'
                outprefix = "valid_" + validpref.split("/")[-1].split("_")[-1]
                out_file_pref = os.path.join(args.destdir, outprefix)
                task_obj.build_actions_states_info(en_file, actions_file, machine_config_file, out_file_pref, num_workers=args.workers)
            #test
            for (i,testpref) in enumerate(args.testpref.split(",")):
                en_file = testpref + '.en'
                actions_file = testpref + '.actions'
                outprefix = "test_" + testpref.split("/")[-1].split("_")[-1]
                out_file_pref = os.path.join(args.destdir, outprefix)
                task_obj.build_actions_states_info(en_file, actions_file, machine_config_file, out_file_pref, num_workers=args.workers)
                
        # create empty file flag
        open(os.path.join(args.destdir, '.done'), 'w').close()

    # save RoBERTa embeddings
    # TODO refactor this code
    if run_roberta_emb:
        make_bart_encodings(args, tokenize=tokenize)
        # create empty file flag
        open(os.path.join(args.embdir, '.done'), 'w').close()

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
