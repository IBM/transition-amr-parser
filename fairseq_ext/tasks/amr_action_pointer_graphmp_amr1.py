# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import numpy as np
import torch

from fairseq import options, utils, tokenizer
from fairseq.data import (
    data_utils,
    Dictionary
)
from fairseq.tasks import FairseqTask, register_task

from fairseq_ext.data.language_pair_dataset import LanguagePairDataset
from fairseq_ext.data.amr_action_pointer_graphmp_dataset import AMRActionPointerGraphMPDataset
from fairseq_ext.data.data_utils import load_indexed_dataset
from fairseq_ext.amr_spec.action_info_binarize_graphmp_amr1 import (
    ActionStatesBinarizer,
    binarize_actstates_tofile_workers,
    load_actstates_fromfile
)
from fairseq_ext.binarize import binarize_file


def load_amr_action_pointer_dataset(data_path, emb_dir, split, src, tgt, src_dict, tgt_dict, tokenize, dataset_impl,
                                    max_source_positions, max_target_positions, shuffle,
                                    append_eos_to_target, collate_tgt_states):
    src_tokens = None
    src_dataset = None
    src_fixed_embeddings = None
    src_wordpieces = None
    src_wp2w = None
    tgt_dataset = None
    tgt_pos = None
    # target action states
    tgt_vocab_masks = None
    tgt_actnode_masks = None
    tgt_src_cursors = None
    # graph structure
    tgt_actedge_masks = None
    tgt_actedge_cur_nodes = None
    tgt_actedge_pre_nodes = None
    tgt_actedge_directions = None

    assert src == 'en'
    assert tgt == 'actions'
    tgt_nopos = tgt + '_nopos'
    tgt_pos = tgt + '_pos'

    filename_prefix = os.path.join(data_path, f'{split}.{src}-{tgt}.')
    embfile_prefix = os.path.join(emb_dir, f'{split}.{src}-{tgt}.')

    # src: en tokens
    with open(embfile_prefix + src, 'r', encoding='utf-8') as f:
        src_tokens = []
        src_sizes = []
        for line in f:
            if line.strip():
                line = tokenize(line)
                src_tokens.append(line)
                src_sizes.append(len(line))

    # src: numerical values encoded by a dictionary, although not directly used if we use RoBERTa embeddings
    # NOTE this is still used to get padding masks; maybe refactor later
    src_dataset = load_indexed_dataset(filename_prefix + src, src_dict, dataset_impl='mmap')
    # NOTE if not specifying dataset_impl and the raw file also exists, then 'raw' will take precedence when
    #      dataset_impl is None

    # src: pre-trained embeddings
    src_fixed_embeddings = load_indexed_dataset(embfile_prefix + 'en.bert', None, dataset_impl)

    # src: wordpieces
    src_wordpieces = load_indexed_dataset(embfile_prefix + 'en.wordpieces', None, dataset_impl)

    # src: wordpieces to word map
    src_wp2w = load_indexed_dataset(embfile_prefix + 'en.wp2w', None, dataset_impl)

    # tgt: actions (encoded by a vocabulary)
    # tgt_dataset = load_indexed_dataset(filename_prefix + tgt_nopos, tgt_dict, dataset_impl)

    # tgt: actions pointers
    # tgt_pos = load_indexed_dataset(filename_prefix + tgt_pos, tgt_dict, dataset_impl)

    # tgt: actions states information
    try:
        tgt_actstates = load_actstates_fromfile(filename_prefix + tgt, tgt_dict, dataset_impl)
    except:
        assert not collate_tgt_states, ('the target actions states information does not exist --- '
                                        'collate_tgt_states must be 0')

    # build dataset
    dataset = AMRActionPointerGraphMPDataset(src_tokens=src_tokens,
                                             src=src_dataset,
                                             src_sizes=src_sizes,
                                             src_dict=src_dict,
                                             src_fix_emb=src_fixed_embeddings,
                                             src_fix_emb_sizes=src_fixed_embeddings.sizes,
                                             src_wordpieces=src_wordpieces,
                                             src_wordpieces_sizes=src_wordpieces.sizes,
                                             src_wp2w=src_wp2w,
                                             src_wp2w_sizes=src_wp2w.sizes,
                                             # tgt
                                             tgt=tgt_actstates['tgt_nopos_out'],
                                             tgt_sizes=tgt_actstates['tgt_nopos_out'].sizes,
                                             tgt_in=tgt_actstates['tgt_nopos_in'],
                                             tgt_in_sizes=tgt_actstates['tgt_nopos_in'].sizes,
                                             tgt_dict=tgt_dict,
                                             tgt_pos=tgt_actstates['tgt_pos'],
                                             tgt_pos_sizes=tgt_actstates['tgt_pos'].sizes,
                                             tgt_vocab_masks=tgt_actstates['tgt_vocab_masks'],
                                             tgt_actnode_masks=tgt_actstates['tgt_actnode_masks'],
                                             tgt_src_cursors=tgt_actstates['tgt_src_cursors'],
                                             tgt_actedge_masks=tgt_actstates['tgt_actedge_masks'],
                                             tgt_actedge_1stnode_masks=tgt_actstates['tgt_actedge_1stnode_masks'],
                                             tgt_actedge_indexes=tgt_actstates['tgt_actedge_indexes'],
                                             tgt_actedge_cur_node_indexes=tgt_actstates['tgt_actedge_cur_node_indexes'],
                                             tgt_actedge_cur_1stnode_indexes=tgt_actstates['tgt_actedge_cur_'
                                                                                           '1stnode_indexes'],
                                             tgt_actedge_pre_node_indexes=tgt_actstates['tgt_actedge_pre_node_indexes'],
                                             tgt_actedge_directions=tgt_actstates['tgt_actedge_directions'],
                                             tgt_actedge_allpre_indexes=tgt_actstates['tgt_actedge_allpre_indexes'],
                                             tgt_actedge_allpre_pre_node_indexes=tgt_actstates['tgt_actedge_allpre_'
                                                                                               'pre_node_indexes'],
                                             tgt_actedge_allpre_directions=tgt_actstates['tgt_actedge_allpre_'
                                                                                         'directions'],
                                             # batching
                                             left_pad_source=True,
                                             left_pad_target=False,
                                             max_source_positions=max_source_positions,
                                             max_target_positions=max_target_positions,
                                             shuffle=shuffle,
                                             append_eos_to_target=append_eos_to_target,
                                             collate_tgt_states=collate_tgt_states
                                             )
    return dataset


@ register_task('amr_action_pointer_graphmp_amr1')
class AMRActionPointerGraphMPParsingTaskAMR1(FairseqTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    word_sep = '\t'

    @ staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--emb-dir', type=str, help='pre-trained src embeddings directory')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on
        # customized additional arguments
        parser.add_argument('--append-eos-to-target', default=0, type=int,
                            help='whether to append eos to target')
        parser.add_argument('--collate-tgt-states', default=1, type=int,
                            help='whether to collate target actions states information')

    def __init__(self, args, src_dict=None, tgt_dict=None):
        super().__init__(args)
        self.src_dict = src_dict    # src_dict is not necessary if we use RoBERTa embeddings for source
        self.tgt_dict = tgt_dict
        self.action_state_binarizer = None
        assert self.args.source_lang == 'en' and self.args.target_lang == 'actions'

    @ classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        assert args.target_lang == 'actions', 'target extension must be "actions"'
        args.target_lang_nopos = 'actions_nopos'    # only build dictionary without pointer values
        args.target_lang_pos = 'actions_pos'
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang_nopos)))
        # TODO target dictionary 'actions_nopos' is hard coded now; change it later
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang_nopos, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @ classmethod
    def tokenize(cls, line):
        line = line.strip()
        return line.split(cls.word_sep)

    @ classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8,
                         tokenize=tokenizer.tokenize_line):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if hasattr(cls, 'tokenize'):
            tokenize = cls.tokenize
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenize, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @ classmethod
    def split_actions_pointer(cls, actions_filename):
        """Split actions file into two files, one without the arc pointers and one only pointer values.

        Args:
            actions_filename (str): actions file name.
        """
        actions_filename = os.path.abspath(actions_filename)
        actions_ext = os.path.splitext(actions_filename)[1]
        assert actions_ext == '.actions', 'AMR actions file name must end with ".actions"'

        def peel_pointer(action, pad=-1):
            if action.startswith('LA') or action.startswith('RA'):
                action, properties = action.split('(')
                properties = properties[:-1]    # remove the ')' at last position
                properties = properties.split(',')    # split to pointer value and label
                pos = int(properties[0].strip())
                label = properties[1].strip()    # remove any leading and trailing white spaces
                action_label = action + '(' + label + ')'
                return (action_label, pos)
            else:
                return (action, pad)

        with open(actions_filename, 'r') as f, \
                open(actions_filename + '_nopos', 'w') as g, open(actions_filename + '_pos', 'w') as h:
            for line in f:
                if line.strip():
                    line_actions = line.strip().split('\t')
                    line_actions = [peel_pointer(act) for act in line_actions]
                    line_actions_nopos, line_actions_pos = zip(*line_actions)
                    g.write(cls.word_sep.join(line_actions_nopos))
                    g.write('\n')
                    h.write(cls.word_sep.join(map(str, line_actions_pos)))
                    h.write('\n')
                else:
                    pass

    @classmethod
    def split_actions_pointer_files(cls, actions_filenames):
        for actions_filename in actions_filenames:
            cls.split_actions_pointer(actions_filename)

    @classmethod
    def binarize_actions_pointer_file(cls, pos_file, out_file_pref):
        """Save the action pointer values in a file to binary values in mmap format."""
        out_file_pref = out_file_pref + '.en-actions.actions_pos'
        binarize_file(pos_file, out_file_pref, impl='mmap', dtype=np.int64, tokenize=cls.tokenize)

    def build_actions_states_info(self, en_file, actions_file, out_file_pref, num_workers=1):
        """Preprocess to get the actions states information and save to binary files.

        Args:
            en_file (str): English sentence file path.
            actions_file (str): actions file path.
            out_file_pref (str): output file prefix.
            num_workers (int, optional): number of workers for multiprocessing. Defaults to 1.
        """
        out_file_pref = out_file_pref + '.en-actions.actions'
        if self.action_state_binarizer is None:
            # for reuse (e.g. train/valid/test data preprocessing) to avoid building the canonical action to
            # dictionary id mapping repeatedly
            self.action_state_binarizer = ActionStatesBinarizer(self.tgt_dict)
        res = binarize_actstates_tofile_workers(en_file, actions_file, out_file_pref,
                                                action_state_binarizer=self.action_state_binarizer,
                                                impl='mmap', tokenize=self.tokenize, num_workers=num_workers)
        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                'actions',
                actions_file + '_nopos',
                res['nseq'],
                res['ntok'],
                100 * res['nunk'] / res['ntok'],
                self.tgt_dict.unk_word,
            )
        )

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_amr_action_pointer_dataset(data_path, self.args.emb_dir, split,
                                                               src, tgt, self.src_dict, self.tgt_dict,
                                                               self.tokenize,
                                                               dataset_impl=self.args.dataset_impl,
                                                               max_source_positions=self.args.max_source_positions,
                                                               max_target_positions=self.args.max_target_positions,
                                                               shuffle=True,
                                                               append_eos_to_target=self.args.append_eos_to_target,
                                                               collate_tgt_states=self.args.collate_tgt_states
                                                               )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # TODO this is legacy not used as of now
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_generator(self, args, model_args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            if 'graphmp' in model_args.arch:
                from fairseq_ext.sequence_generator_graphmp import SequenceGenerator
            elif 'graph' in model_args.arch:
                from fairseq_ext.sequence_generator_graph import SequenceGenerator
            else:
                from fairseq_ext.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                stop_early=(not getattr(args, 'no_early_stop', False)),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                shift_pointer_value=getattr(model_args, 'shift_pointer_value', 0),
                stats_rules=getattr(args, 'machine_rules', None)
            )

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        # import pdb; pdb.set_trace()

        # NOTE detect_anomaly() would raise an error for fp16 training due to overflow, which is automatically handled
        #      by gradient scaling with fp16 optimizer
        # with torch.autograd.detect_anomaly():
        loss, sample_size, logging_output = criterion(model, sample)
        # if torch.isnan(loss):
        #     import ipdb; ipdb.set_trace(context=30)
        #     loss, sdample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        # import ipdb; ipdb.set_trace(context=10)

        # # NaN weigths
        # nan_weights = {
        #     name: param
        #     for name, param in model.named_parameters()
        #     if torch.isnan(param).any()
        # }
        # if nan_weights:
        #     import ipdb; ipdb.set_trace(context=30)
        #     print()

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # import pdb; pdb.set_trace()

        # # for debugging: check the target vocab mask
        # # NOTE for validation and testing, <unk> will cause inf loss when apply the target vocab mask!!!
        # true_tgt_mask = sample['net_input']['tgt_vocab_masks'].view(-1, 9000).gather(
        #     dim=-1,
        #     index=sample['target'].view(-1, 1)
        #     )
        # true_mask_sum = true_tgt_mask.sum()[sample['target'].view(-1, 1).ne(self.tgt_dict.pad())].sum()
        # non_pad_num = sample['target'].ne(self.tgt_dict.pad()).sum()
        # if tru_mask_sum != non_pad_num:
        #     import pdb; pdb.set_trace()

        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, args, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens,
                                      run_amr_sm=args.run_amr_sm,
                                      modify_arcact_score=args.modify_arcact_score,
                                      use_pred_rules=args.use_pred_rules)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @ property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @ property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
