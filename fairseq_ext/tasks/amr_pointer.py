# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import torch

from fairseq import options, utils, tokenizer
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    Dictionary
)

from fairseq.tasks import FairseqTask, register_task
from fairseq_ext.data.language_pair_dataset import LanguagePairDataset
from fairseq_ext.data.data_utils import load_indexed_dataset


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    state_machine=True
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    src_fixed_embeddings = []
    src_wordpieces = []
    src_wp2w = []
    tgt_datasets = []
    tgt_pos = []
    # memory_datasets = []
    # memory_pos_datasets = []
    # target_mask_datasets = []
    # active_logits_datasets = []

    for k in itertools.count():
        # k = 0, 1, 2, 3, etc.
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                # when k == 0 and no data exists
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        # source
        src_file = (prefix + src, src_dict, dataset_impl)
        src_datasets.append(load_indexed_dataset(*src_file))

        # pre-trained embeddings
        fixed_embeddings_file = (prefix + 'en.bert', None, dataset_impl)
        src_fixed_embeddings.append(
            load_indexed_dataset(*fixed_embeddings_file)
        )

        # wordpieces
        wordpieces_file = (prefix + 'en.wordpieces', None, dataset_impl)
        src_wordpieces.append(load_indexed_dataset(*wordpieces_file))

        # wordpieces to word map
        wp2w_file = (prefix + 'en.wp2w', None, dataset_impl)
        src_wp2w.append(load_indexed_dataset(*wp2w_file))

        # actions
        tgt_file = prefix + tgt + '_nopos', tgt_dict, dataset_impl
        tgt_datasets.append(load_indexed_dataset(*tgt_file))

        # actions pointers
        tgt_pos_file = prefix + tgt + '_pos', None, dataset_impl
        tgt_pos.append(load_indexed_dataset(*tgt_pos_file))

        # # state machine states (buffer/stack) and positions
        # memory_file = prefix + 'memory', None, dataset_impl
        # memory_datasets.append(load_indexed_dataset(*memory_file))
        # memory_pos_file = prefix + 'memory_pos', None, dataset_impl
        # memory_pos_datasets.append(load_indexed_dataset(*memory_pos_file))

        # # logit masks
        # target_mask_file = prefix + 'target_masks', None, dataset_impl
        # target_mask_datasets.append(load_indexed_dataset(*target_mask_file))

        # # active logits
        # active_logits_file = prefix + 'active_logits', None, dataset_impl
        # active_logits_datasets.append(load_indexed_dataset(*active_logits_file))

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        # TODO combine is only used here; when False, the iteration when k > 0 is not used anyway
        # what's the logic here
        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        src_fixed_embeddings = src_fixed_embeddings[0]
        src_wordpieces = src_wordpieces[0]
        src_wp2w = src_wp2w[0]
        tgt_pos = tgt_pos[0]
        # memory_dataset = memory_datasets[0]
        # memory_pos_dataset = memory_pos_datasets[0]
        # target_mask_datasets = target_mask_datasets[0]
        # active_logits_datasets = active_logits_datasets[0]
    else:
        # not implemented for stack-transformer
        raise NotImplementedError()
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        src_fixed_embeddings, src_fixed_embeddings.sizes,
        src_wordpieces, src_wordpieces.sizes,
        src_wp2w, src_wp2w.sizes,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        tgt_pos, tgt_pos.sizes,
        # memory_dataset, memory_dataset.sizes,
        # memory_pos_dataset, memory_pos_dataset.sizes,
        # target_mask_datasets, target_mask_datasets.sizes,
        # active_logits_datasets, active_logits_datasets.sizes,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        state_machine=state_machine
    )


@register_task('amr_pointer')
class AMRPointerParsingTask(FairseqTask):
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

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
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

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
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
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang + '_nopos')))
        # TODO target dictionary 'actions_nopos' is hard coded now; change it later
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang + '_nopos', len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def tokenize(cls, line):
        line = line.strip()
        return line.split(cls.word_sep)

    @classmethod
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

    @classmethod
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

    def load_dataset(self, split, epoch=0, combine=False, state_machine=True, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            state_machine=state_machine
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
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
            )

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
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
        with torch.autograd.detect_anomaly():
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

    def inference_step(self, generator, models, sample, prefix_tokens=None, run_amr_sm=True, modify_arcact_score=True):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens,
                                      run_amr_sm=run_amr_sm,
                                      modify_arcact_score=modify_arcact_score)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
