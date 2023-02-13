# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import json
from typing import List, Dict, Tuple
import time

import numpy as np
import torch
from tqdm import tqdm

from fairseq import options, utils, tokenizer
from fairseq.data import (
    data_utils,
    Dictionary
)
from fairseq.tasks import FairseqTask, register_task

from fairseq_ext.data.language_pair_dataset import LanguagePairDataset
from fairseq_ext.data.amr_action_pointer_goldamr_dataset import AMRActionPointerGoldAMRDataset as AMRActionPointerDataset
from fairseq_ext.data.data_utils import load_indexed_dataset
from fairseq_ext.amr_spec.action_info_binarize import (
    ActionStatesBinarizer,
    binarize_actstates_tofile_workers,
    load_actstates_fromfile
)
from fairseq_ext.binarize import binarize_file
from transition_amr_parser.io import read_amr, read_neural_alignments, read_neural_alignments_from_memmap
from transition_amr_parser.amr_machine import AMRStateMachine, AMROracle, peel_pointer
from fairseq_ext.amr_spec.action_info import get_actions_states
from fairseq_ext.data.data_utils import collate_tokens
from fairseq_ext.utils import time_since


def load_amr_action_pointer_dataset(data_path, emb_dir, split, src, tgt, src_dict, tgt_dict, tokenize, dataset_impl,
                                    max_source_positions, max_target_positions, shuffle,
                                    append_eos_to_target,
                                    collate_tgt_states, collate_tgt_states_graph,
                                    src_fix_emb_use):
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
    if src_fix_emb_use:
        src_fixed_embeddings = load_indexed_dataset(embfile_prefix + 'en.bert', None, dataset_impl)
        src_fixed_embeddings_sizes = src_fixed_embeddings.sizes
    else:
        src_fixed_embeddings = None
        src_fixed_embeddings_sizes = None

    # src: wordpieces
    src_wordpieces = load_indexed_dataset(embfile_prefix + 'en.wordpieces', None, dataset_impl)

    # src: wordpieces to word map
    src_wp2w = load_indexed_dataset(embfile_prefix + 'en.wp2w', None, dataset_impl)

    # tgt: actions (encoded by a vocabulary)
    tgt_dataset = load_indexed_dataset(filename_prefix + tgt_nopos, tgt_dict, dataset_impl)

    # tgt: actions pointers
    tgt_pos = load_indexed_dataset(filename_prefix + tgt_pos, tgt_dict, dataset_impl)

    # tgt: actions states information
    try:
        tgt_actstates = load_actstates_fromfile(filename_prefix + tgt, tgt_dict, dataset_impl)
    except:
        assert not collate_tgt_states, ('the target actions states information does not exist --- '
                                        'collate_tgt_states must be 0')

    # gold AMR with alignments (to enable running oracle on the fly)
    aligned_amr_path = os.path.join(data_path, f'{split}.aligned.gold-amr')
    gold_amrs = read_amr(aligned_amr_path)
    # read alignment probabilities
    if split == 'train':
        amr_align_probs_path = os.path.join(data_path, 'alignment.trn.pretty')
        amr_align_probs_npy_path = os.path.join(data_path, 'alignment.trn.align_dist.npy')
        if os.path.exists(amr_align_probs_npy_path):
            corpus_align_probs = read_neural_alignments_from_memmap(amr_align_probs_npy_path, gold_amrs)

        else:
            corpus_align_probs = read_neural_alignments(amr_align_probs_path)
            assert len(gold_amrs) == len(corpus_align_probs), \
                "Different number of AMR and probabilities"
    else:
        corpus_align_probs = None

    # build dataset
    dataset = AMRActionPointerDataset(src_tokens=src_tokens,
                                      src=src_dataset,
                                      src_sizes=src_sizes,
                                      src_dict=src_dict,
                                      src_fix_emb=src_fixed_embeddings,
                                      src_fix_emb_sizes=src_fixed_embeddings_sizes,
                                      src_fix_emb_use=src_fix_emb_use,
                                      src_wordpieces=src_wordpieces,
                                      src_wordpieces_sizes=src_wordpieces.sizes,
                                      src_wp2w=src_wp2w,
                                      src_wp2w_sizes=src_wp2w.sizes,
                                      # tgt
                                      tgt=tgt_dataset,
                                      tgt_sizes=tgt_dataset.sizes,
                                      tgt_dict=tgt_dict,
                                      tgt_pos=tgt_pos,
                                      tgt_pos_sizes=tgt_pos.sizes,
                                      tgt_vocab_masks=tgt_actstates['tgt_vocab_masks'],
                                      tgt_actnode_masks=tgt_actstates['tgt_actnode_masks'],
                                      tgt_src_cursors=tgt_actstates['tgt_src_cursors'],
                                      tgt_actedge_masks=tgt_actstates.get('tgt_actedge_mask', None),
                                      tgt_actedge_cur_nodes=tgt_actstates.get('tgt_actedge_cur_nodes', None),
                                      tgt_actedge_pre_nodes=tgt_actstates.get('tgt_actedge_pre_nodes', None),
                                      tgt_actedge_directions=tgt_actstates.get('tgt_actedge_directions', None),
                                      # gold AMR with alignments (to enable running oracle on the fly)
                                      gold_amrs=gold_amrs,
                                      corpus_align_probs=corpus_align_probs,
                                      # batching
                                      left_pad_source=True,
                                      left_pad_target=False,
                                      max_source_positions=max_source_positions,
                                      max_target_positions=max_target_positions,
                                      shuffle=shuffle,
                                      append_eos_to_target=append_eos_to_target,
                                      collate_tgt_states=collate_tgt_states,
                                      collate_tgt_states_graph=collate_tgt_states_graph,
                                      )
    return dataset


@ register_task('amr_action_pointer_bart_dyo')
class AMRActionPointerBARTDyOracleParsingTask(FairseqTask):
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
        parser.add_argument('--collate-tgt-states-graph', default=0, type=int,
                            help='whether to collate target actions states information for graph (shallow, no message passing)')
        parser.add_argument('--initialize-with-bart', default=1, type=int,
                            help='whether to initialize the model parameters with pretrained BART')
        parser.add_argument('--initialize-with-bart-enc', default=1, type=int,
                            help='whether to initialize the model parameters with pretrained BART encoder')
        parser.add_argument('--initialize-with-bart-dec', default=1, type=int,
                            help='whether to initialize the model parameters with pretrained BART decoder')
        parser.add_argument('--src-fix-emb-use', default=0, type=int,
                            help='whether to use fixed pretrained RoBERTa contextual embeddings for src')
        parser.add_argument('--on-the-fly-oracle', default=1, type=int,
                            help='whether to run oracle on the fly for each batch to get target data')
        parser.add_argument('--on-the-fly-oracle-start-update-num', default=0, type=int,
                            help='Starting number of updates for the first run of on-the-fly oracle')
        parser.add_argument('--on-the-fly-oracle-run-freq', default=1, type=int,
                            help='Number of updates until next run of on-the-fly oracle')
        parser.add_argument('--sample-alignments', default=1, type=int,
                            help='Number of samples from alignments (default=1).')
        parser.add_argument('--alignment-sampling-temp', default=1, type=float,
                            help="Temperature for alignment sampling.")
        parser.add_argument('--rescale-align', action='store_true',
                            help='If true, then rescale loss by number of samples.')
        parser.add_argument('--importance-weighted-align', action='store_true',
                            help='If true, then use importance weighted loss.')
        parser.add_argument('--importance-weighted-temp', default=1, type=float,
                            help='Temperature for importance weights. Higher values give more uniform weighting.')

    def __init__(self, args, src_dict=None, tgt_dict=None, bart=None, machine_config_file=None):
        super().__init__(args)
        self.src_dict = src_dict    # src_dict is not necessary if we use RoBERTa embeddings for source
        self.tgt_dict = tgt_dict
        self.action_state_binarizer = None
        assert self.args.source_lang == 'en' and self.args.target_lang == 'actions'

        # pretrained BART model
        self.bart = bart
        self.bart_dict = bart.task.target_dictionary if bart is not None else None   # src dictionary is the same

        # for dynamic oracle at training time: amr machine and oracle config file
        if machine_config_file is not None:
            self.machine_config = json.load(open(machine_config_file, 'r'))
        else:
            self.machine_config = {'reduce_nodes': None,
                                   'absolute_stack_pos': True}
        self.machine = AMRStateMachine(**self.machine_config)
        self.oracle = AMROracle(machine_config=self.machine.config)

        self.canonical_actions = self.machine.base_action_vocabulary
        self.canonical_act_ids = self.machine.canonical_action_to_dict(self.tgt_dict)

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
        assert args.target_lang == 'actions', 'target extension must be "actions"'
        args.target_lang_nopos = 'actions_nopos'    # only build dictionary without pointer values
        args.target_lang_pos = 'actions_pos'

        # FIXME: This is still not robust
        if hasattr(args, 'save_dir'):
            # standalone mode
            dict_path = args.save_dir
        else:
            # training mode
            dict_path = paths[0]

        src_dict = cls.load_dictionary(os.path.join(dict_path, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(dict_path, 'dict.{}.txt'.format(args.target_lang_nopos)))
        # TODO target dictionary 'actions_nopos' is hard coded now; change it later
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang_nopos, len(tgt_dict)))

        # ========== load the pretrained BART model ==========
        if getattr(args, 'arch', None):
            # training time: pretrained BART needs to be used for initialization
            if 'bart_base' in args.arch:
                print('-' * 10 + ' loading pretrained bart.base model ' + '-' * 10)
                bart = torch.hub.load('pytorch/fairseq', 'bart.base')
            elif 'bart_large' in args.arch:
                print('-' * 10 + 'loading pretrained bart.large model ' + '-' * 10)
                bart = torch.hub.load('pytorch/fairseq', 'bart.large')
            else:
                raise ValueError
        else:
            # inference time: pretrained BART is only used for dictionary related things; size does not matter
            # NOTE size does matter; update this later in model initialization if model is with "bart.large"
            print('-' * 10 + ' (for bpe vocab and embed size at inference time) loading pretrained bart.base model '
                  + '-' * 10)
            bart = torch.hub.load('pytorch/fairseq', 'bart.base')

        bart.eval()    # the pretrained BART model is only for assistance
        # ====================================================

        # for dynamic oracle at training time
        machine_config_file = getattr(args, 'machine_config', None)

        return cls(args, src_dict, tgt_dict, bart, machine_config_file)

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
            if action.startswith('>LA') or action.startswith('>RA'):
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

    def build_actions_states_info(self, en_file, actions_file, machine_config_file, out_file_pref, num_workers=1):
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
            self.action_state_binarizer = ActionStatesBinarizer(self.tgt_dict, machine_config_file)
        res = binarize_actstates_tofile_workers(en_file, actions_file, machine_config_file, out_file_pref,
                                                action_state_binarizer=self.action_state_binarizer,
                                                impl='mmap', tokenize=self.tokenize, num_workers=num_workers)
        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                'actions',
                actions_file + '_nopos',
                res['nseq'],
                res['ntok'],
                100 * res['nunk'] / (res['ntok'] + 1e-6),    # when it is not recorded: denominator being 0
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

        self.datasets[split] = load_amr_action_pointer_dataset(
            data_path, self.args.emb_dir, split,
            src, tgt, self.src_dict, self.tgt_dict,
            self.tokenize,
            dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=True,
            append_eos_to_target=self.args.append_eos_to_target,
            collate_tgt_states=self.args.collate_tgt_states,
            collate_tgt_states_graph=self.args.collate_tgt_states_graph,
            src_fix_emb_use=self.args.src_fix_emb_use
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # TODO this is legacy not used as of now
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_generator(self, args, model_args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            if 'graph' in model_args.arch:
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
                stats_rules=getattr(args, 'machine_rules', None),
                machine_config_file=getattr(args, 'machine_config', None)
            )

    @staticmethod
    def run_oracle(aligned_amr, machine, oracle) -> List[str]:
        """Run oracle for a gold AMR graph with alignments.

        Args:
            aligned_amr ([type]): [description]
            machine ([type]): [description]
            oracle ([type]): [description]

        Returns:
            [type]: [description]
        """
        # spawn new machine for this sentence
        machine.reset(aligned_amr.tokens)

        # initialize new oracle for this AMR
        oracle.reset(aligned_amr)

        # proceed left to right throughout the sentence generating nodes
        while not machine.is_closed:

            # get valid actions
            # valid_actions = machine.get_valid_actions()

            # oracle
            action = oracle.get_action(machine)

            # update machine
            machine.update(action)

        assert machine.action_history[-1] == 'CLOSE'

        # NOTE we include 'CLOSE' here, which will be removed later for all data in "get_sample_data()"
        actions = machine.action_history
        # actions = machine.action_history[:-1]

        return actions

    def run_oracle_batch(self, aligned_amrs) -> List[List[str]]:
        """Run oracle on the fly for each batch.

        Args:
            aligned_amrs ([type]): [description]

        Returns:
            [type]: [description]
        """
        action_sequences = []
        # for amr in tqdm(aligned_amrs, desc='dynamic_oracle'):
        for amr in aligned_amrs:
            # get the action sequence
            actions = self.run_oracle(amr, self.machine, self.oracle)
            action_sequences.append(actions)

        return action_sequences

    def get_sample_data(self, token_sequences, action_sequences) -> List[Dict]:
        """Based on the action sequences returned by dynamic oracle run at the batch level, get the numerical data
        to update the sample. In particular, for the batch ("sample"), we need to at least update:
        - sample['target']
        - sample['tgt_pos']
        - sample['net_input']['tgt_*']    # for any parser states
        - sample['net_input']['prev_output_tokens']
        Here this function extract any needed data to be fed into model based on the oracle actions, and convert all
        the data to tensors to prepare for batch collating and padding.

        Args:
            token_sequences (List[List]): List of tokenized src sentences
            action_sequences (List[List]): List of tokenized tgt actions

        Returns:
            [type]: [description]
        """
        # parser state names mapped to sample attribute names
        names_states2data = {
            'allowed_cano_actions': 'tgt_vocab_masks',
            'token_cursors': 'tgt_src_cursors',
            'actions_nodemask': 'tgt_actnode_masks',
            'actions_nopos_in': 'tgt_in',
            'actions_nopos_out': 'target',
            'actions_pos': 'tgt_pos',
            }
        names_data = [v for v in names_states2data.values()]

        data_samples = []

        # for tokens, actions in tqdm(zip(token_sequences, action_sequences), desc='numericalizing'):
        for tokens, actions in zip(token_sequences, action_sequences):
            # store all needed numerical data for one sentence/amr
            data_piece = dict()

            # ===== separate action with pointer values
            line_actions = [peel_pointer(act) for act in actions]
            actions_nopos, actions_pos = zip(*line_actions)

            # ===== vocab encoded actions (target)
            tgt_tensor = self.tgt_dict.encode_line(
                line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in actions_nopos],
                line_tokenizer=lambda x: x,    # already tokenized
                add_if_not_exist=False,
                consumer=None,
                append_eos=False,
                reverse_order=False
                ).long()    # NOTE default return type here is torch.int32, conversion to long is needed
            data_piece['target'] = tgt_tensor

            # ===== target side pointer values
            data_piece['tgt_pos'] = torch.tensor(actions_pos)

            # ===== parser state information (NOTE CLOSE action at last step is included)
            actions_states = get_actions_states(tokens=tokens, actions=actions,
                                                machine=self.machine)
            # convert state vectors to tensors
            allowed_cano_actions = actions_states['allowed_cano_actions']
            del actions_states['allowed_cano_actions']
            vocab_mask = torch.zeros(len(allowed_cano_actions), len(self.tgt_dict), dtype=torch.uint8)
            for i, act_allowed in enumerate(allowed_cano_actions):
                # vocab_ids_allowed = list(set().union(*[set(canonical_act_ids[act]) for act in act_allowed]))
                # this is a bit faster than above
                vocab_ids_allowed = list(
                    itertools.chain.from_iterable(
                        [self.canonical_act_ids[act] for act in act_allowed]
                    )
                )
                vocab_mask[i][vocab_ids_allowed] = 1
            data_piece['tgt_vocab_masks'] = vocab_mask

            for k, v in actions_states.items():
                data_key = names_states2data[k]
                if 'mask' in k:
                    data_piece[data_key] = torch.tensor(v, dtype=torch.uint8)
                elif 'actions_nopos_in' == k:
                    # input sequence
                    data_piece[data_key] = self.tgt_dict.encode_line(
                        line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in v],
                        line_tokenizer=lambda x: x,    # already tokenized
                        add_if_not_exist=False,
                        consumer=None,
                        append_eos=False,
                        reverse_order=False
                    ).long()    # NOTE default return type here is torch.int32, conversion to long is needed
                elif 'actions_nopos_out' == k:
                    # output sequence
                    data_piece[data_key] = self.tgt_dict.encode_line(
                        line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in v],
                        line_tokenizer=lambda x: x,    # already tokenized
                        add_if_not_exist=False,
                        consumer=None,
                        append_eos=False,
                        reverse_order=False
                    ).long()    # NOTE default return type here is torch.int32, conversion to long is needed
                else:
                    data_piece[data_key] = torch.tensor(v)    # int64

            # shift the target input sequence
            if 'tgt_in' in data_piece:
                data_piece['tgt_in'][1:] = data_piece['tgt_in'][:-1]
                data_piece['tgt_in'][0] = self.tgt_dict.eos()

            # remove the last CLOSE in the action states
            for k, v in data_piece.items():
                if k in names_data:
                    data_piece[k] = v[:-1]

            # append the numerical data for one sentence/amr
            data_samples.append(data_piece)

        return data_samples

    def run_oracle_get_data(self, aligned_amr, align_probs, machine, oracle) -> Tuple[List[str], Dict, Dict]:
        """Run oracle for a gold AMR graph with alignments, get the needed parser states at the same time, and
        numericalize the data into Tensors finally.

        Args:
            aligned_amr ([type]): [description]
            machine ([type]): [description]
            oracle ([type]): [description]

        Returns:
            [type]: [description]
        """
        # spawn new machine for this sentence
        machine.reset(aligned_amr.tokens)

        # initialize new oracle for this AMR
        oracle.reset(aligned_amr, alignment_probs=align_probs,
            alignment_sampling_temp=self.args.alignment_sampling_temp)

        # token indices for nodes, and probabilities
        align_info = oracle.align_info

        # store needed parser states
        allowed_cano_actions = []
        token_cursors = []
        # proceed left to right throughout the sentence generating nodes
        while not machine.is_closed:

            # get valid actions
            act_allowed = machine.get_valid_actions(max_1root=True)
            allowed_cano_actions.append(act_allowed)
            token_cursors.append(machine.tok_cursor)

            # oracle
            action = oracle.get_action(machine)

            # check if valid
            assert machine.get_base_action(
                action) in act_allowed, 'current action not in the allowed space? check the rules.'

            # update machine
            machine.update(action)

        assert machine.action_history[-1] == 'CLOSE'

        # NOTE we include 'CLOSE' here, which will be removed later for all data in "get_sample_data()"
        actions = machine.action_history
        # actions = machine.action_history[:-1]

        # ===== store all needed numerical data for one sentence/amr
        data_piece = dict()

        # ===== separate action with pointer values
        line_actions = [peel_pointer(act) for act in actions]
        actions_nopos, actions_pos = zip(*line_actions)

        # ===== vocab encoded actions (target)
        tgt_tensor = self.tgt_dict.encode_line(
            line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in actions_nopos],
            line_tokenizer=lambda x: x,    # already tokenized
            add_if_not_exist=False,
            consumer=None,
            append_eos=False,
            reverse_order=False
            ).long()    # NOTE default return type here is torch.int32, conversion to long is needed
        data_piece['target'] = tgt_tensor

        # ===== target side pointer values
        data_piece['tgt_pos'] = torch.tensor(actions_pos)

        # ===== parser state information (NOTE CLOSE action at last step is included)
        # parser state names mapped to sample attribute names
        names_states2data = {
            'allowed_cano_actions': 'tgt_vocab_masks',
            'token_cursors': 'tgt_src_cursors',
            'actions_nodemask': 'tgt_actnode_masks',
            'actions_nopos_in': 'tgt_in',
            'actions_nopos_out': 'target',
            'actions_pos': 'tgt_pos',
            }
        names_data = [v for v in names_states2data.values()]

        actions_nodemask = machine.get_actions_nodemask()
        assert len(actions_nodemask) == len(actions)

        actions_states = {'allowed_cano_actions': allowed_cano_actions,
                          'actions_nodemask': actions_nodemask,
                          'token_cursors': token_cursors}

        # convert state vectors to tensors
        if self.args.apply_tgt_vocab_masks:
            allowed_cano_actions = actions_states['allowed_cano_actions']
            del actions_states['allowed_cano_actions']
            vocab_mask = torch.zeros(len(allowed_cano_actions), len(self.tgt_dict), dtype=torch.uint8)
            for i, act_allowed in enumerate(allowed_cano_actions):
                # vocab_ids_allowed = list(set().union(*[set(canonical_act_ids[act]) for act in act_allowed]))
                # this is a bit faster than above
                vocab_ids_allowed = []
                for act in act_allowed:
                    vocab_ids_allowed.extend([j for j in self.canonical_act_ids[act]])
                vocab_mask[i][vocab_ids_allowed] = 1
            data_piece['tgt_vocab_masks'] = vocab_mask

        else:
            data_piece['tgt_vocab_masks'] = None

        for k, v in actions_states.items():
            data_key = names_states2data[k]
            if 'mask' in k:
                data_piece[data_key] = torch.tensor(v, dtype=torch.uint8)
            elif 'actions_nopos_in' == k:
                # input sequence
                data_piece[data_key] = self.tgt_dict.encode_line(
                    line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in v],
                    line_tokenizer=lambda x: x,    # already tokenized
                    add_if_not_exist=False,
                    consumer=None,
                    append_eos=False,
                    reverse_order=False
                ).long()    # NOTE default return type here is torch.int32, conversion to long is needed
            elif 'actions_nopos_out' == k:
                # output sequence
                data_piece[data_key] = self.tgt_dict.encode_line(
                    line=[act if act != 'CLOSE' else self.tgt_dict.eos_word for act in v],
                    line_tokenizer=lambda x: x,    # already tokenized
                    add_if_not_exist=False,
                    consumer=None,
                    append_eos=False,
                    reverse_order=False
                ).long()    # NOTE default return type here is torch.int32, conversion to long is needed

            elif not self.args.apply_tgt_vocab_masks and k == 'allowed_cano_actions':
                data_piece[data_key] = None
            else:
                data_piece[data_key] = torch.tensor(v)    # int64

        # shift the target input sequence
        if 'tgt_in' in data_piece:
            data_piece['tgt_in'][1:] = data_piece['tgt_in'][:-1]
            data_piece['tgt_in'][0] = self.tgt_dict.eos()

        # remove the last CLOSE in the action states
        for k, v in data_piece.items():
            if k in names_data:
                if v is None:
                    data_piece[k] = None
                else:
                    data_piece[k] = v[:-1]

        # add align_info
        if align_info is not None:
            for k, v in align_info.items():
                data_piece['align_info_{}'.format(k)] = v

        return actions, actions_states, data_piece

    def run_oracle_get_data_batch(self, aligned_amrs, align_probs, sample) -> Tuple[List[List[str]], List[Dict], Dict]:
        """Run oracle on the fly and get needed parser states and numerical Tensor data for each batch.

        Args:
            aligned_amrs ([type]): [description]

        Returns:
            [type]: [description]
        """
        action_sequences = []
        data_samples = []

        if self.args.sample_alignments <= 1:
            sample_new = sample

            for index, amr in enumerate(aligned_amrs):
                # get alignment probabilities if available
                aprobs = align_probs[index] if align_probs else None
                # get the action sequence
                actions, actions_states, data_piece = self.run_oracle_get_data(
                    amr, aprobs, self.machine, self.oracle)
                action_sequences.append(actions)
                # append the numerical data for one sentence/amr
                data_samples.append(data_piece)

        else:
            device = sample['id'].device

            def make_new(obj):
                obj_new = {}

                for k, v in obj.items():
                    if k == 'net_input':
                        continue
                    assert not isinstance(v, dict)
                    if isinstance(v, torch.Tensor):
                        shape = list(v.shape)
                        shape[0] = shape[0] * self.args.sample_alignments
                        obj_new[k] = torch.empty(shape, dtype=v.dtype, device=device)
                    elif isinstance(v, (list, tuple)):
                        obj_new[k] = []
                    else:
                        obj_new[k] = v

                return obj_new

            sofar = 0

            def update_new_obj(obj, obj_new, i):
                for k, v in obj.items():
                    if k == 'net_input':
                        continue
                    assert not isinstance(v, dict)
                    if isinstance(v, torch.Tensor):
                        obj_new[k][sofar] = obj[k][i]
                    elif isinstance(v, (list, tuple)):
                        obj_new[k].append(obj[k][i])
                    else:
                        continue

            sample_new = make_new(sample)
            sample_new['net_input'] = make_new(sample['net_input'])

            for index, amr in enumerate(aligned_amrs):
                # get alignment probabilities if available
                aprobs = align_probs[index] if align_probs else None

                for _ in range(self.args.sample_alignments):
                    # get the action sequence
                    actions, actions_states, data_piece = self.run_oracle_get_data(
                        amr, aprobs, self.machine, self.oracle)
                    action_sequences.append(actions)
                    # append the numerical data for one sentence/amr
                    data_samples.append(data_piece)
                    # update sample
                    update_new_obj(sample, sample_new, index)
                    update_new_obj(sample['net_input'], sample_new['net_input'], index)
                    # add alignment lprob
                    if 'lp_align' not in sample_new:
                        sample_new['lp_align'] = []
                    sample_new['lp_align'].append(np.log(data_piece['align_info_p']).sum().item())
                    sofar += 1

        return action_sequences, data_samples, sample_new

    def collate_sample_data(self, data_samples):
        """Collate a batch of data instances, only for the target related data.

        Args:
            data_samples (List[Dict]): A list of data examples.

        Returns:
            [type]: [description]
        """
        # ===== default hyper-parameters
        pad_idx = self.tgt_dict.pad()
        eos_idx = self.tgt_dict.eos()
        left_pad_target = False
        input_feeding = True
        collate_tgt_states = self.args.collate_tgt_states
        collate_tgt_states_graph = self.args.collate_tgt_states_graph

        # ===== collate tensors in the batch
        if len(data_samples) == 0:
            return {}

        # TODO make the functions outside for better code reuse
        def merge(key, left_pad=left_pad_target, move_eos_to_beginning=False, pad_idx=pad_idx, eos_idx=eos_idx):
            return collate_tokens(
                [s[key] for s in data_samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

        def merge_tgt_pos(key, left_pad=left_pad_target, move_eos_to_beginning=False):
            return collate_tokens(
                [s[key] for s in data_samples],
                -2, eos_idx, left_pad, move_eos_to_beginning,
            )

        target = merge('target')
        # # for sanity checks
        # tgt_lengths = torch.LongTensor([len(s['target']) for s in data_samples])
        # tgt_num_tokens = tgt_lengths.sum().item()
        tgt_pos = merge_tgt_pos('tgt_pos')

        if data_samples[0].get('tgt_in', None) is not None:
            # NOTE we do not shift here, as it is already shifter 1 position to the right in `self.get_sample_data`
            tgt_in = merge('tgt_in')
            prev_output_tokens = tgt_in

        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge('target', move_eos_to_beginning=True)

        else:
            raise ValueError

        # TODO write a function to collate 2-D matrices, similar to the collate_tokens function
        def merge_tgt_vocab_masks():
            # default right padding: left_pad_target should be False
            # TODO organize the code here
            masks = [s['tgt_vocab_masks'] for s in data_samples]
            max_len = max([len(m) for m in masks])
            merged = masks[0].new(len(masks), max_len, masks[0].size(1)).fill_(pad_idx)
            for i, v in enumerate(masks):
                merged[i, :v.size(0), :] = v
            return merged

        if collate_tgt_states:
            if data_samples[0]['tgt_vocab_masks'] is not None:
                tgt_vocab_masks = merge_tgt_vocab_masks()
            else:
                tgt_vocab_masks = None
            tgt_actnode_masks = merge('tgt_actnode_masks', pad_idx=0)
            tgt_src_cursors = merge('tgt_src_cursors')
        else:
            tgt_vocab_masks = None
            tgt_actnode_masks = None
            tgt_src_cursors = None

        assert not collate_tgt_states_graph, 'currently not supporting collating graph masks in dynamic batch oracle'

        batch_tgt = {
            'net_input': {
                # AMR actions states
                'tgt_vocab_masks': tgt_vocab_masks,
                'tgt_actnode_masks': tgt_actnode_masks,
                'tgt_src_cursors': tgt_src_cursors,
                # target decoder input
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
            'tgt_pos': tgt_pos
        }

        return batch_tgt

    def update_sample(self, sample_tgt, sample):
        """Update the batched sample as a final step, with the combination of dynamically generated target side data
        and previous fixed sample containing the source side data.

        Args:
            sample_tgt ([type]): [description]
            sample ([type]): [description]

        Returns:
            [type]: [description]
        """
        # ===== move to device
        device = sample['id'].device
        # device = sample['net_input']['src_wordpieces'].device
        if device.type != 'cpu':
            sample_new = utils.move_to_cuda(sample_tgt, device)
        else:
            sample_new = sample_tgt

        # ===== combine with the src data to generate the complete new sample batch
        for k, v in sample.items():
            if k not in ['target', 'tgt_pos', 'net_input', 'gold_amrs']:
                sample_new[k] = v

        for k, v in sample['net_input'].items():
            if k.startswith('src'):
                sample_new['net_input'][k] = v

        return sample_new

    def get_sample_from_oracle(self, sample):
        """Run oracle on the fly for each batched data, and update the batch sample to be used for model training.

        Args:
            sample ([type]): [description]
        """
        if 'gold_amrs' not in sample:
            print('Gold AMR and alignments are not provided in batched data -> do not run dynamic oracle')
            return sample, None

        # sanity check
        for tokens, amr in zip(sample['src_sents'], sample['gold_amrs']):
            # if tokens != amr.tokens:
            #     breakpoint()
            assert tokens == amr.tokens, 'raw src sentence tokens not equal to gold amr src tokens'

        # run oracle
        # start0 = time.time()
        # action_sequences = self.run_oracle_batch(sample['gold_amrs'])
        # print('time for running oracle', time.time() - start0)

        # extract parser states, convert everything to tensors
        # start = time.time()
        # data_samples = self.get_sample_data(sample['src_sents'], action_sequences)
        # print('time for getting data', time.time() - start)
        # print('time for running oracle and getting data', time.time() - start0)

        # run oracle, extract parser states, convert everything to tensors
        # start = time.time()
        action_sequences, data_samples, sample_new = self.run_oracle_get_data_batch(
            sample['gold_amrs'],
            sample.get('align_probs', None),
            sample,
        )
        # Need to update sample if duplicating batch items (e.g. for importance sampling).
        sample = sample_new
        # print('time for running oracle and getting data', time.time() - start)

        # collate tensors into batch, with paddings
        # start = time.time()
        sample_tgt = self.collate_sample_data(data_samples)
        # print('time for collating', time.time() - start)

        # get the new complete batch, move to device
        # start = time.time()
        sample_new = self.update_sample(sample_tgt, sample)
        # print('time for updating', time.time() - start)

        return sample_new, data_samples

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

        # ========== run oracle dynamically for each batched data ==========
        if (self.args.on_the_fly_oracle and 'gold_amrs' in sample
                and update_num >= self.args.on_the_fly_oracle_start_update_num
                and (update_num - self.args.on_the_fly_oracle_start_update_num)
                % self.args.on_the_fly_oracle_run_freq == 0):
            sample_new, data_samples = self.get_sample_from_oracle(sample)

            # breakpoint()
            # sanity check: with no stochasticity, new sample should equal to the fixed previous one
            # diffs = {}
            # for k, v in sample_new.items():
            #     if k != 'net_input' and id(sample_new[k]) != id(sample[k]):
            #         diffs[k] = (abs(v - sample[k])).max().item()
            # for k, v in sample_new['net_input'].items():
            #     if id(sample_new['net_input'][k]) != id(sample['net_input'][k]):
            #         diffs[k] = (abs(v - sample['net_input'][k])).max().item()
            # if max(list(diffs.values())) != 0:
            #     print('new sample differences from fixed one:', diffs)
            #     breakpoint()

        else:
            sample_new = sample

        # breakpoint()
        # substitute with new sample for training
        sample = sample_new
        sample['sample_alignments'] = self.args.sample_alignments
        # ==================================================================

        # NOTE detect_anomaly() would raise an error for fp16 training due to overflow, which is automatically handled
        #      by gradient scaling with fp16 optimizer
        # with torch.autograd.detect_anomaly():
        loss, sample_size, logging_output = criterion(model, sample)
        # if torch.isnan(loss):
        #     import ipdb; ipdb.set_trace(context=30)
        #     loss, sdample_size, logging_output = criterion(model, sample)

        if self.args.rescale_align:
            loss = loss / self.args.sample_alignments

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
