#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
import os
from collections import defaultdict

import torch

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.scoring import bleu
from fairseq.meters import StopwatchMeter, TimeMeter

from fairseq_ext.utils_import import import_user_module
from fairseq_ext import options
from fairseq_ext.utils import (post_process_action_pointer_prediction,
                               clean_pointer_arcs,
                               post_process_action_pointer_prediction_bartsv)


class Examples():
    def __init__(self, path, results_path, gen_subset, nbest,avoid_indices=None):
        self.examples = []
        self.path = path    # model path
        self.results_path = results_path    # save prefix
        self.gen_subset = gen_subset
        self.nbest = nbest
        self.sample_ids = []
        self.avoid_indices=avoid_indices

    def append(self, example):
        self.examples.append(example)

    def save(self):
        # Get unique sample ids
        sample_ids = sorted(set([x['sample_id'] for x in self.examples]))
        # make sure no id is missing
        #FIXME compensating of temp removal of index 34
        checklist = list(range(max(sample_ids) + 1))
        #if self.avoid_indices is not None and len(self.avoid_indices)>0:
        #    for ai in self.avoid_indices:
        #        checklist.remove(ai)
        assert checklist  == sample_ids

        # Collect example per id
        results = defaultdict(list)
        for example in self.examples:
            results[example['sample_id']].append(example)

        # make sure right number of examples
        assert all(len(x) == self.nbest for x in results.values())

        # Write data
        dirname = os.path.dirname(self.path.split(':')[0])
        if self.results_path:
            file_path = f'{self.results_path}'
        else:
            file_path = f'{dirname}/{self.gen_subset}'
        # Write actions
        for n in range(self.nbest):
            if n > 0:
                dfile_path = f'{file_path}.{n}'
            else:
                dfile_path = file_path
            with open(f'{dfile_path}.actions_nopos', 'w') as fid1, \
                    open(f'{dfile_path}.actions_pos', 'w') as fid2, \
                    open(f'{dfile_path}.actions', 'w') as fid3:
                for sid in sample_ids:
                    fid1.write(
                        "{}\n".format('\t'.join(results[sid][n]['actions_nopos']))
                    )
                    fid2.write(
                        '{}\n'.format('\t'.join(map(str, results[sid][n]['actions_pos'])))
                    )
                    fid3.write(
                        "{}\n".format('\t'.join(results[sid][n]['actions']))
                    )
        # Write source sentences
        with open(f'{file_path}.en', 'w') as fid:
            for sid in sample_ids:
                fid.write("{}\n".format(results[sid][0]['src_str']))


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'


    # otherwise we run into problems with support for Half
    if not torch.cuda.is_available():
        args.fp16 = False

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # ========== for bartsv task, rebuild dictionary after model args are loaded ==========
    # assert not hasattr(args, 'node_freq_min'), 'node_freq_min should be read from model args'
    # args.node_freq_min = 5    # temporarily set before model loading, as this is needed in tasks.setup_task(args)
    # =====================================================================================

    # Load dataset splits
    task = tasks.setup_task(args)
    # Note: states are not needed since they will be provided by the state
    # machine
    task.load_dataset(args.gen_subset, state_machine=False)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    try:
        models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(':'),
            arg_overrides=eval(args.model_overrides),
            task=task,
        )
    except:
        # NOTE this is for "bartsv" models when default "args.node_freq_min" (5) is not equal to the model
        #      when loading model with the above task there will be an error when building the model with the task's
        #      target vocabulary, which would be of different size
        # TODO better handle these cases (without sacrificing compatibility with other model archs)
        models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(':'),
            arg_overrides=eval(args.model_overrides),
            task=None,
        )

    # ========== for bartsv task, rebuild the dictionary based on model args ==========
    if 'bartsv' in _model_args.arch and args.node_freq_min != _model_args.node_freq_min:
        args.node_freq_min = _model_args.node_freq_min
        # Load dataset splits
        task = tasks.setup_task(args)
        # Note: states are not needed since they will be provided by the state machine
        task.load_dataset(args.gen_subset, state_machine=False)

        # Set dictionaries
        try:
            src_dict = getattr(task, 'source_dictionary', None)
        except NotImplementedError:
            src_dict = None
        tgt_dict = task.target_dictionary
    # ==================================================================================

    # import pdb; pdb.set_trace()
    # print(_model_args)

    # ========== for previous model trained when new arguments were not there ==========
    if not hasattr(_model_args, 'shift_pointer_value'):
        _model_args.shift_pointer_value = 1
    # ==================================================================================

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align
    # dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=None,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers
        # avoid_indices=args.avoid_indices
        # avoid_range=args.avoid_range
        # large_sent_first=False        # not in fairseq
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args, _model_args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    examples = Examples(args.path, args.results_path, args.gen_subset, args.nbest,args.avoid_indices)

    error_stats = {'num_sub_start': 0}

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                raise Exception("Did not expect empty sample")
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            # breakpoint()

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, args, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            # breakpoint()

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                # debug: '<<unk>>' is added to the dictionary
                # if 'unk' in target_str:
                #     breakpoint()
                # ==========> NOTE we do not really have the ground truth target (with the same alignments)
                #                  target_str might have <unk> as the target dictionary is only built on training data
                #                  but it doesn't matter. It should not affect the target dictionary!

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    # hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    #     hypo_tokens=hypo['tokens'].int().cpu(),
                    #     src_str=src_str,
                    #     alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    #     align_dict=align_dict,
                    #     tgt_dict=tgt_dict,
                    #     remove_bpe=args.remove_bpe,
                    #     # FIXME: AMR specific
                    #     split_token="\t",
                    #     line_tokenizer=task.tokenize,
                    # )

                    if 'bartsv' in _model_args.arch:
                        if not tgt_dict[hypo['tokens'][0]].startswith(tgt_dict.bpe.INIT):
                            error_stats['num_sub_start'] += 1

                        actions_nopos, actions_pos, actions = post_process_action_pointer_prediction_bartsv(hypo,
                                                                                                                tgt_dict)
                    else:
                        actions_nopos, actions_pos, actions = post_process_action_pointer_prediction(hypo, tgt_dict)

                    # breakpoint()

                    if args.clean_arcs:
                        actions_nopos, actions_pos, actions, invalid_idx = clean_pointer_arcs(actions_nopos,
                                                                                              actions_pos,
                                                                                              actions)

                    # TODO these are just dummy for the reference below to run
                    hypo_tokens = hypo['tokens'].int().cpu()
                    hypo_str = '/t'.join(actions)
                    alignment = None

                    # update the list of examples
                    examples.append({
                        'actions_nopos': actions_nopos,
                        'actions_pos': actions_pos,
                        'actions': actions,
                        'reference': target_str,
                        'src_str': src_str,
                        'sample_id': sample_id
                    })

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo_str, hypo['score']))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=False)
                            # NOTE do not modify the tgt dictionary with 'add_if_not_exist=True'!
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    # Save examples to files
    examples.save()

    print('| Error case (handled by manual fix) statistics:')
    print(error_stats)

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
