# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from copy import deepcopy
import json
import os
import re
import warnings
from ipdb import set_trace
from collections import defaultdict, Counter

import torch
from packaging import version

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder

from transition_amr_parser.amr_machine import AMRStateMachine
from fairseq_ext.amr_reform.o10_action_reformer_subtok import AMRActionReformerSubtok


BOOL_TENSOR_TYPE = torch.bool if version.parse(torch.__version__) >= version.parse('1.2.0') else torch.uint8


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        stop_early=True,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        shift_pointer_value=0,
        stats_rules=None,
        machine_config_file=None
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            stop_early (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_topp (float, optional): only sample among the smallest set
                of words whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.shift_pointer_value = shift_pointer_value

        if stats_rules is not None and os.path.exists(stats_rules):    # NOTE this is not used for now
            self.stats_rules = json.load(open(stats_rules, 'r'))
            self.pred_rules = self.stats_rules['possible_predicates']
        else:
            self.stats_rules = None
            self.pred_rules = None

        if machine_config_file is not None:
            self.machine_config = json.load(open(machine_config_file, 'r'))
        else:
            self.machine_config = {'reduce_nodes': None,
                                   'absolute_stack_pos': True}

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        run_amr_sm=True,
        modify_arcact_score=True,
        use_pred_rules=False,
        **kwargs
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            run_amr_sm (bool): whether to run AMR state machine to restrict the next allowable actions.
        """
        model = EnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            # max_len = min(
            #     int(self.max_len_a * src_len + self.max_len_b),
            #     # exclude the EOS marker
            #     model.max_decoder_positions() - 1,    # model.max_decoder_positions() is 1024 by default
            # )
            max_len = min(src_len * 5,    # the max ratio for train, dev and test is around 3
                          # exclude the EOS marker
                          model.max_decoder_positions() - 1)
            # model.max_decoder_positions() is 1024 by default; it also limits the max of model's positional embeddings

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        # "new_order": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] if bsz is 5 and beam_size is 3
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = bos_token or self.eos
        attn, attn_buf = None, None
        attn_tgt = None
        tgt_pointers, scores_tgt_pointers = None, None
        nonpad_idxs = None
        nonpad_idxs_tgt = None
        if prefix_tokens is not None:
            partial_prefix_mask_buf = torch.zeros_like(src_lengths).byte()

        # initialize AMR state machine
        if run_amr_sm:
            # if use_pred_rules:
            #     amr_state_machines = [
            #         AMRStateMachine(tokseq_len=src_lengths[i].item(),
            #                         tokens=sample['src_sents'][i],
            #                         canonical_mode=True)
            #         for i in new_order
            #         ]    # length should be bsz * beam_size
            # else:
            #     amr_state_machines = [
            #         AMRStateMachine(tokseq_len=length.item(),
            #                         canonical_mode=True)
            #         for length in src_lengths[new_order]
            #         ]    # length should be bsz * beam_size

            amr_state_machines = []
            # length should be bsz * beam_size
            for i in new_order:
                sm = AMRActionReformerSubtok(dictionary=self.tgt_dict,
                                             machine_config=self.machine_config,
                                             restrict_subtoken=True)
                if 'gold_amr' in sample:
                    # align mode
                    sm.reset(tokens=sample['src_sents'][i],
                             gold_amr=sample['gold_amr'][i])
                else:
                    # normal mode
                    sm.reset(tokens=sample['src_sents'][i])

                amr_state_machines.append(sm)

            canonical_act_ids = amr_state_machines[0].machine_sub.canonical_action_to_dict(self.tgt_dict)
        else:
            amr_state_machines = None
            canonical_act_ids = None

        # setup for modify the arc action scores based on pointer scores
        if modify_arcact_score:
            if canonical_act_ids is None:
                canonical_act_ids = amr_state_machines[0].machine_sub.canonical_action_to_dict(self.tgt_dict)
            # coefficient for the loss
            coef = 1

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS
        # NOTE this could only happen when beam_size == 1,2, or at the first step, where EOS takes a place so that the
        # remaining number of beams would < beam_size (confirm)

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.

            Args:
                sent (int): sentence id as in the original batch (fixed)
                step (int): search step number
                unfin_idx (int): sentence id as in the current batch (dynamic, as the current batch becomes smaller)
                unfinalized_scores (torch.Tensor): candidate scores, size (current_batch_size, 2 * beam_size)
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[unfin_idx].max()
                if self.normalize_scores:
                    best_unfinalized_score /= max_len ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True

            # ========== contrained beam search: we can end for a sentence without reaching beam_size ==========
            # a new condition for contraint beam search: if all the unfinished beams are disallowed
            # we finish no matter beam_size is reached or not
            # NOTE this needs to be coupled with "unfinalized_scores" removing the just finalized <eos> beams by
            #      setting their scores to -math.inf
            if unfinalized_scores[unfin_idx].max() == -math.inf:
                return True
            # ==================================================================================================

            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, state_machines, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses

            Returns:
                A list of sentence indices what are finished.
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # ========== get the beam info upto the current step ==========
            attn_tgt_clone = {s: attn_tgt[s].index_select(0, bbsz_idx) for s in range(1, step + 2)} \
                if attn_tgt is not None else None
            nonpad_idxs_tgt_clone = {s: nonpad_idxs_tgt[s].index_select(0, bbsz_idx) for s in range(1, step + 2)} \
                if nonpad_idxs_tgt is not None else None
            tgt_pointers_clone = tgt_pointers.index_select(0, bbsz_idx)
            tgt_pointers_clone = tgt_pointers_clone[:, 1:step + 2]    # skip the first index, EOS
            scores_tgt_pointers_clone = scores_tgt_pointers.index_select(0, bbsz_idx)[:, :step + 1]
            # =============================================================

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size    # the current sentence id in the reduced batch
                sent = unfin_idx + cum_unfin[unfin_idx]
                # if "finished" is [False, True, True, False, False] (supposing there are 5 sentences in the batch)
                # then "cum_ufin" will be [0, 2, 2]
                # NOTE this will recover "sent" as the original sentence id in the batch, as the current number of
                # sentences may be smaller due to finished sentences have been moved out

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:  # TODO should this be "sent"?
                    score = -math.inf

                def get_hypo(state_machine):

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    # ========== take out the information of the current (single) beam and return ==========
                    if attn_tgt_clone is not None:
                        # remove padding tokens from attn_tgt scores
                        hypo_attn_tgt = {s: attn_tgt_clone[s][i][nonpad_idxs_tgt_clone[s][i]]
                                         for s in range(1, step + 2)}
                        alignment_tgt = [hypo_attn_tgt[s].max(dim=0)[1] for s in range(1, step + 2)]
                        # .view(-1) to avoid zero-dimensional tensor which cannot be concatenated
                        alignment_tgt = torch.cat([a.view(-1) for a in alignment_tgt])
                    else:
                        hypo_attn_tgt = None
                        alignment_tgt = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                        'attention_tgt': hypo_attn_tgt,
                        'alignment_tgt': alignment_tgt,
                        'pointer_tgt': tgt_pointers_clone[i],
                        'pointer_scores': scores_tgt_pointers_clone[i],
                        'state_machine': state_machine
                    }
                    # ===========================================================

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo(state_machines[idx]))
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo(state_machines[idx])

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)

            return newly_finished

        reorder_state = None
        batch_idxs = None
        # mask for valid beams after search selection: size (bsz * beam_size, )
        # valid_bbsz_mask = tokens.new_ones(bsz * beam_size, dtype=torch.uint8)
        # for pytorch >= 1.2, bool is encouraged to mask index
        valid_bbsz_mask = tokens.new_ones(bsz * beam_size, dtype=BOOL_TENSOR_TYPE)
        valid_bbsz_idx = valid_bbsz_mask.nonzero().squeeze(-1)
        # index mapping from full bsz * beam_size vector to the valid-only vector with reduced size
        bbsz_to_valid_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                # this is equivalent to "if step >= 1" since "reorder_state" will never be None after the 0-th step
                # reorder_state is active_bbsz_idx
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)

                # take care of invalid beams: exclude them in bsz * beam_size
                if not valid_bbsz_mask.all():
                    # take out only valid beams
                    reorder_state = reorder_state[valid_bbsz_mask]
                if bbsz_to_valid_idxs is not None:
                    # the states from last step are not full bsz * beam_size, but only those for valid beams are stored
                    reorder_state = bbsz_to_valid_idxs[reorder_state]
                    assert (reorder_state >= 0).all(), 'check: invalid beam is selected from last step'

                # reorder/reduce the states
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

                # get the new index mapping from full bsz * beam_size to valid size for next step reordering
                if valid_bbsz_mask.all():
                    bbsz_to_valid_idxs = None
                else:
                    bbsz_to_valid_idxs = tokens.new(tokens.size(0)).fill_(-1)
                    bbsz_to_valid_idxs[valid_bbsz_mask] = torch.arange(valid_bbsz_mask.sum()).to(tokens.device)

            # ========== take out the beams selected from last step that are valid ==========
            tokens_valid = tokens[valid_bbsz_mask, :step + 1]
            valid_bbsz_num = tokens_valid.size(0)    # this may be smaller than bsz * beam_size
            # ===============================================================================

            # ========== use the AMR state machine (if turned on) to restrict the next action space ==========

            # restrict the action space for next candidate tokens
            # allowed_mask = tokens.new_zeros(valid_bbsz_num, self.vocab_size, dtype=torch.uint8)  # only for pytorch <= 1.1
            allowed_mask = tokens.new_zeros(valid_bbsz_num, self.vocab_size, dtype=BOOL_TENSOR_TYPE)
            tok_cursors = tokens.new_zeros(valid_bbsz_num, dtype=torch.int64)
            # tgt input tokens
            tgt_in = tokens_valid.clone()

            # debug: on dev data 2nd batch
            # if sample['nsentences'] == 208:
            #     breakpoint()
            # ==========> bug: self.tgt_dict is somehow changed with an additional token '<<unk>>' at the end

            if amr_state_machines is not None:

                constrain_pointer = defaultdict(lambda: defaultdict(list))
                for i, j in enumerate(valid_bbsz_idx):
                    sm = amr_state_machines[j]
                    # get the machine states
                    states = sm.get_states()
                    act_allowed = states['allowed_cano_actions']

                    # use predicate rules to further restrict the action space for PRED actions
                    pred_allowed = None
                    if use_pred_rules:
                        assert self.pred_rules is not None
                        # TODO update below (currently not used)
                        #      we use "NODE" keyword instead of "PRED"
                        if 'PRED' in act_allowed:
                            raise NotImplementedError()
                            src_token = sm.get_current_token()
                            if src_token in self.pred_rules:
                                act_allowed.remove('PRED')
                                pred_allowed = list(self.pred_rules[src_token].keys())

                    # get valid actions
                    # TODO: Move out of fairseq code into own function
                    vocab_ids_allowed = set()
                    for act in act_allowed:
                        if act == 'CLOSE':
                            vocab_ids_allowed.add(self.tgt_dict.index('</s>'))
                        elif act in canonical_act_ids:
                            vocab_ids_allowed |= set(canonical_act_ids[act])
                        elif re.match(r'>[LR]A\(([0-9]+),([^)]+)\)', act):
                            arc, index, label = re.match(r'>([LR]A)\(([0-9]+),([^)]+)\)', act).groups()
                            new_act = f'>{arc}({label})'
                            vocab_ids_allowed.add(self.tgt_dict.index(new_act))
                            constrain_pointer[j.item()][int(index)].append(new_act)
                        else:
                            # non canonincal actions (explicit node names)
                            vocab_ids_allowed.add(self.tgt_dict.index(act))

                    # in align mode
                    # there can not be any <unk>
                    # if a node name is not in the vocabulary, we need to force
                    # a COPY at least on the last time that it is possible,
                    # taking into account that there may be more than one token
                    if (
                        sm.gold_amr is not None
                        and sm.get_current_token() is not None
                    ):

                        # forced COPY
                        # count the number of times each token appears
                        future_token_nname_counts = Counter([
                            normalize(t)
                            for t in sm.tokens[sm.tok_cursor+1:]
                        ])

                        # count the number of times each node not in vocabulary
                        # appears
                        missing_unk_nodes = Counter()
                        unk_idx = self.tgt_dict.index('<unk>')
                        for nname in sm.align_tracker.get_missing_nnames(repeat=True):
                            nname = normalize(nname)
                            if self.tgt_dict.index(nname) == unk_idx:
                                missing_unk_nodes.update([nname])

                        # if token under cursor matches a node not in
                        # vocabulary and we have not enough future tokens of
                        # this type to produce all missing nodes, we have no
                        # option but to COPY
                        nname = normalize(sm.get_current_token())
                        if (
                            nname in missing_unk_nodes
                            and future_token_nname_counts[nname]
                                < missing_unk_nodes[nname]
                        ):
                            copy_idx = self.tgt_dict.index('COPY')
                            assert copy_idx in vocab_ids_allowed, \
                                'align mode needs all nodes to be in ' \
                                'the vocabulary or predictable by COPY.' \
                                ' This should not be happening.'
                            vocab_ids_allowed = {copy_idx}

                        # in align mode, there can not be <unk>
                        unk_set = set([self.tgt_dict.index('<unk>')])
                        unk_actions_str = ' '.join(act_allowed)
                        assert bool(vocab_ids_allowed -  unk_set), \
                            f'action(s) "{unk_actions_str}" not in vocabulary' \
                            ' align mode needs all actions in vocabualry'

                    # TODO update below
                    # use predicate rules to further restrict the action space for PRED actions
                    if pred_allowed is not None:
                        raise NotImplementedError()
                        pred_ids_allowed = set(self.tgt_dict.index(f'PRED({sym})') for sym in pred_allowed)
                        vocab_ids_allowed = vocab_ids_allowed.union(pred_ids_allowed)

                    allowed_mask[i, list(vocab_ids_allowed)] = 1

                    tok_cursors[i] = states['token_cursors'][-1]

                    if step >= 1:
                        # target input sequence
                        # NOTE self.tgt_dict.encode_line() returns an IntTensor (torch.int32);
                        # without making it to list there will be an error "Segmentation fault" hard to debug
                        tgt_in_values = self.tgt_dict.encode_line(
                                            line=[act if act != 'CLOSE' else self.tgt_dict.eos_word
                                                  for act in states['actions_nopos_in']],
                                            line_tokenizer=lambda x: x,    # already tokenized
                                            add_if_not_exist=False,
                                            consumer=None,
                                            append_eos=False,
                                            reverse_order=False
                                            ).tolist()
                        tgt_in[i][1:] = tgt_in.new(tgt_in_values)


                # NOTE blocking <unk> separately is needed when `use_pred_rules` is True, as the possible predicates
                #      generated by training oracle are not fully contained in the dictionary
                allowed_mask[:, self.unk] = 0    # TODO to look further into this and maybe clean it
                # without `use_pred_rules`:
                # pad and unk tokens are never allowed from the state machine above
            else:
                allowed_mask.fill_(1)
                allowed_mask[:, self.pad] = 0
                allowed_mask[:, self.unk] = 0
                allowed_mask[:, self.tgt_dict.bos()] = 0
                # explicitly mask out the pad and unk tokens (bos should be masked out probably as well)

            # ====================================================================

            # ========== get the actions states auxiliary information needed for the model to run in real-time ========

            actions_states = {'tgt_vocab_masks': allowed_mask.unsqueeze(1),
                              'tgt_src_cursors': tok_cursors.unsqueeze(1)}

            # lprobs, avg_attn_scores = model.forward_decoder(
            #     tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            # )
            avg_attn_scores = None    # this for the cross attention on the source tokens

            lprobs, avg_attn_tgt_scores = model.forward_decoder(
                tgt_in, encoder_outs, temperature=self.temperature, **actions_states
            )

            # lprobs[:, self.pad] = -math.inf  # never select pad
            # lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            lprobs[~allowed_mask] = -math.inf
            # may not need if the model forward process includes masking softmax output already

            # =========================================================================================================

            # NOTE this is currently operating on the full bsz * beam_size beams
            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                if valid_bbsz_num == bsz * beam_size:
                    attn[:, :, step + 1].copy_(avg_attn_scores)
                else:
                    attn[valid_bbsz_mask, :, step + 1].copy_(avg_attn_scores)

            # record target self-attention scores
            if avg_attn_tgt_scores is not None:
                if attn_tgt is None:
                    attn_tgt = dict()
                    # attn_tgt_buf = dict()
                    nonpad_idxs_tgt = dict()
                if valid_bbsz_num == bsz * beam_size:
                    attn_tgt[step + 1] = avg_attn_tgt_scores
                else:
                    attn_tgt[step + 1] = scores.new(bsz * beam_size, avg_attn_tgt_scores.size(1)).fill_(-1)
                    attn_tgt[step + 1][valid_bbsz_mask] = avg_attn_tgt_scores
                # NOTE we blocked generating pad token so this might not be necessary
                nonpad_idxs_tgt[step + 1] = tokens[:, :step + 1].ne(self.pad)

            # ========== pointer values decoding ==========

            # record best alignment on the previous target actions and associated log attention scores
            if tgt_pointers is None:
                tgt_pointers = torch.zeros_like(tokens).fill_(-1)
            if scores_tgt_pointers is None:
                scores_tgt_pointers = scores.new(bsz * beam_size, max_len + 1).fill_(0)

            # 1) get a mask for valid previous actions that can generate nodes
            # mask for (previous + current) actions (generated tgt tokens) that are corresponding to AMR nodes
            # the mask includes the current action
            # tgt_actions_nodemask = tokens.new_zeros(valid_bbsz_num, step + 1).byte()  # only for pytorch <= 1.1
            tgt_actions_nodemask = tokens.new_zeros(valid_bbsz_num, step + 1, dtype=BOOL_TENSOR_TYPE)

            if step == 0:
                # do nothing to the mask, since we don't have any action history yet, no pointer is generated
                pass
            else:
                if amr_state_machines is not None:
                    # get the previous action-to-node mask: 1 if an action generates a node, 0 otherwise
                    for i, j in enumerate(valid_bbsz_idx):
                        sm = amr_state_machines[j]
                        # NOTE the 0-th target token is the eos </s> token
                        actions_nodemask = sm.get_actions_nodemask().copy()
                        # mask out the last node generating action, as it will never be selected by the pointer, except
                        # the LA(root) action (as root does not need to be generated as a node)
                        # ---> do not do this now
                        # if 1 in actions_nodemask:
                        #     last_node_act_idx = len(actions_nodemask) - 1 - actions_nodemask[::-1].index(1)
                        #     actions_nodemask[last_node_act_idx] = 0
                        if self.shift_pointer_value:
                            tgt_actions_nodemask[i, 1:] = tgt_actions_nodemask.new(actions_nodemask)
                        else:
                            tgt_actions_nodemask[i, :-1] = tgt_actions_nodemask.new(actions_nodemask)

                else:
                    if self.shift_pointer_value:
                        # only mask out the current action, and the first position, which is the eos (</s>) token
                        tgt_actions_nodemask[:, 1:-1] = 1
                    else:
                        tgt_actions_nodemask[:, :-1] = 1
                    # NOTE we need to run the state machine; here just leave a warning instead of stopping the program
                    # for debugging convenience under different setups, even if the generated pointers are not valid
                    warnings.warn('actions to node mask not provided; the pointer values may not be valid.')

            # 2) get the argmax and max out of the pointer (tgt attention) distribution constraint to the above mask
            pointer_probs = avg_attn_tgt_scores.clone()
            pointer_probs[~tgt_actions_nodemask] = 0

            if constrain_pointer:
                # if you have more than action that requires pointer,
                # add masks, but also forbid the labels of the non chosen
                # NOTE: We need to shitf index by 1, see below
                bsize, seqlen = pointer_probs.shape
                for j in range(bsize):
                    if not bool(constrain_pointer[j]):
                        continue
                    num_valid = len([1 for x in constrain_pointer[j].values() if x])
                    for i in range(seqlen - 1):
                        if constrain_pointer[j][i]:
                            pointer_probs[j, i+1] =  1 / num_valid
                        else:
                            pointer_probs[j, i+1] = 0

            # TODO: most probable pointing determines valid label choice
            pointer_max, pointer_argmax = pointer_probs.max(dim=1)
            """
            NOTE the pointer distribution is from the target input side self-attention, which is shifted to the right
            by 1, and the first token is always </s> which will always be masked out, thus the "pointer_argmax"
            is always >= 1 whenever it is valid (specified by "tgt_actions_nodemask_any" mask below).
            we have to shift the pointer values back to match the target output side index (starting from 0)
            """
            if self.shift_pointer_value:
                pointer_argmax = pointer_argmax - 1

            tgt_actions_nodemask_any = tgt_actions_nodemask.sum(dim=1) > 0
            # max will be log pointer probs, except that rows with all 0 action-to-node mask will be 0
            # argmax will be pointer positions, except that rows with all 0 action-to-node mask will be set to -1
            pointer_max[tgt_actions_nodemask_any] = pointer_max[tgt_actions_nodemask_any].log()
            if valid_bbsz_num == bsz * beam_size:
                scores_tgt_pointers[:, step] = pointer_max
                # a different implementation for same results
                # pointer_argmax[~tgt_actions_nodemask_any] = -1
                # tgt_pointers[:, step + 1] = pointer_argmax
                tgt_pointers[tgt_actions_nodemask_any, step + 1] = pointer_argmax[tgt_actions_nodemask_any]
            else:
                scores_tgt_pointers[valid_bbsz_mask, step] = pointer_max
                tgt_pointers_valid = tgt_pointers[valid_bbsz_mask, step + 1]
                tgt_pointers_valid[tgt_actions_nodemask_any] = pointer_argmax[tgt_actions_nodemask_any]
                tgt_pointers[valid_bbsz_mask, step + 1] = tgt_pointers_valid

            # 3) use the pointer log probs to modify the next ARC ('LA', 'RA', 'LA(root)') actions scores
            # for rows with valid pointer value, modify the arc action scores;
            # for rows with no valid pointer value, set the arc action scores to -inf to block
            if modify_arcact_score:
                # NOTE for either we use '>LA(root)' or not in our oracle for handling root node
                arc_action_ids = list(set().union(*[canonical_act_ids[act] for act in ['>LA', '>RA', '>LA(root)']
                                                    if act in canonical_act_ids]))
                lprobs_arcs = lprobs[:, arc_action_ids]
                lprobs_arcs[tgt_actions_nodemask_any, :] += coef * pointer_max[tgt_actions_nodemask_any].unsqueeze(1)
                lprobs_arcs[~tgt_actions_nodemask_any, :] = -math.inf
                lprobs[:, arc_action_ids] = lprobs_arcs

                if constrain_pointer:
                    # If pointer is constrained, we need to also constrain arc
                    # labels consistently. Since we selected already an action
                    # history position per sentence, this tells use which are
                    # actions to restrict to
                    for j, idx_act in constrain_pointer.items():
                        if not bool(constrain_pointer[j]):
                            continue
                        save_action_lprobs = []
                        for action in constrain_pointer[j][pointer_argmax[j].item()]:
                            idx = self.tgt_dict.index(action)
                            save_action_lprobs.append((idx, lprobs[j, idx].item()))
                        # all other actions are forbidden
                        lprobs[j, :] = -math.inf
                        for idx, lprob in save_action_lprobs:
                            lprobs[j, idx] = lprob

                        # model may have assigned -inf to forced action, set to prob 1
                        if (lprobs[j, :] == -math.inf).all():
                            # FIXME: Add a warning for this
                            lprobs[j, idx] = 0.0

                        #assert (lprobs[j, :] != -math.inf).any(), \
                        #    "Alignment error, one force arc must have p>0"


            # ====================================================

            # ========== convert back the size of lprobs to bsz * beam_size from only valid beams ==========
            # stuff -inf scores to invalid positions, to have the full size for matrix reshape during search

            lprobs_buf = lprobs.new(bsz * beam_size, self.vocab_size).fill_(-math.inf)
            lprobs_buf[valid_bbsz_mask] = lprobs
            lprobs = lprobs_buf

            # ====================================================

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < max_len:
                self.search.set_src_lengths(src_lengths)

                if self.no_repeat_ngram_size > 0:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                    for bbsz_idx in range(bsz * beam_size):
                        lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

                # TODO if prefix_tokens is not None: not supported now
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    assert isinstance(self.search, search.BeamSearch) or bsz == 1, \
                        "currently only BeamSearch supports decoding with prefix_tokens"
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1)
                    ).view(-1, 1).repeat(1, cand_size)
                    if step > 0:
                        # save cumulative scores for each hypothesis
                        cand_scores.add_(scores[:, step - 1].view(bsz, beam_size).repeat(1, 2))
                    cand_indices = prefix_tokens[:, step].view(-1, 1).repeat(1, cand_size)
                    cand_beams = torch.zeros_like(cand_indices)

                # handle prefixes of different lengths
                # when step == prefix_tokens.size(1), we'll have new free-decoding batches
                if prefix_tokens is not None and step <= prefix_tokens.size(1):
                    if step < prefix_tokens.size(1):
                        partial_prefix_mask = prefix_tokens[:, step].eq(self.pad)
                    else:  # all prefixes finished force-decoding
                        partial_prefix_mask = torch.ones(bsz).to(prefix_tokens).byte()
                    if partial_prefix_mask.any():
                        # track new free-decoding batches, at whose very first step
                        # only use the first beam to eliminate repeats
                        prefix_step0_mask = partial_prefix_mask ^ partial_prefix_mask_buf
                        lprobs.view(bsz, beam_size, -1)[prefix_step0_mask, 1:] = -math.inf
                        partial_scores, partial_indices, partial_beams = self.search.step(
                            step,
                            lprobs.view(bsz, -1, self.vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                        )
                        cand_scores[partial_prefix_mask] = partial_scores[partial_prefix_mask]
                        cand_indices[partial_prefix_mask] = partial_indices[partial_prefix_mask]
                        cand_beams[partial_prefix_mask] = partial_beams[partial_prefix_mask]
                        partial_prefix_mask_buf = partial_prefix_mask

                else:
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                    )
            else:
                # NOTE the ending condition should never be max step reached; in principle our generation is contraint
                # on the the source sequences, and we finish generation of an action sequence only when we have
                # processed all the source words
                #raise ValueError('max step reached; we should set proper max step value so that this does not happen.')

                warnings.warn('max step reached; we should set proper max step value so that this does not happen. '
                              'OR: the generation is stuck at some repetitive patterns.')

                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores, amr_state_machines))
                # assert num_remaining_sent == 0
                # stop the loop here
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size] (NOTE: cand_size = 2 * beam_size)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            assert set(cand_bbsz_idx.unique().tolist()).issubset(valid_bbsz_idx.tolist()), \
                'new beam candidates should only stem from valid beams last step'

            # disallowed candidates mask (invalid positions)
            cand_disallowed = cand_scores == -math.inf

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)    # size (bsz, 2 * beam_size)
            eos_mask[cand_disallowed] = 0

            finalized_sents = set()    # NOTE is this line necessary? Yes, to empty the list from previous step.
            if step >= self.min_len:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,    # here eos_bbsz_idx is 1-D
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,  # here eos_scores is 1-D
                    )
                    # ========== disallow the finalized hypos ==========
                    # this is used for the ending condition in function "is_finished()", when there is no more valid
                    # options but we haven't reached beam_size --> force quit for this sentence
                    cand_scores[:, :beam_size][eos_mask[:, :beam_size]] = -math.inf
                    # ==================================================
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, amr_state_machines, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                # there are > 0 candidates ending with <eos> that are in the first beam_size candidates, and finalized
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)    # size (bsz,)
                batch_mask[cand_indices.new(finalized_sents)] = 0    # finished sentences have mask 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)    # indices of unfinished sentences, 1-D, length new_bsz

                eos_mask = eos_mask[batch_idxs]        # size (new_bsz, 2 * beam_size)
                cand_disallowed = cand_disallowed[batch_idxs]    # size (new_bsz, 2 * beam_size)
                cand_beams = cand_beams[batch_idxs]    # size (new_bsz, 2 * beam_size)
                bbsz_offsets.resize_(new_bsz, 1)
                # NOTE Tensor().resize_() actually prune the original elements when the new size is smaller!
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]  # size (new_bsz, 2 * beam_size)
                cand_indices = cand_indices[batch_idxs]    # size (new_bsz, 2 * beam_size)
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                    partial_prefix_mask_buf = partial_prefix_mask_buf[batch_idxs]
                src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                # NOTE resize_as_: here only the sizes are reduced, but the content might be messed up
                # NOTE so the contents of buffer here are not important, since they will be refreshed later (? confirm)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)

                # ========== index the action related beam info to remove the finished sentences ==========
                # indexing on the batch dimension
                if attn_tgt is not None:
                    for s, v in attn_tgt.items():
                        # NOTE must call contiguous() before view in some cases; or use reshape
                        attn_tgt[s] = v.reshape(bsz, -1)[batch_idxs].contiguous().view(new_bsz * beam_size, v.size(1))
                    # TODO figure out how the above .resize_as_() works for buf when the sizes are actually different?
                    # (cont) DONE
                    # for s, v in attn_tgt_buf.items():
                    #     attn_tgt_buf[s] = v.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, v.size(1))
                    # NOTE following the fairseq implementation, buffer contents here are not important since they will
                    # be replaced. Only the size matters.
                if amr_state_machines is not None:
                    batch_idxs_list = batch_idxs.tolist()
                    amr_state_machines = [sm for i, sm in enumerate(amr_state_machines)
                                          if i // beam_size in batch_idxs_list]

                if tgt_pointers is not None:
                    tgt_pointers = tgt_pointers.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                    scores_tgt_pointers = scores_tgt_pointers.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                # =================================================================

                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values >= cand_size indicate eos hypos or disallowed hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            # NOTE we try to exclude the following candidates in the selection:
            #      a) <eos> in the first beam_size candidates (which are already finished)
            #      b) <eos> in the second beam_size candidates (beams are only finished if <eos> is in the first half
            #         candidates within beam_size)
            #      c) disallowed candidates (in our case, candidates with -inf scores)
            active_mask = buffer('active_mask')
            eos_or_disallowed = eos_mask | cand_disallowed
            torch.add(
                eos_or_disallowed.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_or_disallowed.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            # NOTE this is where cand_size = 2 * beam_size plays its role
            # NOTE here after removing the EOS candidates, we can always ensure beam_size candidates remained
            active_hypos, active_mask_selected = buffer('active_hypos'), buffer('active_mask_selected')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(active_mask_selected, active_hypos)
            )

            # ========== get the valid/disallowed beam mask for the selected top beam_size beams ==========
            # top beam_size beams could include disallowed candidates when beam_size is larger than the allowed
            # candidate space size

            valid_bbsz_mask = active_mask_selected.lt(cand_size)    # size (bsz, beam_size)
            if not valid_bbsz_mask.any(dim=1).all():
                # if we are in align mode, locate the first infringing machine
                if any(bool(m.gold_amr) for m in amr_state_machines):
                    for i, m in enumerate(amr_state_machines):
                        if not valid_bbsz_mask[i].item():
                            all_valid_actions = m.get_valid_actions()
                            raise Exception(
                                'there must be remaining valid candidates '
                                'for each sentence in batch.' 'Maybe trying to'
                                ' align a node name not in vocabulary?, \n'
                                f'check: {all_valid_actions}'
                            )
                else:
                    raise Exception('there must be remaining valid candidates for each sentence in batch')
            valid_bbsz_mask = valid_bbsz_mask.view(-1)    # size (bsz x beam_size,)
            valid_bbsz_idx = valid_bbsz_mask.nonzero().squeeze(-1)    # size (valid_bbsz_num,)

            # =============================================================================================

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            # reorder tokens
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            # add new tokens
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )

            # reorder the AMR state machine and target pointer info on previous beams
            # print([(i, sm.is_closed, sm.action_history) for i, sm in enumerate(amr_state_machines)])
            # import pdb; pdb.set_trace()

            # reorder target pointer values and scores
            tgt_pointers[:, :step + 2] = torch.index_select(tgt_pointers[:, :step + 2], dim=0, index=active_bbsz_idx)
            if step > 0:
                scores_tgt_pointers[:, step + 1] = torch.index_select(scores_tgt_pointers[:, step + 1], dim=0,
                                                                      index=active_bbsz_idx)

            # reorder state machines
            if amr_state_machines is not None:
                if step > 0:
                    # NOTE here must use copy since there could be same ids from active_bbsz_idx
                    # (from the same last beam)
                    amr_state_machines = [deepcopy(amr_state_machines[i]) for i in active_bbsz_idx]

                # add and apply new action tokens to state machine
                for i, (sm, act_id, act_pos, is_valid) in enumerate(zip(amr_state_machines,
                                                                    tokens_buf[:, step + 1],
                                                                    tgt_pointers[:, step + 1],
                                                                    valid_bbsz_mask)):
                    # do not run the state machine if the action is not valid; this state machine will be "deleted"
                    # as it will not be indexed out anymore in the next step
                    if not is_valid:
                        continue
                    # eos changed to CLOSE action (although NOTE currently this will never be eos at this step)
                    # FIXME: This is state machine specific, we need to remove it
                    act = self.tgt_dict[act_id] if act_id != self.eos else 'ĠCLOSE'
                    sm.apply_action_and_update_states(act, act_pos.item())


            # ============================================================

            if step > 0:
                # reorder scores
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            # add new scores
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # ========== reorder the target self-attention information in beams ==========
            # copy decoder self attention for active hypotheses
            if attn_tgt is not None:
                # reorder all the past attentions on the bsz * beam_size dimension
                for s in attn_tgt.keys():
                    # s is the step id, starting from 1
                    attn_tgt[s] = torch.index_select(attn_tgt[s], dim=0, index=active_bbsz_idx)
            # ==================================================

            # swap buffers (then buffers are freed to use for next step)
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., **kwargs):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                **kwargs
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                **kwargs
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., **kwargs
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.decoder(tokens, encoder_out, incremental_state=self.incremental_states[model],
                                             **kwargs))
        else:
            decoder_out = list(model.decoder(tokens, encoder_out, **kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if isinstance(attn, dict):
            attn = attn['attn']
        if attn is not None:
            if isinstance(attn, dict):
                attn = attn['attn']
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state_scripting(self.incremental_states[model], new_order)
