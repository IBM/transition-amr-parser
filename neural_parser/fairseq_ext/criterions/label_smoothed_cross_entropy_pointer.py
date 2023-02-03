# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions import LegacyFairseqCriterion    # for version >= 10.0.0


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    # remove entries of ignored indices
    if ignore_index is not None:
        # FIXME: May have broken other cases in orther to make the smoothed
        # loss suppor -Inf logits
        assert lprobs.dim() == 2
        lprobs = lprobs[target.ne(ignore_index).view(-1), :]
        target = target[target.ne(ignore_index)].unsqueeze(1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # support -Inf
    smooth_loss = -lprobs[lprobs != float("-Inf")].sum(dim=-1, keepdim=True)
    if ignore_index is None:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        raise NotImplementedError("Suporting -Inf removed non reduce mode")
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def safe_mask(tensor, mask, default_val=0):
    """
    Returns a new tensor of shape `tensor.shape` such that values where
    mask is True are values from tensor, and `default_val` otherwise.
    """
    new_tensor = torch.full(tensor.shape, default_val, dtype=tensor.dtype, device=tensor.device)
    new_tensor[mask] = tensor[mask]
    return new_tensor


def label_smoothed_nll_loss_enhanced(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    Same as `label_smoothed_nll_loss` except supports reduce=True. Might be slower due to different
    approach to masking.
    """
    assert lprobs.dim() == 3

    batch_size, seq_length, vocab_size = lprobs.shape
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    if ignore_index is not None and ignore_index < 0:
        # Need this condition, since gather fails for values < 0.
        mask = target != ignore_index
        index = safe_mask(target, mask, default_val=0)
        nll_loss = -lprobs.gather(dim=-1, index=index)
    else:
        nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -safe_mask(lprobs, lprobs != float("-Inf"), default_val=0).sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        mask = target != ignore_index
        nll_loss = safe_mask(nll_loss, mask, default_val=0)
        smooth_loss = safe_mask(smooth_loss, mask, default_val=0)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        nll_loss = nll_loss.view(batch_size, seq_length).sum(1)
        smooth_loss = smooth_loss.view(batch_size, seq_length).sum(1)

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss_pointer(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    # remove entries of ignored indices
    if ignore_index is not None:
        # FIXME: May have broken other cases in orther to make the smoothed
        # loss suppor -Inf logits
        assert lprobs.dim() == 2
        lprobs = lprobs[target.ne(ignore_index).view(-1), :]
        target = target[target.ne(ignore_index)].unsqueeze(1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    # import pdb
    # pdb.set_trace()
    # nll_loss = lprobs.gather(dim=-1, index=target)
    # nll_loss = -torch.log(nll_loss)
    # support -Inf
    smooth_loss = -lprobs[lprobs != float("-Inf")].sum(dim=-1, keepdim=True)
    if ignore_index is None:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        raise NotImplementedError("Suporting -Inf removed non reduce mode")
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_pointer')
class LabelSmoothedCrossEntropyPointerCriterion(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.loss_coef = getattr(args, 'loss_coef', 1)
        self.shift_pointer_value = args.shift_pointer_value

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--loss-coef', default=1., type=float, metavar='C',
                            help='lambda for combining the pointer position loss, 0 means no pointer loss')
        parser.add_argument('--shift-pointer-value', default=1, type=int,
                            help='whether to shift the pointer value one to the right, to tie with the target input '
                                 'positions which are shifted by one')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        def compute_importance_weighted_p(log_p_token, log_p_pointer, log_q):
            log_p = log_p_token + self.loss_coef * log_p_pointer
            with torch.no_grad():
                log_w = log_p.detach() - log_q.detach() # extra cautious with detach.
                w_tilde = (log_w / self.args.importance_weighted_temp).softmax(-1)
            # NOTE: Since we don't backprop through q, it's okay to use log_p instead of log_w.
            new_p = w_tilde * log_p
            return new_p

        sample_alignments = sample.get('sample_alignments', 1)
        if sample_alignments > 1 and self.args.importance_weighted_align:
            k = sample_alignments
            batch_size = sample['target'].shape[0] // k
            loss_seq, nll_loss_seq = self.compute_loss(model, net_output, sample, reduce=False)
            loss_pos, nll_loss_pos = self.compute_pointer_loss(net_output, sample, reduce=False)

            loss_seq = loss_seq.view(batch_size, k)
            nll_loss_seq = nll_loss_seq.view(batch_size, k)
            loss_pos = loss_pos.view(batch_size, k)
            nll_loss_pos = nll_loss_pos.view(batch_size, k)

            # Importance weights.
            log_q = torch.tensor(sample['lp_align'], dtype=torch.float, device=loss_seq.device).view(batch_size, k)
            loss = -compute_importance_weighted_p(-loss_seq, -loss_pos, log_q).sum()
            nll_loss = -compute_importance_weighted_p(-nll_loss_seq, -nll_loss_pos, log_q).sum()

            # Used for logging.
            loss_seq = loss_seq.sum()
            nll_loss_seq = nll_loss_seq.sum()
            loss_pos = loss_pos.sum()
            nll_loss_pos = nll_loss_pos.sum()

        else:
            loss_seq, nll_loss_seq = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss_pos, nll_loss_pos = self.compute_pointer_loss(net_output, sample, reduce=reduce)
            loss = loss_seq + self.loss_coef * loss_pos
            nll_loss = nll_loss_seq + self.loss_coef * nll_loss_pos

        # TODO use different normalization factor for two types of losses
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'loss_seq': utils.item(loss_seq.data) if reduce else loss_seq.data,
            'nll_loss_seq': utils.item(nll_loss_seq.data) if reduce else nll_loss_seq.data,
            'loss_pos': utils.item(loss_pos.data) if reduce else loss_pos.data,
            'nll_loss_pos': utils.item(nll_loss_pos.data) if reduce else nll_loss_pos.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        if reduce:
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = target.view(-1, 1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        else:
            loss, nll_loss = label_smoothed_nll_loss_enhanced(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )

        return loss, nll_loss

    def compute_pointer_loss(self, net_output, sample, reduce=True):
        target_pos = sample['tgt_pos']
        # shift the pointer value 1 to the right as it's for the input with the first token </s>
        # while keeping the pointer positions unchanged as it's for the output
        if self.shift_pointer_value:
            target_pos[target_pos >= 0] += 1
        target_pos[target_pos < 0] = -1    # use the same value -1 to mask out tgt tokens that do not have pos or padded
        # NOTE in above 0 is a valid pos value
        loss_all, nll_loss_all = [], []
        for attn in net_output[1]['attn_all']:
            # attn = attn.float()
            # this is for numerical stability; otherwise log backward will get nan
            attn = attn.float().clamp(min=1e-8)
            # attn = attn.float() + 1e-8
            attn = torch.log(attn)
            # includes '-inf' when attn is 0 (e.g. for future positions) -> causes log backward to get nan
            # or if cast type after log
            # attn = attn.clamp(min=1e-6)    # fp16 range +-65504, precision 5.96e-8 TODO this also has error for float16
            # attn = torch.log(attn).float()
            # NOTE dtype transfer is needed for float16 training
            # NOTE we do not want to do label smoothing for pointer (right?)
            if reduce:
                attn = attn.contiguous().view(-1, attn.size(-1))    # size (bsz, tgt_len * tgt_len)
                target_pos = target_pos.view(-1, 1)
                loss, nll_loss = label_smoothed_nll_loss_pointer(attn, target_pos, 0, ignore_index=-1, reduce=reduce)
            else:
                # TODO:
                bsz, tgt_len, tgt_len2 = attn.shape
                assert tgt_len == tgt_len2
                assert target_pos.shape == (bsz, tgt_len)
                loss, nll_loss = label_smoothed_nll_loss_enhanced(attn, target_pos, 0, ignore_index=-1, reduce=reduce)
            loss_all.append(loss)
            nll_loss_all.append(nll_loss)
        loss = sum(loss_all) / len(loss_all)
        nll_loss = sum(nll_loss_all) / len(nll_loss_all)
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'loss_seq': sum(log.get('loss_seq', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss_seq': sum(log.get('nll_loss_seq', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'loss_pos': sum(log.get('loss_pos', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss_pos': sum(log.get('nll_loss_pos', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
