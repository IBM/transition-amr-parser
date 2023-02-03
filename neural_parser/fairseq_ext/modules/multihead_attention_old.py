# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        # NOTE from pytorch 1.2.0 it will automatically use the pytorch version of attention
        #      this will be different from pytorch 1.1.0
        # ---> So we have to shut this off!
        # if hasattr(F, "multi_head_attention_forward"):
        #     self.enable_torch_version = True
        # else:
        #     self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None,
                head_attention_masks=None, head_positions=None,
                cross_attention_mask=None,
                ptr_self_attn_mask=None,
                graph_self_attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and not self.onnx_trace and incremental_state is None and not static_kv:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      self.in_proj_weight,
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      torch.empty([0]),
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask, use_separate_proj_weight=True,
                                                      q_proj_weight=self.q_proj_weight,
                                                      k_proj_weight=self.k_proj_weight,
                                                      v_proj_weight=self.v_proj_weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # encoder-decoder attention
        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:

                # key/value linear projections
                # (source_size, batch_size, target_emb_size)
                # ->
                # (source_size, batch_size, target_emb_size *2)
                # -> (chunked)
                # (source_size, batch_size, target_emb_size) * 2
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

            if head_positions is not None:
                # project position embeddings
                # (batch_size, source_size, target_size, target_emb_size)
                # ->
                # (batch_size, source_size, target_size, target_emb_size) * 2
                head_pos_emb_k = self.in_proj_k(head_positions)
                head_pos_emb_v = self.in_proj_v(head_positions)
                # FIXME: this assumes first two heads are stack/buffer
                # only the first two heads get stack/buffer pos added
                head_dim = head_pos_emb_k.shape[3] // self.num_heads
                head_pos_emb_k[:, :, :, 2 * head_dim:] = 0
                head_pos_emb_v[:, :, :, 2 * head_dim:] = 0

                # sanity check: assuming first action is a shift, this is
                # just inserting position zero at leftmost element
                # assert (head_pos_emb_k[0, 1:, 1, :] == head_pos_emb_k[0, :-1, 0, :]).all()
                # assert (head_pos_emb_k[0, 0, 1, :] ==  head_pos_emb_k[0, 0, 0, :]).all()

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            raise NotImplementedError(
                "attention bias not implemented for stack-transformer"
            )
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        # NOTE: It splits the emb_dim across heads and unfolds heads in batch
        # dimension. This is not standard multi-head attention!
        #
        # (target_size, batch_size, target_emb_size)
        # ->
        # (target_size, batch_size * num_heads, target_emb_size / num_heads)
        # ->
        # (batch_size * num_heads, target_size, target_emb_size / num_heads)
        #
        # dimension. Heads for same batch element are contiguous on that
        # dimensions. See example
        # dummy = torch.zeros(q.shape)
        # dummy[:, 0, :] = torch.ones(dummy[:, 0, :].shape)
        # dummy2 = dummy.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # (dummy2[:self.num_heads, :, :] == 1).sum() == (dummy2 == 1).sum()
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            # This saves key and value for this head with key e.g.
            # 'MultiheadAttention.1.attn_state'
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        if head_positions is not None:
            # (batch_size, source_size, target_size, target_emb_size)
            # ->
            # (batch_size * num_heads, source_size, target_size, target_emb_size / num_heads)
            # assert (head_pos_emb_k[0, 1:, 1, :] == head_pos_emb_k[0, :-1, 0, :]).all()
            head_pos_emb_k = head_pos_emb_k.transpose(0, 2).contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 2)
            head_pos_emb_v = head_pos_emb_v.transpose(0, 2).contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 2)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        # Compute unnormalized attention
        # q              (batch_size * num_heads, target_size, target_emb_size / num_heads)
        # k              (batch_size * num_heads, source_size, target_emb_size / num_heads)
        # ->
        # attn_weights   (batch_size * num_heads, target_size, source_size)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        # import pdb; pdb.set_trace()

        # mask out cross attention
        if cross_attention_mask is not None:
            # attn_weights[~cross_attention_mask[0]] = -float('inf')
            attn_weights = attn_weights.masked_fill(~cross_attention_mask[0], float('-inf'))

        if ptr_self_attn_mask is not None:
            attn_weights[:ptr_self_attn_mask[0].size(0)][~ptr_self_attn_mask[0]] = -float('inf')
            # attn_weights[:ptr_self_attn_mask[0].size(0)].masked_fill(~ptr_self_attn_mask[0], float('-inf'))

        # graph structure encoding: decoder self-attention mask
        if graph_self_attn_mask is not None:
            attn_weights = attn_weights.masked_fill(~graph_self_attn_mask[0], float('-inf'))

        if head_positions is not None:
            # if buffer/stack positions provided, add them to attention computation
            # Note that batched inner product is here implemented as
            # elements-wise product and sum across common axis
            # head_positions (batch_size * num_heads, source_size, target_size, target_emb_size / num_heads)
            # q              (batch_size * num_heads, target_size, target_emb_size / num_heads)
            # ->
            # attn_weights   (batch_size * num_heads, source_size, target_size)
            attn_weights += self.scaling * (q.unsqueeze(1) * head_pos_emb_k).sum(3).transpose(1, 2)

        # FIXME: What is this
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # Stack/Buffer individual mask per head
        if head_attention_masks is not None:
            if self.onnx_trace:
              # dunno whats this better die
              raise NotImplementedError()
            # mask in log domain with pre_mask
            attn_weights += head_attention_masks[0]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # if torch.isnan(attn_weights).any():
        #     import pdb; pdb.set_trace()

        # NOTE after softmax, we need to mask out the rows that are all masked out, which are full rows of
        # '-inf' before softmax -> 'nan' after softmax
        if cross_attention_mask is not None:
            # attn_weights[cross_attention_mask.sum(dim=2) == 0] *= 0
            # post_mask = cross_attention_mask.new_ones(*cross_attention_mask.size()[:2], 1).float()
            # post_mask[cross_attention_mask.sum(dim=2) == 0] = 0.0
            attn_weights = attn_weights * cross_attention_mask[1]

        if ptr_self_attn_mask is not None:
            # NOTE in-place operation doesn't work for backward
            # attn_weights[:ptr_self_attn_mask[1].size(0)] *= ptr_self_attn_mask[1]
            # attn_weights[:ptr_self_attn_mask[1].size(0)] = \
            #     attn_weights[:ptr_self_attn_mask[1].size(0)] * ptr_self_attn_mask[1]    # this doesn't work either
            # NOTE not sure why though
            ptr_self_attn_mask_tmp = ptr_self_attn_mask[1].new_ones(attn_weights.size(0),
                                                                    *ptr_self_attn_mask[1].size()[1:])
            ptr_self_attn_mask_tmp[:ptr_self_attn_mask[1].size(0)] = ptr_self_attn_mask[1]
            attn_weights = attn_weights * ptr_self_attn_mask_tmp

        # graph structure encoding: decoder self-attention mask
        if graph_self_attn_mask is not None:
            attn_weights = attn_weights * graph_self_attn_mask[1]

        # if torch.isnan(attn_weights).any():
        #     import pdb; pdb.set_trace()

        # post mask for empty buffer/stack
        if head_attention_masks is not None:
            # sanity check, this really blocks only all inf weights
            attn_weights = attn_weights * head_attention_masks[1]

        # Compute attended source
        # attn_weights   (batch_size * num_heads, target_size, source_size)
        # v              (batch_size * num_heads, source_size, target_emb_size / num_heads)
        # ->
        # attn           (batch_size * num_heads, target_size, target_emb_size / num_heads)
        attn = torch.bmm(attn_weights, v)
        if head_positions is not None:
            # if buffer/stack positions provided, add them to attention computation
            # Note that batched inner product is here implemented as
            # elements-wise product and sum across common axis
            # attn_weights   (batch_size * num_heads, target_size, source_size)
            # head_positions (batch_size * num_heads, source_size, target_size, target_emb_size / num_heads)
            # ->
            # attn           (batch_size * num_heads, target_size, target_emb_size / num_heads)
            attn += (attn_weights.transpose(1, 2).unsqueeze(3) * head_pos_emb_v).sum(1)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        # import pdb; pdb.set_trace()

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    # def _get_input_buffer(self, incremental_state):
    #     return utils.get_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #     ) or {}

    # def _set_input_buffer(self, incremental_state, buffer):
    #     utils.set_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #         buffer,
    #     )

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights
