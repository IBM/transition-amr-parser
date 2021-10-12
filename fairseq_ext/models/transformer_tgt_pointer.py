# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    # TransformerDecoderLayer,
    # TransformerEncoderLayer,
)

from torch_scatter import scatter_mean

from ..modules.transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from .attention_masks import get_cross_attention_mask, get_cross_attention_mask_heads


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_tgt_pointer')
class TransformerTgtPointerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', type=int, default=0,
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # Use stack transformer
        parser.add_argument('--encode-state-machine', type=bool,
                            help='controls encoding of stack and buffer')
        # control BERT backprop
        parser.add_argument('--bert-backprop', action='store_true',
                            help='Backpropagate through BERT', default=False)
        parser.add_argument('--no-bert-precompute', action='store_true',
                            help='Compute BERT on the fly (debugging)',
                            default=False)
        parser.add_argument('--pretrained-embed-dim', type=int,
                            help='Pretrained embeddings size',
                            default=768)
        # additional
        # NOTE do not set default values here; if set, make sure they are consistent with the arch registry
        #      since when loading model (e.g. saved pre some argument additions), the default values will be used first
        #      then the arch registry
        parser.add_argument('--apply-tgt-vocab-masks', type=int,
                            help='whether to apply target (actions) vocabulary mask for output')
        parser.add_argument('--apply-tgt-actnode-masks', type=int,
                            help='whether to apply target (actions) node mask for pointer')
        parser.add_argument('--apply-tgt-src-align', type=int,
                            help='whether to apply target source alignment to guide the cross attention')
        parser.add_argument('--apply-tgt-input-src', type=int,
                            help='whether to apply target input to include source token embeddings for better '
                                 'representations of the graph nodes')
        # additional: tgt src alignment masks for decoder cross-attention
        parser.add_argument('--tgt-src-align-layers', nargs='*', type=int,
                            help='target source alignment in decoder cross-attention: which layers to use')
        parser.add_argument('--tgt-src-align-heads', type=int,
                            help='target source alignment in decoder cross-attention: how many heads per layer to use')
        parser.add_argument('--tgt-src-align-focus', nargs='*', type=str,
                            help='target source alignment in decoder cross-attention: what to focus per head')
        # additional: pointer distribution from decoder self-attentions
        parser.add_argument('--pointer-dist-decoder-selfattn-layers', nargs='*', type=int,
                            help='pointer distribution from decoder self-attention: which layers to use')
        parser.add_argument('--pointer-dist-decoder-selfattn-heads', type=int,
                            help='pointer distribution from decoder self-attention: how many heads per layer to use')
        parser.add_argument('--pointer-dist-decoder-selfattn-avg', type=int,
                            help='pointer distribution from decoder self-attention: whether to use the average '
                                 'self-attention each layer (arithmetic mean); if set to 0, geometric mean is used; '
                                 'if set to -1, no average is used and all the pointer distributions are used to '
                                 'compute the loss')
        parser.add_argument('--pointer-dist-decoder-selfattn-infer', type=int,
                            help='pointer distribution from decoder self-attention: at inference, which layer to use')
        # additional: combine source token embeddings into action embeddings for decoder input for node representation
        parser.add_argument('--tgt-input-src-emb', type=str, choices=['raw', 'bot', 'top'],
                            help='target input to include aligned source tokens: where to take the source embeddings; '
                                 '"raw": raw RoBERTa embeddings from the very beginning; '
                                 '"bot": bottom source embeddings before the encoder; '
                                 '"top": top source embeddings after the encoder')
        parser.add_argument('--tgt-input-src-backprop', type=int,
                            help='target input to include aligned source tokens: whether to back prop through the '
                                 'source embeddings')
        parser.add_argument('--tgt-input-src-combine', type=str, choices=['cat', 'add'],
                            help='target input to include aligned source tokens: how to combine the source token '
                                 'embeddings and the action embeddings')
        # additional: factored embeddings for the target actions
        parser.add_argument('--tgt-factored-emb-out', type=int,
                            help='whether to set target output embeddings to be factored')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # user specific: make sure all arguments are present in older models during development
        transformer_pointer(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.bert_backprop = args.bert_backprop
        self.no_bert_precompute = args.no_bert_precompute

        # backprop needs on the fly extraction
        if self.bert_backprop or self.no_bert_precompute:
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
            roberta.cuda()
            self.roberta = roberta
            # if args.no_bert_precompute:
            #    # Set BERT to purely evaluation mode
            #    self.roberta.eval()

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        # BERT embeddings as input
        input_embed_dim = args.pretrained_embed_dim
        self.subspace = Linear(input_embed_dim, embed_dim, bias=False)

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # copying the arguments for the separate model in decoding to use
        self.args = args

    def forward(self, src_tokens, src_lengths, memory, memory_pos, source_fix_emb, src_wordpieces, src_wp2w, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        # embed tokens and positions
        # x = self.embed_scale * self.embed_tokens(src_tokens)
        # if self.embed_positions is not None:
        #     x += self.embed_positions(src_tokens)

        if self.bert_backprop or self.no_bert_precompute:
            # extract roberta on the fly
            last_layer = self.roberta.extract_features(src_wordpieces)
            # remove sentence start
            bsize, max_len, emb_size = last_layer.shape
            mask = (src_wordpieces != 0).unsqueeze(2).expand(last_layer.shape)
            last_layer = last_layer[mask].view((bsize, max_len - 1, emb_size))
            # remove sentence end
            last_layer = last_layer[:, :-1, :]
            # apply scatter, src_wp2w was inverted in pre-processing to use
            # scatter's left side padding . We need to flip the result.
            source_fix_emb2 = scatter_mean(
                last_layer,
                src_wp2w.unsqueeze(2),
                dim=1
            )
            source_fix_emb2 = source_fix_emb2.flip(1)
            # Remove extra padding
            source_fix_emb2 = source_fix_emb2[:, -src_tokens.shape[1]:, :]

            # do not backprop for on-the-fly computing
            if self.no_bert_precompute:
                bert_embeddings = source_fix_emb2.detach()
            else:
                bert_embeddings = source_fix_emb2

            # DEBUG: check precomputed and on the fly sufficiently close
            # abs(source_fix_emb2 - source_fix_emb).max()
        else:
            # use pre-extracted roberta
            bert_embeddings = source_fix_emb

        x = self.subspace(bert_embeddings)

        if self.args.apply_tgt_input_src and self.args.tgt_input_src_emb == 'bot':
            src_embs = x    # size (B, T, C)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        if self.args.apply_tgt_input_src:
            if self.args.tgt_input_src_emb == 'top':
                src_embs = x.transpose(0, 1)
            elif self.args.tgt_input_src_emb == 'raw':
                src_embs = bert_embeddings
            elif self.args.tgt_input_src_emb == 'bot':
                pass    # already dealt with above
            else:
                raise NotImplementedError
        else:
            src_embs = None

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'src_embs': src_embs,  # B x T x C
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['src_embs'] is not None:
            encoder_out['src_embs'] = encoder_out['src_embs'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        # controls the use of stack transformer
        self.encode_state_machine = args.encode_state_machine

        if self.encode_state_machine:
            # positions of buffer and stack for each time step
            self.embed_stack_positions = PositionalEmbedding(
                args.max_target_positions, args.decoder_embed_dim,
                padding_idx, learned=args.decoder_learned_pos,
            )

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            if not args.tgt_factored_emb_out:
                self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
                nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)
            else:
                # factored embeddings for target actions on the decoder output side
                from ..modules.factored_embeddings import FactoredEmbeddings
                self.factored_embeddings = FactoredEmbeddings(dictionary, self.output_embed_dim)
                self.dictionary_arange = torch.arange(len(dictionary)).unsqueeze(0)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # copying the arguments for the separate model in decoding to use
        self.args = args

        # target input to include source token embeddings
        if self.args.apply_tgt_input_src:
            assert self.args.tgt_input_src_emb != 'raw', 'Not implemented yet'
            if self.args.tgt_input_src_combine == 'cat':
                self.combine_src_embs = Linear(input_embed_dim + args.encoder_embed_dim, input_embed_dim, bias=False)

    def forward(self, prev_output_tokens, encoder_out, memory=None, memory_pos=None,
                incremental_state=None, logits_mask=None, logits_indices=None,
                tgt_vocab_masks=None, tgt_actnode_masks=None, tgt_src_cursors=None,
                **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            memory,
            memory_pos,
            encoder_out,
            incremental_state,
            tgt_src_cursors=tgt_src_cursors,
            tgt_actnode_masks=tgt_actnode_masks
        )
        x = self.output_layer(
            x,
            logits_mask=logits_mask,
            logits_indices=logits_indices,
            tgt_vocab_masks=tgt_vocab_masks,
        )

        # DEBUG: (consumes time)
        # if (x != x).any():
        #    import pdb; pdb.set_trace()
        #    print()

        return x, extra

    def extract_features(self, prev_output_tokens, memory, memory_pos,
                         encoder_out=None, incremental_state=None,
                         tgt_src_cursors=None, tgt_actnode_masks=None,
                         **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # It needs only the last auto-regressive element. Rest is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            # TODO this is a hacky way of ignoring these two
            memory = memory[:, :, -1:] if memory is not None else None
            memory_pos = memory_pos[:, :, -1:] if memory_pos is not None else None

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # ========== combine the corresponding source token embeddings with the action embeddings as input ==========
        if self.args.apply_tgt_input_src:
            # 1) take out the source embeddings
            src_embs = encoder_out['src_embs']    # size (batch_size, src_max_len, encoder_emb_dim)
            if not self.args.tgt_input_src_backprop:
                src_embs = src_embs.detach()

            # 2) align the source embeddings to the tgt input actions
            assert tgt_src_cursors is not None
            tgt_src_index = tgt_src_cursors.clone()    # size (bsz, tgt_max_len)
            if encoder_out['encoder_padding_mask'] is not None:
                src_num_pads = encoder_out['encoder_padding_mask'].sum(dim=1, keepdim=True)
                tgt_src_index = tgt_src_index + src_num_pads    # NOTE this is key to left padding!

            tgt_src_index = tgt_src_index.unsqueeze(-1).repeat(1, 1, src_embs.size(-1))
            # or
            # tgt_src_index = tgt_src_index.unsqueeze(-1).expand(-1, -1, src_embs.size(-1))
            src_embs = torch.gather(src_embs, 1, tgt_src_index)
            # size (bsz, tgt_max_len, src_embs.size(-1))

            # 3) combine the action embeddings with the aligned source token embeddings
            if self.args.tgt_input_src_combine == 'cat':
                x = self.combine_src_embs(torch.cat([src_embs, x], dim=-1))
            elif self.args.tgt_input_src_combine == 'add':
                x = src_embs + x
            else:
                raise NotImplementedError
        # ===========================================================================================================

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        attn_all = []

        inner_states = [x]

        # ========== alignment guidance in the cross-attention: get the mask ==========
        if self.args.apply_tgt_src_align:
            assert tgt_src_cursors is not None
            cross_attention_mask = get_cross_attention_mask_heads(tgt_src_cursors,
                                                                  encoder_out['encoder_out'].size(0),
                                                                  encoder_out['encoder_padding_mask'],
                                                                  self.args.tgt_src_align_focus,
                                                                  self.args.tgt_src_align_heads,
                                                                  self.layers[0].encoder_attn.num_heads)
        else:
            cross_attention_mask = None
        # ==============================================================================

        # TODO there are some problems with the pointer mask
        #      attention mask dimension is (bsz * num_heads, target_size, source_size)
        #      where for each batch, head dimension comes first
        #      BUT here, the pointer mask assumes the batch dimension comes first, since it only generates
        #      the number of heads needed for pointer and stacks them

        # ========== pointer distribution (decoder self-attention) mask ==========
        if self.args.apply_tgt_actnode_masks:
            assert tgt_actnode_masks is not None
            if self.args.shift_pointer_value:
                tgt_actnode_masks[:, 1:] = tgt_actnode_masks[:, :-1]
                tgt_actnode_masks[:, 0] = 0
            ptr_self_attn_mask = tgt_actnode_masks.unsqueeze(dim=1).repeat(1, tgt_actnode_masks.size(1), 1)
            ptr_self_attn_mask = ptr_self_attn_mask.unsqueeze(dim=1).repeat(
                1, self.args.pointer_dist_decoder_selfattn_heads, 1, 1).view(
                    -1, tgt_actnode_masks.size(1), tgt_actnode_masks.size(1))
            # NOTE need to include the causal mask as well in case some rows are completely masked out
            # in which case we need to do the post mask
            ptr_self_attn_mask &= (self.buffered_future_mask(x) != -float('inf')).unsqueeze(dim=0)
            # NOTE when one row out of bsz * num_heads (tgt_max_len, src_max_len) masks is full zeros, after softmax the
            # distribution will be all "nan"s, which will cause problem when calculating gradients.
            # Thus, we mask these positions after softmax
            ptr_self_attn_mask_post_softmax = ptr_self_attn_mask.new_ones(*ptr_self_attn_mask.size()[:2], 1,
                                                                          dtype=torch.float)
            ptr_self_attn_mask_post_softmax[ptr_self_attn_mask.sum(dim=2) == 0] = 0
            # we need to modify the pre-softmax as well, since after we get nan, multiplying by 0 is still nan
            ptr_self_attn_mask[(ptr_self_attn_mask.sum(dim=2, keepdim=True) == 0).
                               repeat(1, 1, tgt_actnode_masks.size(1))] = 1
            # NOTE must use torch.bool for mask for PyTorch >= 1.2, otherwise there will be problems around ~mask
            # for compatibility of PyTorch 1.1
            if version.parse(torch.__version__) < version.parse('1.2.0'):
                ptr_self_attn_mask = (ptr_self_attn_mask, ptr_self_attn_mask_post_softmax)
            else:
                ptr_self_attn_mask = (ptr_self_attn_mask.to(torch.bool), ptr_self_attn_mask_post_softmax)
        else:
            ptr_self_attn_mask = None
        # ========================================================================
        # import pdb; pdb.set_trace()

        # TODO tgt_src_align_layers are not really controlled!!!

        # decoder layers
        for layer_index, layer in enumerate(self.layers):

            # Encode state of state machine as attention masks and encoded
            # token positions changing for each target action
            if self.encode_state_machine is None:
                head_attention_masks = None
                head_positions = None
            else:
                raise NotImplementedError('Deprecated: use stack-transformer branch')

            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                head_attention_masks=head_attention_masks,
                head_positions=head_positions,
                cross_attention_mask=(cross_attention_mask
                                      if layer_index in self.args.tgt_src_align_layers
                                      else None),
                ptr_self_attn_mask=(ptr_self_attn_mask
                                    if layer_index in self.args.pointer_dist_decoder_selfattn_layers
                                    else None)
            )
            inner_states.append(x)

            if layer_index not in self.args.pointer_dist_decoder_selfattn_layers:
                continue

            # attn is tgt self-attention of size (bsz, num_heads, tgt_len, tgt_len) with future masks
            if self.args.pointer_dist_decoder_selfattn_heads == 1:
                attn = attn[:, 0, :, :]
                attn_all.append(attn)
            else:
                attn = attn[:, :self.args.pointer_dist_decoder_selfattn_heads, :, :]
                if self.args.pointer_dist_decoder_selfattn_avg == 1:
                    # arithmetic mean
                    attn = attn.sum(dim=1) / self.args.pointer_dist_decoder_selfattn_heads
                    attn_all.append(attn)
                elif self.args.pointer_dist_decoder_selfattn_avg == 0:
                    # geometric mean
                    attn = attn.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                    # TODO there is an nan bug when backward for the above power
                    attn_all.append(attn)
                elif self.args.pointer_dist_decoder_selfattn_avg == -1:
                    # no mean
                    pointer_dists = list(map(lambda x: x.squeeze(1),
                                             torch.chunk(attn, self.args.pointer_dist_decoder_selfattn_heads, dim=1)))
                    # for decoding: using a single pointer distribution
                    attn = attn.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                    attn_all.extend(pointer_dists)
                else:
                    raise ValueError

        # for decoding: which pointer distribution to use
        attn = attn_all[self.args.pointer_dist_decoder_selfattn_layers.index(
            self.args.pointer_dist_decoder_selfattn_infer)]

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # NOTE here 'attn' is used for inference pointer prediction, 'attn_all' is used for loss calculation
        # TODO change the names to be more straightforward, such as 'pointer_dist_infer', 'pointer_dist_list'
        # TODO add teacher forcing; this will change the backward behavior
        return x, {'attn': attn, 'inner_states': inner_states, 'attn_all': attn_all}

    def output_layer(self, features, logits_mask=None, logits_indices=None,
                     tgt_vocab_masks=None,
                     **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                emb_weights = self.embed_tokens.weight
            else:
                if not self.args.tgt_factored_emb_out:
                    emb_weights = self.embed_out
                else:
                    # factored embeddings for target actions on the decoder output side
                    emb_weights = self.factored_embeddings(self.dictionary_arange).squeeze(0)
            if logits_indices:

                # indices of active logits
                indices = torch.tensor(list(logits_indices.keys()))
                # compute only active logits
                # (batch_size, target_size, target_emb_size)
                active_output = F.linear(features, emb_weights[indices, :])
                # forbid masked elements
                active_output[logits_mask == 0] = float("-Inf")
                # assign output
                emb_size = emb_weights.shape[0]
                batch_size, target_size, _ = features.shape
                out_shape = (batch_size, target_size, emb_size)
                output = features.new_ones(out_shape) * float("-Inf")
                output[:, :, indices] = active_output

            else:
                output = F.linear(features, emb_weights)

            # TODO fix this when decoding
            # if args is not None and args.apply_tgt_vocab_masks:
            # import pdb; pdb.set_trace()
            if self.args.apply_tgt_vocab_masks:
                assert tgt_vocab_masks is not None
                output[tgt_vocab_masks == 0] = float('-inf')
        else:
            assert not logits_mask
            output = features

        return output

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(
                self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


# @register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


# @register_model_architecture('transformer', 'transformer_iwslt_de_en')
# def transformer_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_wmt_en_de')
# def transformer_wmt_en_de(args):
#     base_architecture(args)


# # parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
# def transformer_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.3)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
# def transformer_vaswani_wmt_en_fr_big(args):
#     args.dropout = getattr(args, 'dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# @register_model_architecture('transformer', 'transformer_wmt_en_de_big')
# def transformer_wmt_en_de_big(args):
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# # default parameters used in tensor2tensor implementation
# @register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
# def transformer_wmt_en_de_big_t2t(args):
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
#     args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# @register_model_architecture('transformer', 'transformer_2x2')
# def transformer_2x2(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', None)
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 2)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_6x6')
# def transformer_6x6(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', None)
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_3x8')
# def transformer_3x8(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', None)
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 3)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 8)
#     base_architecture(args)


# # Stack-Transformer code


# @register_model_architecture('transformer', 'stack_transformer_2x2_layer0')
# def stack_transformer_2x2_layer0(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "layer0")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 2)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6_layer0')
# def stack_transformer_6x6_layer0(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "layer0")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_2x2_nopos_layer0')
# def stack_transformer_2x2_nopos_layer0(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "layer0_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 2)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_2x2_nopos')
# def stack_transformer_2x2_nopos(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "all-layers_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 2)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6')
# def stack_transformer_6x6(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "all-layers")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6_nopos')
# def stack_transformer_6x6_nopos(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "all-layers_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6_tops_nopos')
# def stack_transformer_6x6_tops_nopos(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6_only_buffer_nopos')
# def stack_transformer_6x6_only_buffer_nopos(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "only_buffer_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# @register_model_architecture('transformer', 'stack_transformer_6x6_only_stack_nopos')
# def stack_transformer_6x6_only_stack_nopos(args):
#     args.encode_state_machine = getattr(args, 'encode_state_machine', "only_stack_nopos")
#     #
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 6)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 6)
#     base_architecture(args)


# pointer transformer


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer_3x3')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer_2x2')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer_1x3')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer_3x3_2h')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)


@register_model_architecture('transformer_tgt_pointer', 'transformer_tgt_pointer_3x3_3h')
def transformer_pointer(args):
    # args.encode_state_machine = getattr(args, 'encode_state_machine', "stack_top_nopos")
    args.encode_state_machine = getattr(args, 'encode_state_machine', None)
    #
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 192)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 192)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 3)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    # additional control of whether to use various action states information in the model
    args.apply_tgt_vocab_masks = getattr(args, 'apply_tgt_vocab_masks', 0)
    args.apply_tgt_actnode_masks = getattr(args, 'apply_tgt_actnode_masks', 0)
    args.apply_tgt_src_align = getattr(args, 'apply_tgt_src_align', 1)
    args.apply_tgt_input_src = getattr(args, 'apply_tgt_input_src', 0)
    # target source alignment masks for decoder cross-attention
    args.tgt_src_align_layers = getattr(args, 'tgt_src_align_layers', list(range(args.decoder_layers)))
    args.tgt_src_align_heads = getattr(args, 'tgt_src_align_heads', 1)
    args.tgt_src_align_focus = getattr(args, 'tgt_src_align_focus', ['p0c1n0'])
    # pointer distribution from decoder self-attention
    args.pointer_dist_decoder_selfattn_layers = getattr(args, 'pointer_dist_decoder_selfattn_layers',
                                                        list(range(args.decoder_layers)))
    args.pointer_dist_decoder_selfattn_heads = getattr(args, 'pointer_dist_decoder_selfattn_heads',
                                                       args.decoder_attention_heads)
    args.pointer_dist_decoder_selfattn_avg = getattr(args, 'pointer_dist_decoder_selfattn_avg', 1)
    args.pointer_dist_decoder_selfattn_infer = getattr(args, 'pointer_dist_decoder_selfattn_infer',
                                                       args.pointer_dist_decoder_selfattn_layers[-1])
    # combine source token embeddings into action embeddings for decoder input for node representation
    args.tgt_input_src_emb = getattr(args, 'tgt_input_src_emb', 'top')
    args.tgt_input_src_backprop = getattr(args, 'tgt_input_src_backprop', 1)
    args.tgt_input_src_combine = getattr(args, 'tgt_input_src_combine', 'cat')
    # target factored embeddings
    args.tgt_factored_emb_out = getattr(args, 'tgt_factored_emb_out', 0)

    # process some of the args for compatibility issue with legacy versions
    if isinstance(args.tgt_src_align_focus, str):
        assert len(args.tgt_src_align_focus) == 4, 'legacy version has "p-n-" format'
        args.tgt_src_align_focus = [args.tgt_src_align_focus[:2] + 'c1' + args.tgt_src_align_focus[-2:]]
    elif isinstance(args.tgt_src_align_focus, list):
        if len(args.tgt_src_align_focus) == 1 and len(args.tgt_src_align_focus[0]) == 4:
            args.tgt_src_align_focus[0] = args.tgt_src_align_focus[0][:2] + 'c1' + args.tgt_src_align_focus[0][-2:]
    else:
        raise TypeError

    base_architecture(args)
