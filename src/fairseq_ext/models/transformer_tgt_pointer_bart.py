# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    # TransformerDecoderLayer,
    # TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from torch_scatter import scatter_mean

from ..modules.transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from .attention_masks import get_cross_attention_mask_heads
from ..extract_bart.composite_embeddings import CompositeEmbeddingBART


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_tgt_pointer_bart")
class TransformerTgtPointerBARTModel(FairseqEncoderDecoderModel):
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

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

        # (below borrowed from https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/fairseq/models/bart/model.py#L41)
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

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
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        # parser.add_argument('--share-decoder-input-output-embed', action='store_true',
        #                     help='share decoder input and output embeddings')
        parser.add_argument('--share-decoder-input-output-embed', type=int,
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
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')

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
        # additional: bart gradient control & component control
        parser.add_argument('--bart-encoder-backprop', type=int,
                            help='whether to backprop through BART encoder')
        parser.add_argument('--bart-emb-backprop', type=int,
                            help='whether to backprop through BART embeddings')
        parser.add_argument('--bart-emb-decoder', type=int,
                            help='whether to use BART bpe dictionary embeddings for the decoder (compositionally)')
        parser.add_argument('--bart-emb-decoder-input', type=int,
                            help='whether to use BART bpe dictionary embeddings for the decoder input (compositionally')
        parser.add_argument('--bart-emb-init-composition', type=int,
                            help='whether to initialize the decoder embeddings with composed sub-tokens '
                                 'from BART vocabulary')
        parser.add_argument('--bart-emb-composition-pred', type=int,
                            help='whether to use the compositional embedding on top of BART embeddings only for the '
                                 'PRED node actions')
        # additional: use pretrained roberta embeddings
        parser.add_argument('--src-roberta-emb', type=int,
                            help='src tokens: whether to use pretrained (fix) RoBERTa embeddings to input to encoder')
        parser.add_argument('--src-pool-wp2w', type=str, choices=['none', 'bot', 'top'],
                            help='src tokens: where to pool the wordpiece embeddings to words; '
                                 '"none": keep the wordpiece embeddings; '
                                 '"bot": pool at the bottom of the encoder; '
                                 '"top": pool at the top of the encoder.'
                                 'NOTE this removes the BOS <s> and EOS </s> positions.')
        parser.add_argument('--src-avg-layers', type=int, nargs='*',
                            help='average encoder layers for src contextual embeddings')
        parser.add_argument('--src-roberta-enc', type=int,
                            help='whether to use RoBERTa encoder to replace BART encoder')
        parser.add_argument('--src-roberta-enc-size', type=str,
                            help='size of RoBERTa model to replace BART encoder')
        # additional: structure control
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

        # customized for tgt pointer: make sure all arguments are present in older models during development
        transformer_pointer(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        # ========== at inference time, update the auxiliary BART model if it is bart.large as a workaround
        # NOTE reason: `arch` is not input from outside to `args` from outside at inference time; `args` is loaded from
        # checkpoint to initialize the model, thus containing `arch`.
        # but `task.bart` is initialized first before the checkpoint is loaded with model `args`
        if 'bart_large' in args.arch:
            print('-' * 10 + ' task bart rewind: loading pretrained bart.large model ' + '-' * 10)
            bart = torch.hub.load('pytorch/fairseq', 'bart.large')
            if 'initialize_with_watbart' in args.__dict__.keys() and args.initialize_with_watbart is not None:
                try:
                    bart_local = torch.load(args.initialize_with_watbart)
                    bart.model.load_state_dict(bart_local['model'])
                except Exception:
                    raise ValueError("the specified path at initialize_with_watbart \
                        is incorrect; please double-check config file.")
            task.bart = bart
            task.bart_dict = bart.task.target_dictionary    # src dictionary is the same

        if 'roberta_base' in args.arch:
            print('-' * 10 + ' task bart rewind: loading pretrained roberta.base model ' + '-' * 10)
            bart = torch.hub.load('pytorch/fairseq', 'roberta.base')
            task.bart = bart
            task.bart_dict = bart.task.target_dictionary    # src dictionary is the same

        if 'roberta_large' in args.arch:
            print('-' * 10 + ' task bart rewind: loading pretrained roberta.large model ' + '-' * 10)
            bart = torch.hub.load('pytorch/fairseq', 'roberta.large')
            task.bart = bart
            task.bart_dict = bart.task.target_dictionary    # src dictionary is the same
        # ================================================================================================

        # overwrite the src and tgt dictionary used to build learnable embeddings
        # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        src_dict = tgt_dict = task.bart_dict
        src_dict_raw, tgt_dict_raw = task.source_dictionary, task.target_dictionary

        # control how the tgt embeddings are built
        if not args.bart_emb_decoder:
            assert not args.share_all_embeddings, 'use separate embeddings for tgt -> can not be shared with src'

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding( 
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            if not args.bart_emb_decoder:
                # separate embedding for the tgt actions
                if not args.bart_emb_decoder_input:
                    # do not use the BART embedding for target input and output
                    decoder_embed_tokens = cls.build_embedding(
                        args, tgt_dict_raw, args.decoder_embed_dim, args.decoder_embed_path
                    )
                else:
                    # still use the BART embedding for target input
                    # NOTE this below is not tying the decoder embeddings with encoder embeddings
                    # decoder_embed_tokens = cls.build_embedding(
                    #     args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                    # )
                    decoder_embed_tokens = encoder_embed_tokens
            else:
                assert args.bart_emb_decoder_input
                # compositional embeddings for the tgt actions on top of BART bpe embeddings
                decoder_embed_tokens = cls.build_embedding(
                    args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, tgt_dict_raw, task.bart,
                                    encoder_embed_tokens    # for args.decoder_emb_composition_pred
                                    )
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, tgt_dict_raw, bart, encoder_embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            dictionary_raw=tgt_dict_raw,
            bart=bart,
            encoder_embed_tokens=encoder_embed_tokens
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        # customized
        src_fix_emb: Optional[torch.Tensor] = None,
        src_wordpieces: torch.Tensor = None,
        src_wp2w: torch.Tensor = None,
        tgt_vocab_masks: torch.Tensor = None,
        tgt_actnode_masks: torch.Tensor = None,
        tgt_src_cursors: torch.Tensor = None,
        # unused
        **unused
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            # customized
            src_fix_emb=src_fix_emb,
            src_wordpieces=src_wordpieces,
            src_wp2w=src_wp2w
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            # customized
            tgt_vocab_masks=tgt_vocab_masks,
            tgt_actnode_masks=tgt_actnode_masks,
            tgt_src_cursors=tgt_src_cursors
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


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
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # customized
        self.bart_encoder_backprop = args.bart_encoder_backprop
        # fix pretrained bart embeddings
        if not args.bart_emb_backprop:
            self.embed_tokens.weight.requires_grad = False

        # use fixed pretrained embeddings
        self.src_roberta_emb = args.src_roberta_emb
        if args.src_roberta_emb:
            self.fix_emb_proj = Linear(args.pretrained_embed_dim, embed_dim, bias=False)

        self.src_pool_wp2w = args.src_pool_wp2w

        # average encoder layers
        self.src_avg_layers = args.src_avg_layers

        # use roberta encoder to replace bart encoder directly
        self.src_roberta_enc = args.src_roberta_enc
        if args.src_roberta_enc:
            print('-' * 10 + ' loading RoBERTa base model to replace BART encoder directly ' + '-' * 10)
            self.roberta = torch.hub.load('pytorch/fairseq', f'roberta.{args.src_roberta_enc_size}')
            if not args.bart_encoder_backprop:
                self.roberta.eval()
            assert args.src_pool_wp2w == 'top', 'currently for RoBERTa encoder we only support pooling to words on top'

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def pool_wp2w(self, x, src_tokens, src_wordpieces, src_wp2w):
        """Pool the wordpieces to words, with src left padding.
        NOTE wp2w does not have <s> and </s> considered, whereas wordpieces and x have them.

        Args:
            x (torch.Tensor): hidden states of size (T, B, C)
            src_tokens (torch.Tensor): token indexes based on the word vocabulary for x,
                of size (B, T') (only need the batch max length information here)
            src_wordpieces (torch.Tensor): wordpiece token indexes in the bpe vocabulary for x,
                of size (B, T'')
            src_wp2w (torch.Tensor): wordpieces to word mapping, reversed, left padding,
                of size (B, T''')

        Returns:
            src_pooled (torch.Tensor): pooled hidden states, of size (T', B, C)
        """
        x = x.transpose(0, 1)

        # remove sentence start <s>
        bsize, max_len, emb_size = x.shape
        mask = (src_wordpieces != 0).unsqueeze(2).expand(x.shape)    # TODO 0 for <s> is fixed here
        x = x[mask].view((bsize, max_len - 1, emb_size))
        # remove sentence end
        x = x[:, :-1, :]

        # apply scatter
        # `src_wp2w` was inverted in pre-processing to account for src left side padding, to easily
        # add numbers at the beginning when collating to a batch, so that
        # each <pad> token pooled by itself, e.g.
        # [45 (<pad>), 44 (<pad>), 43 (<pad>), 42 (<pad>), 41 (useful), 41, 40, 40, 40, 39, 39, ...]
        src_pooled = scatter_mean(
            x,
            src_wp2w.unsqueeze(2),
            dim=1
        )
        # after the scatter, the orders are reversed due to the index values in `src_wp2w`
        # We need to flip the result to recover the original order, with the beginning being the <pad>
        src_pooled = src_pooled.flip(1)

        # Remove extra padding at the beginning
        src_pooled = src_pooled[:, -src_tokens.shape[1]:, :]

        # recover the original dim
        src_pooled = src_pooled.transpose(0, 1)

        return src_pooled

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_fix_emb: Optional[torch.Tensor] = None,
        src_wordpieces: torch.Tensor = None,
        src_wp2w: torch.Tensor = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        # unused
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # ========== use RoBERTa (base) encoder instead of BART encoder ==========
        if self.src_roberta_enc:
            # directly use RoBERTa encoder
            x = self.roberta.extract_features(src_wordpieces)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            # pool the hidden states from bpe to original word
            x = self.pool_wp2w(x, src_tokens, src_wordpieces, src_wp2w)
            # update the padding mask
            encoder_padding_mask = src_tokens.eq(self.padding_idx)

            if not self.bart_encoder_backprop:
                x = x.detach()

            return EncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask,  # B x T
                encoder_embedding=None,  # B x T x C    # NOTE since we pool, T here would be inconsistent
                encoder_states=None,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )
        # ===================================================================

        if not self.src_roberta_emb:
            # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

            # we do not use the vocabulary built on the fly; we use the pretrained vocabulary
            # `src_wordpieces` is the actual input here (instead of `src_tokens`)
            x, encoder_embedding = self.forward_embedding(src_wordpieces, token_embeddings)
        else:
            # use pretrained fix RoBERTa embeddings
            x = src_fix_emb
            x = self.fix_emb_proj(x)    # to map the dimension of the encoder
            encoder_embedding = None
            # further process
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
            if self.quant_noise is not None:
                x = self.quant_noise(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self.src_pool_wp2w == 'bot':
            # pool the hidden states from bpe to original word
            x = self.pool_wp2w(x, src_tokens, src_wordpieces, src_wp2w)
            # compute padding mask
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
        elif self.src_pool_wp2w == 'top':
            # compute padding mask
            # encoder_padding_mask = src_tokens.eq(self.padding_idx)
            # NOTE `src_tokens` -> `src_wordpieces`
            encoder_padding_mask = src_wordpieces.eq(self.padding_idx)
        elif self.src_pool_wp2w == 'none':
            # NOTE `src_tokens` -> `src_wordpieces`
            encoder_padding_mask = src_wordpieces.eq(self.padding_idx)
            raise NotImplementedError('need to take care of the alignment indexes')
        else:
            raise ValueError

        if self.src_avg_layers:
            return_all_hiddens = True

        encoder_states = [] if return_all_hiddens else None

        # breakpoint()

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        # average across layers
        if self.src_avg_layers:
            # e.g. [1, 2, 3, 4, 5, 6]
            x_layers = [encoder_states[i - 1] for i in self.src_avg_layers]
            x = sum(x_layers) / len(self.src_avg_layers)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.src_pool_wp2w == 'top':
            # pool the hidden states from bpe to original word
            x = self.pool_wp2w(x, src_tokens, src_wordpieces, src_wp2w)
            # update the padding mask
            encoder_padding_mask = src_tokens.eq(self.padding_idx)

        if not self.bart_encoder_backprop:
            x = x.detach()

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C    # NOTE since we pool, T here would be inconsistent
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
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

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, dictionary_raw=None, bart=None,
                 encoder_embed_tokens=None):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            if not args.bart_emb_decoder:
                # separate embedding for the tgt actions (output embedding here)
                self.output_projection = nn.Linear(
                    self.output_embed_dim, len(dictionary_raw), bias=False
                )
            else:
                # compositional embeddings for the tgt actions on top of BART bpe embeddings
                self.output_projection = nn.Linear(
                    self.output_embed_dim, len(dictionary), bias=False
                )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        # ========== pooling embedding ==========
        self.dictionary_raw = dictionary_raw
        if not args.bart_emb_decoder:
            if not args.bart_emb_decoder_input:
                self.composite_embed = None
            else:
                self.composite_embed = CompositeEmbeddingBART(bart, self.embed_tokens, dictionary_raw)

            if args.bart_emb_composition_pred:
                # use compositional embeddings for PRED node actions
                self.composite_embed = CompositeEmbeddingBART(bart, encoder_embed_tokens, dictionary_raw)
        else:
            self.composite_embed = CompositeEmbeddingBART(bart, self.embed_tokens, dictionary_raw)
        # =======================================

        # fix pretrained bart embeddings
        if not args.bart_emb_backprop and args.bart_emb_decoder:
            assert args.bart_emb_decoder, 'must use pre-trained BART embeddings to fix the decoder embedding parameters'
            self.embed_tokens.weight.requires_grad = False

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        # customized: specific to APT
        tgt_vocab_masks=None,
        tgt_actnode_masks=None,
        tgt_src_cursors=None,
        # unused
        **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            # customized for APT
            tgt_src_cursors=tgt_src_cursors,
            tgt_actnode_masks=tgt_actnode_masks
        )
        if not features_only:
            x = self.output_layer(
                x,
                # customized for APT
                tgt_vocab_masks=tgt_vocab_masks
            )
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        # customized for APT
        tgt_src_cursors=None,
        tgt_actnode_masks=None
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            # customized for APT
            tgt_src_cursors=tgt_src_cursors,
            tgt_actnode_masks=tgt_actnode_masks
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        # customized for APT
        tgt_src_cursors=None,
        tgt_actnode_masks=None
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # ========== apply composite embeddings on the raw tgt tokens ==========
        if not self.args.bart_emb_decoder:

            # use the compositional embedding for PRED node actions
            if self.args.bart_emb_composition_pred:
                self.composite_embed.update_embeddings()

                embedding_weight_mixed = torch.zeros_like(self.embed_tokens.weight)
                embedding_weight_mixed[self.composite_embed.dict_pred_mask] = self.composite_embed.embedding_weight[
                    self.composite_embed.dict_pred_mask]
                embedding_weight_mixed[~self.composite_embed.dict_pred_mask] = self.embed_tokens.weight[
                    ~self.composite_embed.dict_pred_mask]

                x = self.embed_scale * nn.functional.embedding(prev_output_tokens,
                                                               embedding_weight_mixed,
                                                               padding_idx=self.dictionary.pad())
            else:
                # separate embeddings
                if not self.args.bart_emb_decoder_input:
                    x = self.embed_scale * self.embed_tokens(prev_output_tokens)
                else:
                    x = self.embed_scale * self.composite_embed(prev_output_tokens, update=True)

        else:
            # compositional embeddings based on BART embeddings
            x = self.embed_scale * self.composite_embed(prev_output_tokens, update=True)

        # ======================================================================

        # ========== combine the corresponding source token embeddings with the action embeddings as input ==========
        if self.args.apply_tgt_input_src:
            assert self.args.tgt_input_src_emb == 'top' and self.args.tgt_input_src_combine == 'add', \
                'currently we do not support other variations (which may have a bit of extra parameters'

            # 1) take out the source embeddings
            src_embs = encoder_out.encoder_out.transpose(0, 1)    # size (batch_size, src_max_len, encoder_emb_dim)
            if not self.args.tgt_input_src_backprop:
                src_embs = src_embs.detach()

            # 2) align the source embeddings to the tgt input actions
            assert tgt_src_cursors is not None
            tgt_src_index = tgt_src_cursors.clone()    # size (bsz, tgt_max_len)
            if encoder_out.encoder_padding_mask is not None:
                src_num_pads = encoder_out.encoder_padding_mask.sum(dim=1, keepdim=True)
                tgt_src_index = tgt_src_index + src_num_pads    # NOTE this is key to left padding!

            # NOTE due to padding value is 1, the indexes could be out of range of src_max_len ->
            #      we fix invalid indexes for padding positions (invalid should only happen at padding positions,
            #      and when the src sentence has max length 1)
            tgt_src_index[tgt_src_index >= src_embs.size(1)] = src_embs.size(1) - 1

            tgt_src_index = tgt_src_index.unsqueeze(-1).repeat(1, 1, src_embs.size(-1))
            # or
            # tgt_src_index = tgt_src_index.unsqueeze(-1).expand(-1, -1, src_embs.size(-1))

            # # NOTE deal with the corner case when the max_src_len in the whole batch is only 1 ->
            # #      already dealt with above!
            # if encoder_out.encoder_out.size(0) == 1:
            #     # NOTE we have to fix all indexes at 0 (including the padding positions)!!
            #     #      (the default padding value is 1, which would cause an index out of range error hard to debug)
            #     tgt_src_index.fill_(0)

            src_embs = torch.gather(src_embs, 1, tgt_src_index)
            # size (bsz, tgt_max_len, src_embs.size(-1))

            # 3) combine the action embeddings with the aligned source token embeddings
            if self.args.tgt_input_src_combine == 'cat':
                x = self.combine_src_embs(torch.cat([src_embs, x], dim=-1))    # NOTE not initialized
            elif self.args.tgt_input_src_combine == 'add':
                x = src_embs + x
            else:
                raise NotImplementedError
        # ===========================================================================================================

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # ========== alignment guidance in the cross-attention: get the mask ==========
        if self.args.apply_tgt_src_align:
            assert tgt_src_cursors is not None
            cross_attention_mask = get_cross_attention_mask_heads(tgt_src_cursors,
                                                                  encoder_out.encoder_out.size(0),
                                                                  encoder_out.encoder_padding_mask,
                                                                  self.args.tgt_src_align_focus,
                                                                  self.args.tgt_src_align_heads,
                                                                  self.layers[0].encoder_attn.num_heads)
        else:
            cross_attention_mask = None
        # ==============================================================================

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        # for pointer distribution
        attn_ptr = None
        attn_all_ptr = []

        # breakpoint()

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # change the decoder layer to output both cross_attention (as in default case)
            # and the decoder self attention
            x, layer_attn, _, self_attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                # customized
                cross_attention_mask=(cross_attention_mask
                                      if idx in self.args.tgt_src_align_layers
                                      else None),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            # ========== for pointer distribution ==========
            if idx not in self.args.pointer_dist_decoder_selfattn_layers:
                continue

            # attn is tgt self-attention of size (bsz, num_heads, tgt_len, tgt_len) with future masks
            if self.args.pointer_dist_decoder_selfattn_heads == 1:
                attn_ptr = self_attn[:, 0, :, :]
                attn_all_ptr.append(attn_ptr)
            else:
                attn_ptr = self_attn[:, :self.args.pointer_dist_decoder_selfattn_heads, :, :]
                if self.args.pointer_dist_decoder_selfattn_avg == 1:
                    # arithmetic mean
                    attn_ptr = attn_ptr.sum(dim=1) / self.args.pointer_dist_decoder_selfattn_heads
                    attn_all_ptr.append(attn_ptr)
                elif self.args.pointer_dist_decoder_selfattn_avg == 0:
                    # geometric mean
                    attn_ptr = attn_ptr.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                    # TODO there is an nan bug when backward for the above power
                    attn_all_ptr.append(attn_ptr)
                elif self.args.pointer_dist_decoder_selfattn_avg == -1:
                    # no mean
                    pointer_dists = list(map(
                        lambda x: x.squeeze(1),
                        torch.chunk(attn_ptr, self.args.pointer_dist_decoder_selfattn_heads, dim=1)))
                    # for decoding: using a single pointer distribution
                    attn_ptr = attn_ptr.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                    attn_all_ptr.extend(pointer_dists)
                else:
                    raise ValueError

        # for decoding: which pointer distribution to use
        attn_ptr = attn_all_ptr[self.args.pointer_dist_decoder_selfattn_layers.index(
            self.args.pointer_dist_decoder_selfattn_infer)]

        # ====================================================

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # NOTE here 'attn_ptr' is used for inference pointer prediction, 'attn_all_ptr' is used for loss calculation
        # TODO change the names to be more straightforward, such as 'pointer_dist_infer', 'pointer_dist_list'
        # TODO add teacher forcing; this will change the backward behavior
        # change the original output TODO change the names to include both original `attn` and that for pointer
        # return x, {"attn": [attn], "inner_states": inner_states}
        return x, {'attn': attn_ptr, 'inner_states': inner_states, 'attn_all': attn_all_ptr}

    def output_layer(self, features, tgt_vocab_masks=None):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            # out = self.output_projection(features)    # original output_projection has a different output size

            # use the composite embeddings
            if not self.args.bart_emb_decoder:
                # separate embeddings
                out = self.output_projection(features)    # original output_projection has a different output size

                # use the compositional embedding for PRED node actions
                if self.args.bart_emb_composition_pred:
                    out_comp = nn.functional.linear(features, self.composite_embed.embedding_weight)
                    out[:, :, self.composite_embed.dict_pred_mask] = out_comp[:, :, self.composite_embed.dict_pred_mask]
            else:
                # compositional embeddings based on BART embeddings
                out = nn.functional.linear(features, self.composite_embed.embedding_weight)

            if self.args.apply_tgt_vocab_masks:
                assert tgt_vocab_masks is not None
                out[tgt_vocab_masks == 0] = float('-inf')
            return out
        else:
            assert not self.args.apply_tgt_vocab_masks
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
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
        nn.init.constant_(m.bias, 0.0)
    return m


# @register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


# @register_model_architecture("transformer", "transformer_iwslt_de_en")
# def transformer_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     base_architecture(args)


# @register_model_architecture("transformer", "transformer_wmt_en_de")
# def transformer_wmt_en_de(args):
#     base_architecture(args)


# # parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
# @register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
# def transformer_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
#     args.dropout = getattr(args, "dropout", 0.3)
#     base_architecture(args)


# @register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
# def transformer_vaswani_wmt_en_fr_big(args):
#     args.dropout = getattr(args, "dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# @register_model_architecture("transformer", "transformer_wmt_en_de_big")
# def transformer_wmt_en_de_big(args):
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# # default parameters used in tensor2tensor implementation
# @register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
# def transformer_wmt_en_de_big_t2t(args):
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# @register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer")
def transformer_pointer(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

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
    # BART control
    args.bart_encoder_backprop = getattr(args, 'bart_encoder_backprop', 1)
    args.bart_emb_backprop = getattr(args, 'bart_emb_backprop', 1)
    args.bart_emb_decoder = getattr(args, 'bart_emb_decoder', 1)

    if not args.bart_emb_decoder:
        # explicitly update setup to guarantee consistency
        args.share_all_embeddings = False

    args.bart_emb_decoder_input = getattr(args, 'bart_emb_decoder_input', None)
    if args.bart_emb_decoder_input is None:
        if not args.bart_emb_decoder:
            # default to not using compositional BART embeddings for decoder input
            args.bart_emb_decoder_input = 0
        else:
            # use compositional BART embeddings for both decoder input and output
            args.bart_emb_decoder_input = 1

    if not args.bart_emb_decoder and args.bart_emb_decoder_input:
        # explicitly update setup to guarantee consistency
        args.share_decoder_input_output_embed = False

    args.bart_emb_init_composition = getattr(args, 'bart_emb_init_composition', 0)

    # RoBERTa embedding and pooling
    args.src_roberta_emb = getattr(args, 'src_roberta_emb', 0)
    args.src_pool_wp2w = getattr(args, 'src_pool_wp2w', 'top')

    args.src_avg_layers = getattr(args, 'src_avg_layers', None)

    args.src_roberta_enc = getattr(args, 'src_roberta_enc', 0)
    args.src_roberta_enc_size = getattr(args, 'src_roberta_enc_size', 'large')

    # whether to use compositional embeddings on top of BART embeddings for the PRED node actions
    args.bart_emb_composition_pred = getattr(args, 'bart_emb_composition_pred', 0)
    if args.bart_emb_composition_pred:
        assert not args.bart_emb_decoder, ('args.bart_emb_composition_pred is used when we do not use BART embeddings '
                                           'for decoder')

    base_architecture(args)


@register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer_bart_large")
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    if args.share_decoder_input_output_embed:
        args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    else:
        args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    # args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    # args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    transformer_pointer(args)


@register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer_bart_base")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    transformer_pointer(args)    # for explicit showing, but actually redundant


@register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer_roberta_large_24x12")
def roberta_large_24x12_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    # NOTE above doesn't matter as we initialize RoBERTa inside model (actual encoder_layer is 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    bart_large_architecture(args)
    transformer_pointer(args)    # for explicit showing, but actually redundant


@register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer_roberta_large_24x3")
def roberta_large_24x3_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    # NOTE above doesn't matter as we initialize RoBERTa inside model (actual encoder_layer is 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    bart_large_architecture(args)
    transformer_pointer(args)    # for explicit showing, but actually redundant

@register_model_architecture("transformer_tgt_pointer_bart", "transformer_tgt_pointer_apt2_mini")
def bart_apt2_mini_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 48)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 48)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    bart_large_architecture(args)
    transformer_pointer(args)    # for explicit showing, but actually redundant