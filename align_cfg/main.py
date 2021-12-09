# -*- coding: utf-8 -*-

import argparse
import collections
import copy
import hashlib
import json
import os
import sys

# TODO: This can be set in the config
os.environ['DGLBACKEND'] = 'pytorch'

try:
    import dgl
    has_dgl = True
except ImportError:
    print('Warning: To use DGL, install.')
    has_dgl = False

try:
    from torch_geometric.data import Batch, Data
    has_geometric = True
except ImportError:
    print('Warning: To use GCN install pytorch geometric.')
    has_geometric = False

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from align_utils import save_align_dist
from amr_utils import convert_amr_to_tree, get_tree_edges, compute_pairwise_distance, get_node_ids
from amr_utils import safe_read as safe_read_
from alignment_decoder import AlignmentDecoder
from evaluation import EvalAlignments
from formatter import amr_to_pretty_format, amr_to_string
from gcn import GCNEncoder
from pretrained_embeddings import read_embeddings, read_amr_vocab_file, read_text_vocab_file
from transition_amr_parser.io import read_amr2
from transformer_lm import BiTransformer, TransformerModel
from tree_lstm import TreeEncoder as TreeLSTMEncoder
from tree_lstm import TreeEncoder_v2 as TreeLSTMEncoder_v2
from tree_rnn import TreeEncoder as TreeRNNEncoder
from vocab import *
from vocab_definitions import MaskInfo


def safe_read(path, **kwargs):
    # TODO: Rename to --jamr or similar
    if args.aligner_training_and_eval:
        # read JAMR
        kwargs['ibm_format'], kwargs['tokenize'] = True, False
    else:
        # read PENMAN
        kwargs['ibm_format'], kwargs['tokenize'] = False, False

    return safe_read_(path, **kwargs)


class JSONConfig(object):
    """
    Command-line parser for JSON serializable objects.
    """
    def __init__(self, default_config):
        self.default_config = default_config

    def parse(self, arg):
        config = self.default_config
        if arg is not None:
            for k, v in json.loads(arg).items():
                config[k] = v
        return config


def argument_parser():

    parser = argparse.ArgumentParser()
    # Single input parameters
    parser.add_argument(
        "--cache-dir",
        help="Folder to store intermediate aligner outputs e.g. pre-computed token embeddings.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--trn-amr",
        help="AMR input file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--val-amr",
        help="AMR input file.",
        action='append',
        default=None
    )
    parser.add_argument(
        "--tst-amr",
        help="AMR input file.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--single-input",
        help="AMR input file.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--single-output",
        help="AMR output file.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--vocab-text",
        help="Vocab file.",
        type=str,
        required=True,
        # default='./align_cfg/vocab.text.2021-06-30.txt'
    )
    parser.add_argument(
        "--vocab-amr",
        help="Vocab file.",
        type=str,
        required=True,
        # default='./align_cfg/vocab.amr.2021-06-30.txt'
    )
    parser.add_argument(
        "--log-dir",
        help="Path to log directory.",
        default="log/demo",
        type=str,
    )
    parser.add_argument(
        "--home",
        help="Used to specify default file paths.",
        # default="/dccstor/ykt-parse/SHARED/misc/adrozdov",
        type=str,
    )
    # Model options
    parser.add_argument(
        "--model-config",
        help="JSON serializable string for model config.",
        default=None,
        type=str,
    )
    # Training hyperparams
    parser.add_argument(
        "--lr",
        help="Learning rate.",
        default=2e-3,
        type=float,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--accum-steps",
        help="Gradient accumulation steps.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--max-epoch",
        help="Max number of training epochs.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--mask",
        help="Chance to mask input token.",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--mask-at-inference",
        help="If true, then mask tokens one at a time at inference.",
        action='store_true',
    )
    parser.add_argument(
        "--no-mask-at-inference",
        help="If true, then mask tokens one at a time at inference.",
        action='store_true',
    )
    # Output options
    parser.add_argument(
        "--write-validation",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--write-pretty",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--write-align-dist",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--write-only",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--write-single",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--no-jamr",
        help="If true, then read penman. Otherwise, read JAMR.",
        action='store_true',
    )
    parser.add_argument(
        "--aligner-training-and-eval",
        help="Set when training or evaluating aligner.",
        action='store_true',
    )
    # Other options
    parser.add_argument(
        "--load",
        help="Path to model checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--load-flags",
        help="Path to model flags.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--max-length",
        help="Max number of text tokens.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--val-max-length",
        help="Max number of text tokens.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda",
        help="If true, then use GPU.",
        action='store_true',
    )
    # Debug options
    parser.add_argument(
        "--debug",
        action='store_true',
    )
    parser.add_argument(
        "--demo",
        help="If true, then print progress bars.",
        action='store_true',
    )
    parser.add_argument(
        "--allow-cpu",
        help="If true, then allow fallback to CPU if GPU not available.",
        action='store_true',
    )
    parser.add_argument(
        "--verbose",
        help="If true, then print progress bars.",
        action='store_true',
    )
    parser.add_argument(
        "--read-only",
        help="If true, then read AMR and quit.",
        action='store_true',
    )
    parser.add_argument(
        "--short-first",
        help="If true, then shorter inputs arrive first.",
        action='store_true',
    )
    parser.add_argument(
        "--long-first",
        help="If true, then longer inputs arrive first.",
        action='store_true',
    )
    parser.add_argument(
        "--batches-per-epoch",
        default=-1,
        help="Can be used to decrease epoch length.",
        type=int,
    )
    parser.add_argument(
        "--save-every-epoch",
        default=50,
        help="Save every N epochs.",
        type=int,
    )
    parser.add_argument(
        "--skip-validation",
        action='store_true'
    )
    parser.add_argument(
        "--jbsub-eval",
        action='store_true'
    )
    parser.add_argument("--name", default=None, type=str,
                        help="Useful for book-keeping.")
    args = parser.parse_args()

    args.hostname = os.popen("hostname").read().strip()

    if args.allow_cpu and args.cuda:
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available. Falling back to CPU.')
            args.cuda = False

    if args.load_flags:
        with open(args.load_flags) as f:
            data = json.loads(f.read())
        args.model_config = data['model_config']

    if args.write_single:
        assert args.single_input is not None
        assert args.single_output is not None

        args.trn_amr = args.single_input
        args.val_amr = []

    return args


class TextTokenizer(object):
    def __init__(self):
        self.vocab = None
        self.frozen = False

    def set_tokens(self, tokens):
        self.vocab = tokens

    def finalize(self):
        assert self.frozen is False
        print('text tokenizer : vocab = {}'.format(len(self.vocab)))
        self.token_TO_idx = {k: i for i, k in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.token_TO_idx)
        self.frozen = True

    def indexify(self, tokens):
        return [self.token_TO_idx[x] for x in tokens]


class AMRTokenizer(object):
    def __init__(self):
        self.vocab = None
        self.frozen = False

    def dfs(self, amr):

        node_TO_edges = collections.defaultdict(list)
        for x0, label, x1 in amr.edges:
            node_TO_edges[x0].append((label, x1))

        node_ids = get_node_ids(amr)
        node_TO_idx = {k: i for i, k in enumerate(node_ids)}

        # build tree
        g = collections.defaultdict(list)
        g_labels = {}

        safe_edges = get_tree_edges(amr)

        def sortkey(x):
            s, y, t, a, b = x
            return (a, b)

        for e in sorted(safe_edges, key=sortkey):
            s, y, t, a, b = e
            assert a <= b
            g[s].append(t)
            g_labels[(s, t)] = y

        # render tree
        def render(s):
            assert s is not None

            if s not in g:
                node_name = amr.nodes[s]
                node_id = node_TO_idx[s]
                tokens = ['(', node_name, ')']
                token_ids = [-1, node_id, -1]
                return tokens, token_ids

            tokens, token_ids = [], []

            for t in g[s]:
                xtokens, xtoken_ids = render(t)
                y = g_labels[(s, t)]
                tokens += [y] + xtokens
                token_ids += [-1] + xtoken_ids

            node_name = amr.nodes[s]
            node_id = node_TO_idx[s]
            tokens = ['(', node_name] + tokens + [')']
            token_ids = [-1, node_id] + token_ids + [-1]

            return tokens, token_ids

        tokens, token_ids = render(amr.root)

        return tokens, token_ids

    def set_tokens(self, tokens):
        self.vocab = tokens

    def finalize(self):
        assert self.frozen is False
        print('amr tokenizer : vocab = {}'.format(len(self.vocab)))
        self.token_TO_idx = {k: i for i, k in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.token_TO_idx)
        self.frozen = True

    def indexify(self, tokens):
        return [self.token_TO_idx[x] for x in tokens]


class Dataset(object):
    def __init__(self, corpus, text_tokenizer=None, amr_tokenizer=None):
        self.corpus = corpus
        self.text_tokenizer = text_tokenizer
        self.amr_tokenizer = amr_tokenizer
        self.cached = {}

    def get_dgl_graph(self, amr):
        vocab = self.amr_tokenizer.token_TO_idx

        # init graph
        g = dgl.DGLGraph()

        # get tree
        tree = convert_amr_to_tree(amr)

        # add node structure
        node_ids = get_node_ids(amr)
        node_TO_idx = {k: i for i, k in enumerate(node_ids)}
        N = len(node_ids)
        g.add_nodes(N)

        # add node features
        node_labels = [amr.nodes[k] for k in node_ids]
        node_tokens = torch.tensor([vocab[tok] for tok in node_labels], dtype=torch.long)
        g.nodes[:].data['node_tokens'] = node_tokens.view(N)
        g.nodes[:].data['node_ids'] = torch.arange(N)

        # add edge structure
        src, tgt, edge_labels = [], [], []
        for s, t in tree['edges']:
            y = tree['edge_to_label'][(s, t)]
            src.append(node_TO_idx[s])
            tgt.append(node_TO_idx[t])
            edge_labels.append(vocab[y])
        g.add_edges(src, tgt)
        E = len(src)

        # add edge features
        edge_tokens = torch.tensor(edge_labels, dtype=torch.long)
        g.edges[:].data['edge_tokens'] = edge_tokens.view(E)

        # add pairwise node distance
        # pairwise_dist = compute_pairwise_distance(tree)
        pairwise_dist = None

        return g, pairwise_dist

    def get_geometric_data(self, amr):
        vocab = self.amr_tokenizer.token_TO_idx

        node_ids = get_node_ids(amr)
        node_TO_idx = {k: i for i, k in enumerate(node_ids)}

        edge_index = []
        for xin, label, xout in amr.edges:
            a = node_TO_idx[xin]
            b = node_TO_idx[xout]
            edge_index.append([a, b])
            edge_index.append([b, a])

        node_labels = [amr.nodes[k] for k in node_ids]
        node_tokens = torch.tensor([vocab[tok] for tok in node_labels], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(edge_index=edge_index.t().contiguous(), y=node_tokens, num_nodes=len(node_ids))
        return data

    def __getitem__(self, idx):
        if idx in self.cached:
            return self.cached[idx]
        amr = self.corpus[idx]

        item = {}

        # text
        tokens = amr.tokens
        item['text_original_tokens'] = tokens
        item['text_tokens'] = self.text_tokenizer.indexify(tokens)

        # amr

        ## specific for linearized parse.
        tokens, token_ids = self.amr_tokenizer.dfs(amr)
        item['amr_tokens'] = self.amr_tokenizer.indexify(tokens)
        item['amr_node_ids'] = token_ids
        item['amr_node_mask'] = [False if x < 0 else True for x in token_ids]

        ##
        item['amr_nodes'] = get_node_ids(amr)

        if has_dgl:
            g, pairwise_dist = self.get_dgl_graph(amr)
            item['g'] = g

        if has_geometric:
            item['geometric_data'] = self.get_geometric_data(amr)

        self.cached[idx] = item

        return item


def batchify(items, cuda=False, train=False):
    device = torch.cuda.current_device() if cuda else None

    dtypes = {
        'text_tokens': torch.long,
        'amr_tokens': torch.long,
        'amr_node_ids': torch.long,
        'amr_node_mask': torch.bool,
    }

    batch_map = {}
    for k in dtypes.keys():
        if k not in items[0]:
            continue

        if dtypes[k] is list:
            batch_map[k] = [item[k].to(device) for item in items]
            continue

        max_length = max([len(x[k]) for x in items])
        max_length = max(max_length, 5) # Should never be too short...

        val = []
        for x in items:
            v = x[k]
            padding = [PADDING_IDX] * (max_length - len(v))
            bos = [PADDING_IDX]
            val.append(bos + v + padding)

        batch_map[k] = torch.tensor(val, dtype=dtypes[k], device=device)

    batch_map['text_original_tokens'] = [x['text_original_tokens'] for x in items]
    batch_map['items'] = items
    batch_map['device'] = device

    if args.mask > 0 and train:
        batch_mask = []
        for x in items:
            tokens = [MaskInfo.unchanged, MaskInfo.masked]
            probs = [1 - args.mask, args.mask]
            size = len(x['amr_nodes'])
            mask = np.random.choice(tokens, p=probs, size=size)

            # If only 1 item, then never mask.
            if size == 1:
                mask[0] = MaskInfo.unchanged_and_predict

            # If nothing is masked, then always predict at least one item.
            if mask.sum() == 0:
                mask[np.random.randint(0, size)] = MaskInfo.unchanged_and_predict

            assert mask.sum() > 0

            mask = torch.from_numpy(mask).long()
            batch_mask.append(mask)
        batch_map['mask'] = batch_mask

    return batch_map


class Embed(nn.Module):
    def __init__(self, vocab_size, size, project_size=0, padding_idx=PADDING_IDX, embed=None, dropout_p=0, mode='learn'):
        super().__init__()

        self.mode = mode
        if mode == 'learn':
            self.embed = nn.Embedding(vocab_size, size, padding_idx=padding_idx)
            self.esize = self.embed.weight.shape[1]
        elif mode == 'pretrained':
            self.embed_pt = embed
            self.esize = self.embed_pt.weight.shape[1]
        elif mode == 'concat':
            self.embed = nn.Embedding(vocab_size, size, padding_idx=padding_idx)
            self.embed_pt = embed
            self.esize = self.embed.weight.shape[1] + self.embed_pt.weight.shape[1]

        self.project_size = project_size

        if self.project_size > 0:
            self.project = nn.Linear(self.esize, project_size, bias=False)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    @staticmethod
    def from_mode(mode, size, vocab_size, vocab, project_size, dropout_p,
                  cache_dir):
        if mode == 'word':
            return Embed(vocab_size, size, project_size=project_size, padding_idx=PADDING_IDX, dropout_p=dropout_p, mode='learn')
        elif mode == 'char':
            embeddings = read_embeddings(tokens=vocab, cache_dir=cache_dir)
            embeddings = torch.from_numpy(embeddings).float()
            assert embeddings.shape[0] == len(vocab)
            embed = nn.Embedding.from_pretrained(embeddings, padding_idx=PADDING_IDX, freeze=True)

            size = embed.weight.shape[1]

            return Embed(vocab_size, size, project_size=project_size, padding_idx=PADDING_IDX, dropout_p=dropout_p, mode='pretrained', embed=embed)
        elif mode == 'word+char':
            embeddings = read_embeddings(tokens=vocab, cache_dir=cache_dir)
            embeddings = torch.from_numpy(embeddings).float()
            assert embeddings.shape[0] == len(vocab)
            embed = nn.Embedding.from_pretrained(embeddings, padding_idx=PADDING_IDX, freeze=True)

            size = embed.weight.shape[1]

            return Embed(vocab_size, size, project_size=project_size, padding_idx=PADDING_IDX, dropout_p=dropout_p, mode='concat', embed=embed)

    @property
    def output_size(self):
        if self.project_size > 0:
            return self.project_size
        else:
            return self.esize

    def forward(self, x):
        # embed
        if self.mode == 'learn':
            output = self.embed(x)
        elif self.mode == 'pretrained':
            output = self.embed_pt(x)
        elif self.mode == 'concat':
            output = torch.cat([self.embed(x), self.embed_pt(x)], -1)
        # project
        if self.project_size > 0:
            output = self.project(output)
        # dropout
        output = self.dropout(output)
        return output


class TiedSoftmax(nn.Module):
    def __init__(self, hidden_size, embeddings):
        super().__init__()

        output_size, embed_size = embeddings.shape

        self.embed_size = embed_size
        self.output_size = output_size
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_size),
            )

        self.inv_embeddings = nn.Parameter(embeddings.t())
        self.inv_embeddings.requires_grad = False
        assert embeddings.requires_grad is False
        assert self.inv_embeddings.requires_grad is False

    def forward(self, x):
        query = self.project(x)
        logits = torch.matmul(query, self.inv_embeddings)
        return logits


class Net(nn.Module):
    def __init__(self, encode_text, encode_amr, output_size,
                 output_mode='linear', prior='attn', context='xy',
                 order=1,
                 ):
        super().__init__()

        self.output_size = output_size
        self.output_mode = output_mode
        self.prior = prior
        self.order = order

        self.encode_text = encode_text
        self.encode_amr = encode_amr
        self.project = nn.Linear(self.encode_text.output_size, self.encode_amr.output_size, bias=False)

        if self.order == 2:
            self.f_z = nn.LSTM(input_size=self.encode_amr.output_size,
                               hidden_size=self.encode_amr.output_size,
                               num_layers=1,
                               bidirectional=False,
                               batch_first=True)

        def get_hidden_size():
            size = self.encode_text.output_size + self.encode_amr.output_size
            return size

        def build_predict(hidden_size, output_size, output_mode):
            if output_mode == 'linear':
                layer = nn.Linear(hidden_size, output_size)
            elif output_mode == 'mlp':
                layer = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size),
                    )
            elif output_mode == 'tied':
                embeddings = encode_amr.embed.embed_pt.weight.data
                layer = TiedSoftmax(hidden_size, embeddings)
            return layer

        hidden_size = get_hidden_size()
        self.predict = build_predict(hidden_size, output_size, output_mode)

    @staticmethod
    def from_dataset_and_config(dataset, config, cache_dir):

        num_text_embeddings = len(dataset.text_tokenizer.token_TO_idx)
        num_amr_embeddings = len(dataset.amr_tokenizer.token_TO_idx)
        embedding_dim = config['embedding_dim']
        hidden_size = config['hidden_size']
        dropout = config['dropout']
        num_amr_layers = config.get('num_amr_layers', None)

        # TEXT

        text_embed = Embed.from_mode(mode=config['text_emb'],
                                     size=embedding_dim,
                                     project_size=config['text_project'],
                                     dropout_p=dropout,
                                     vocab_size=num_text_embeddings,
                                     vocab=dataset.text_tokenizer.vocab,
                                     cache_dir=cache_dir
                                     )

        if config['text_enc'] in ('bilstm', 'bitransformer'):
            encode_text = Encoder(text_embed, hidden_size, mode='text', rnn=config['text_enc'], dropout_p=dropout, cfg=config['text_enc_cfg'])

        # AMR

        amr_embed = Embed.from_mode(mode=config['amr_emb'],
                                    size=embedding_dim,
                                    project_size=config['amr_project'],
                                    dropout_p=dropout,
                                    vocab_size=num_amr_embeddings,
                                    vocab=dataset.amr_tokenizer.vocab,
                                    cache_dir=cache_dir
                                    )

        if config['amr_enc'] in ('lstm', 'bilstm', 'transformer', 'bitransformer'):
            encode_amr = Encoder(amr_embed, hidden_size, mode='amr', rnn=config['amr_enc'], dropout_p=dropout, cfg=config['amr_enc_cfg'])
        elif config['amr_enc'] == 'tree_rnn':
            encode_amr = TreeRNNEncoder(amr_embed, hidden_size, mode='tree_rnn', dropout_p=dropout)
        elif config['amr_enc'] == 'tree_lstm':
            encode_amr = TreeLSTMEncoder(amr_embed, hidden_size, mode='tree_lstm', dropout_p=dropout)
        elif config['amr_enc'] == 'tree_lstm_v2':
            encode_amr = TreeLSTMEncoder_v2(amr_embed, hidden_size, mode='tree_lstm', dropout_p=dropout)
        elif config['amr_enc'] == 'tree_lstm_v3':
            encode_amr = TreeLSTMEncoder(amr_embed, hidden_size, mode='tree_lstm_v3', dropout_p=dropout)
        elif config['amr_enc'] == 'tree_lstm_v4':
            encode_amr = TreeLSTMEncoder_v2(amr_embed, hidden_size, mode='tree_lstm_v3', dropout_p=dropout)
        elif config['amr_enc'].startswith('gcn'):
            encode_amr = GCNEncoder(amr_embed, hidden_size, mode=config['amr_enc'], dropout_p=dropout, num_layers=num_amr_layers)

        output_size = num_amr_embeddings

        # Assign kwargs automatically.
        kwargs = config.copy()
        kwargs['encode_text'] = encode_text
        kwargs['encode_amr'] = encode_amr
        kwargs['output_size'] = output_size

        del kwargs['text_emb'], kwargs['text_enc'], kwargs['text_enc_cfg'], kwargs['text_project']
        del kwargs['amr_emb'], kwargs['amr_enc'], kwargs['amr_enc_cfg'], kwargs['amr_project']
        del kwargs['embedding_dim'], kwargs['hidden_size'], kwargs['dropout']
        del kwargs['num_amr_layers']

        net = Net(**kwargs)

        param_count, param_count_rqeuires_grad = 0, 0
        for name, p in net.named_parameters():
            o = collections.OrderedDict(name=name, shape=p.shape, numel=p.shape.numel(), requires_grad=p.requires_grad)
            print(json.dumps(o))
            param_count += p.shape.numel()
            if p.requires_grad:
                param_count_rqeuires_grad += p.shape.numel()

        def count_params(m):
            return sum([p.shape.numel() for p in m.parameters()  if p.requires_grad])

        print('# of parameters (text_enc) = {}'.format(count_params(encode_text)))
        print('# of parameters (amr_enc) = {}'.format(count_params(encode_amr)))

        print('# of parameters = {} , # of trainable-parameters = {}'.format(param_count, param_count_rqeuires_grad))

        return net

    def forward(self, batch_map):
        x_t = batch_map['text_tokens']

        batch_size, len_t = x_t.shape
        device = x_t.device

        h_t, _, _, _ = self.encode_text(batch_map)
        z_t = self.project(h_t)
        h_a, y_a, y_a_mask, label_node_ids = self.encode_amr(batch_map)

        size_t = h_t.shape[-1]
        size_a = h_a.shape[-1]

        batch = collections.defaultdict(list)

        def align_and_predict(h_t, z_t, h_a, y_a, info):
            n_a, n_t = info['n_a'], info['n_t']

            if self.training and 'mask' in info:
                # Only predict for masked items.
                h_a = h_a[info['mask'] > 0]
                y_a = y_a[info['mask'] > 0]
                n_a = h_a.shape[0]

            def get_h_for_predict():
                h_a_expand = h_a.expand(n_a, n_t, size_a)
                h_t_expand = h_t.expand(n_a, n_t, size_t)
                h = torch.cat([h_a_expand, h_t_expand], -1)
                return h

            def get_likelihood():
                # output distribution
                h_for_predict = get_h_for_predict()
                dist = torch.softmax(self.predict(h_for_predict), 2)
                assert dist.shape == (n_a, n_t, self.output_size)

                # output probability
                index = y_a.expand(n_a, n_t, 1)
                p = dist.gather(index=index, dim=-1)
                assert p.shape == (n_a, n_t, 1)

                return p

            def get_posterior(prior, likelihood):
                return prior * likelihood

            def get_prior(h_a, z_t):
                if self.prior == 'unif':
                    prior = torch.full((n_a, n_t, 1), 1 / n_t, dtype=torch.float, device=device)
                elif self.prior == 'attn':
                    prior = torch.softmax(torch.sum(h_a * z_t, -1, keepdims=True), 1)
                assert prior.shape == (n_a, n_t, 1)
                return prior

            # likelihood
            p = get_likelihood()

            # prior
            alpha = get_prior(h_a, z_t)

            if self.order == 2:
                z0 = (h_a * alpha).sum(0).unsqueeze(0)
                shape = (1, z0.shape[0], z0.shape[-1])
                c0 = torch.full(shape, 0, dtype=torch.float, device=device)
                h0 = torch.full(shape, 0, dtype=torch.float, device=device)
                z1, _ = self.f_z(z0, (h0, c0))

                alpha = get_prior(h_a, z1)

            # posterior
            align = get_posterior(alpha, p)

            # marginal probability
            loss_notreduced = -(torch.sum(alpha * p, 1) + 1e-8).log()
            loss = loss_notreduced.sum(0)

            pr = 0

            return alpha, p, loss, loss_notreduced, align, pr

        def mask_and_apply(i_b, func):
            # get amr mask
            local_y_a_mask = y_a_mask[i_b].view(-1)
            n_a = local_y_a_mask.long().sum().item()

            # get text mask
            local_x_t = x_t[i_b]
            local_x_t_mask = local_x_t != PADDING_IDX
            n_t = local_x_t_mask.long().sum().item()

            assert n_a > 0, (n_a, n_t)
            assert n_t > 0, (n_a, n_t)

            # get local amr vec
            local_h_a = h_a[i_b][local_y_a_mask].view(n_a, 1, size_a)

            # get local text vec
            local_h_t = h_t[i_b][local_x_t_mask].view(1, n_t, size_t)

            # project text vec
            local_z_t = z_t[i_b][local_x_t_mask].view(1, n_t, size_a)

            # get labels
            local_y_a = y_a[i_b][local_y_a_mask].view(n_a, 1, 1)

            # info
            info = {}
            info['n_a'] = n_a
            info['n_t'] = n_t

            if args.mask > 0 and self.training:
                info['mask'] = batch_map['mask'][i_b]

            result = func(local_h_t, local_z_t, local_h_a, local_y_a, info)

            return result

        for i_b in range(batch_size):

            alpha, p, loss, loss_notreduced, align, pr = mask_and_apply(i_b, align_and_predict)

            batch['alpha'].append(alpha)
            batch['p'].append(p)
            batch['loss'].append(loss)
            batch['loss_notreduced'].append(loss_notreduced)
            batch['align'].append(align)
            batch['pr'].append(pr)

        model_output = {}
        model_output['batch_alpha'] = batch['alpha']
        model_output['batch_p'] = batch['p']
        model_output['batch_loss'] = batch['loss']
        model_output['batch_loss_notreduced'] = batch['loss_notreduced']
        model_output['batch_align'] = batch['align']
        model_output['batch_pr'] = batch['pr']
        model_output['labels'] = y_a
        model_output['labels_mask'] = y_a_mask
        model_output['label_node_ids'] = label_node_ids

        return model_output


class Encoder(nn.Module):
    def __init__(self, embed, hidden_size, mode='text', dropout_p=0, input_size=None, rnn='lstm', cfg={}):
        super().__init__()

        self.hidden_size = hidden_size
        self.nlayers = nlayers = cfg.get('nlayers', 1)
        self.mode = mode

        self.embed = embed

        if input_size is None:
            input_size = embed.output_size

        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers,
                               bidirectional=False, batch_first=True)
            self.bidirectional = False
            self.model_type = 'rnn'

        elif rnn == 'bilstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers,
                               bidirectional=True, batch_first=True)
            self.bidirectional = True
            self.model_type = 'rnn'

        elif rnn == 'transformer':
            self.rnn = TransformerModel(ninp=input_size, nhead=cfg.get('nhead', 4), nhid=hidden_size, nlayers=nlayers, dropout=dropout_p)
            self.bidirectional = False
            self.model_type = 'transformer'

        elif rnn == 'bitransformer':
            self.rnn = BiTransformer(ninp=input_size, nhead=cfg.get('nhead', 4), nhid=hidden_size, nlayers=nlayers, dropout=dropout_p)
            self.bidirectional = True
            self.model_type = 'transformer'

        self.output_size = 2 * hidden_size if self.bidirectional else hidden_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, batch_map, input_vector=None):
        if self.mode == 'text':
            tokens = batch_map['text_tokens']
            tokens_mask = None
            labels = None
            labels_mask = None
            label_node_ids = None
            node_ids = None
        elif self.mode == 'amr':
            tokens = batch_map['amr_tokens']
            tokens_mask = batch_map['amr_node_mask']
            labels = None
            labels_mask = None
            label_node_ids = None
            node_ids = batch_map['amr_node_ids']

        batch_size, length = tokens.shape
        hidden_size = self.hidden_size
        n = 2 if self.bidirectional else 1
        shape = (n * self.nlayers, batch_size, hidden_size)
        device = tokens.device

        # compute hidden states
        c0 = torch.full(shape, 0, dtype=torch.float, device=device)
        h0 = torch.full(shape, 0, dtype=torch.float, device=device)

        if input_vector is None:
            e = self.embed(tokens)
        else:
            e = input_vector
        if self.model_type == 'rnn':
            output, _ = self.rnn(e, (h0, c0))
        elif self.model_type == 'transformer':
            output = self.rnn(e)
        output = self.dropout(output)

        if self.mode == 'amr' and self.bidirectional:
            h = output
            assert h.shape == (batch_size, length, 2 * hidden_size)
            h_reshape = h.view(batch_size, length, 2, hidden_size)

            new_h = torch.zeros_like(h_reshape)
            new_h[:, :, 0] = h_reshape[:, :, 0]
            new_h[:, :-2, 1] = h_reshape[:, 2:, 1]

            h = new_h.view(batch_size, length, 2 * hidden_size)

            output = h

        # compute labels and masks
        if self.mode == 'amr':
            labels = torch.full(tokens.shape, PADDING_IDX, dtype=torch.long, device=device)
            labels[:, :-1] = tokens[:, 1:]

            labels_mask = torch.full(tokens.shape, False, dtype=torch.bool, device=device)
            labels_mask[:, :-1] = tokens_mask[:, 1:]

            label_node_ids = torch.full(tokens.shape, -1, dtype=torch.long, device=device)
            label_node_ids[:, :-1] = node_ids[:, 1:]

        return output, labels, labels_mask, label_node_ids


def load_checkpoint(path, net, cuda=False):
    try:
        toload = torch.load(path)
    except:
        toload = torch.load(path, map_location=torch.device('cpu'))

    state_dict = net.state_dict()

    for k, v in net.named_parameters():
        if not v.requires_grad:
            print('[load] copying {}'.format(k))
            toload['state_dict'][k] = state_dict[k]

    # TODO: Verify that vocab lines up.
    net.load_state_dict(toload['state_dict'])


def save_checkpoint(path, dataset, net, metrics=None):
    state_dict = net.state_dict()

    for k, v in net.named_parameters():
        if not v.requires_grad:
            print('[save] removing {}'.format(k))
            del state_dict[k]

    tosave = {}
    tosave['state_dict'] = state_dict
    tosave['text_vocab'] = dataset.text_tokenizer.token_TO_idx
    tosave['amr_vocab'] = dataset.amr_tokenizer.token_TO_idx
    tosave['metrics'] = metrics

    try:
        torch.save(tosave, path, _use_new_zipfile_serialization=False)
    except:
        torch.save(tosave, path)


def check_and_update_best(best_metrics, key, val, compare='gt'):
    def compare_func(curr, prev):
        if compare == 'gt':
            return curr > prev
        if compare == 'lt':
            return curr < prev
    is_best = key not in best_metrics or compare_func(val, best_metrics[key])
    if is_best:
        prev = 'none' if key not in best_metrics else '{:.3f}'.format(best_metrics[key])
        curr = '{:.3f}'.format(val)
        print('new_best, key={}, prev={}, curr={}'.format(key, prev, curr))
        best_metrics[key] = val
    return is_best


def save_metrics(path, metrics):
    with open(path, 'w') as f:
        f.write(json.dumps(metrics) + '\n')


def default_model_config():
    config = {}
    config['text_emb'] = 'word'
    config['text_enc'] = 'bilstm'
    config['text_enc_cfg'] = {}
    config['text_project'] = 0
    config['amr_emb'] = 'word'
    config['amr_enc'] = 'bilstm'
    config['amr_enc_cfg'] = {}
    config['amr_project'] = 0
    config['num_amr_layers'] = 2
    config['embedding_dim'] = 100
    config['hidden_size'] = 100
    config['dropout'] = 0
    config['output_mode'] = 'linear'
    config['prior'] = 'attn'
    config['order'] = 1
    return config


def init_tokenizers(text_vocab_file, amr_vocab_file):

    # text tokenizer
    tokens = read_text_vocab_file(text_vocab_file)
    text_tokenizer = TextTokenizer()
    text_tokenizer.set_tokens(tokens)
    text_tokenizer.finalize()
    assert text_tokenizer.token_TO_idx[PADDING_TOK] == PADDING_IDX

    # amr tokenizer
    tokens = read_amr_vocab_file(amr_vocab_file)
    amr_tokenizer = AMRTokenizer()
    amr_tokenizer.set_tokens(tokens)
    amr_tokenizer.finalize()
    assert amr_tokenizer.token_TO_idx[PADDING_TOK] == PADDING_IDX

    return text_tokenizer, amr_tokenizer


def mask_one(batch_map, i=0):
    has_mask = []
    batch_mask = []
    for i_b, x in enumerate(batch_map['items']):
        size = len(x['amr_nodes'])
        mask = np.ones(size)
        mask[:] = MaskInfo.unchanged
        if i < size and size > 1:
            mask[i] = MaskInfo.masked
            has_mask.append(True)
        else:
            has_mask.append(False)

        mask = torch.from_numpy(mask).long()
        batch_mask.append(mask)
    return batch_mask, has_mask


def filter_batch(batch_map, keep_mask):
    new_batch_map = {}

    for k, v in batch_map.items():
        if isinstance(v, torch.Tensor):
            new_batch_map[k] = v[keep_mask]
        elif isinstance(v, list):
            new_batch_map[k] = [v[i_b] for i_b, m in enumerate(keep_mask) if m]
        else:
            new_batch_map[k] = v

    return new_batch_map


def masked_validation_step(net, batch_indices, batch_map):
    batch_size = len(batch_map['items'])
    max_amr_node_size = max([len(x['amr_nodes']) for x in batch_map['items']])
    new_model_output = None
    for i in range(max_amr_node_size):
        batch_mask, batch_has_mask = mask_one(batch_map, i=i)
        bm = copy.deepcopy(batch_map)
        bm['mask'] = batch_mask
        if i > 0:
            bm = filter_batch(bm, keep_mask=batch_has_mask)
            batch_i_b = [i_b for i_b, m in enumerate(batch_has_mask) if m]
        else:
            batch_i_b = list(range(batch_size))

        assert len(batch_i_b) > 0
        model_output = net(bm)
        if i == 0:
            new_model_output = model_output
            continue

        for i_bm, i_b in enumerate(batch_i_b):
            has_mask = batch_has_mask[i_b]
            assert has_mask is True

            mask = batch_mask[i_b]
            assert mask.shape[0] == len(batch_map['items'][i_b]['amr_nodes'])

            # NOTE: There is some redundancy here...
            for k, v in model_output.items():
                if not isinstance(v[i_bm], torch.Tensor):
                    continue

                if v[i_bm].shape[0] != mask.shape[0]:
                    continue

                new_model_output[k][i_b][i] = v[i_bm][i]

    return new_model_output


def standard_validation_step(net, batch_indices, batch_map):
    return net(batch_map)


def shared_validation_step(*args, **kwargs):
    if 'gcn' in cli_args.model_config and not cli_args.no_mask_at_inference:
        return masked_validation_step(*args, **kwargs)
    return standard_validation_step(*args, **kwargs)


def maybe_write(context):

    args = context['args']
    net = context['net']
    dataset = context['dataset']
    corpus = dataset.corpus

    batch_size = args.batch_size

    def write_align_dist(corpus, dataset, path):
        # TODO: Write to numpy array (memmap).
        net.eval()

        metrics = collections.defaultdict(list)

        indices = np.arange(len(corpus))

        dist_list = []
        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
                end = min(start + batch_size, len(corpus))
                batch_indices = indices[start:end]
                items = [dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)

                # forward pass
                model_output = shared_validation_step(net, batch_indices, batch_map)

                # metrics
                metrics['loss_notreduced'] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()

                # write pretty alignment info
                for idx, ainfo in zip(batch_indices, AlignmentDecoder().batch_decode(batch_map, model_output)):
                    dist_list.append(ainfo['posterior'].cpu().numpy())

        for k, v in metrics.items():
            print('{} {:.3f}'.format(k, np.mean(v)))

        # WRITE alignments to file.
        _, corpus_id = save_align_dist(path, corpus, dist_list)

        # WRITE corpus hash to verify data later.
        with open(path + '.corpus_hash', 'w') as f:
            f.write(corpus_id)

    def write_align_pretty(corpus, dataset, path):
        net.eval()

        indices = np.arange(len(corpus))

        print('writing to {}.pretty'.format(os.path.abspath(path)))
        f_pretty = open(path + '.pretty', 'w')
        f_gold = open(path + '.gold', 'w')

        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
                end = min(start + batch_size, len(corpus))
                batch_indices = indices[start:end]
                items = [dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)

                # forward pass
                model_output = shared_validation_step(net, batch_indices, batch_map)

                # write pretty alignment info
                for idx, ainfo in zip(batch_indices, AlignmentDecoder().batch_decode(batch_map, model_output)):
                    amr = corpus[idx]
                    f_pretty.write(amr_to_pretty_format(amr, ainfo, idx).strip() + '\n\n')

                # FIXME: I use write pretty when there is no reference
                # # write reference amr
                # for idx in batch_indices:
                #     amr = corpus[idx]
                #     f_gold.write(amr_to_string(amr).strip() + '\n\n')

        f_pretty.close()
        f_gold.close()

    def write_align(corpus, dataset, path_gold, path_pred, write_gold=True):
        net.eval()

        indices = np.arange(len(corpus))
        metrics = collections.defaultdict(list)

        predictions = collections.defaultdict(list)

        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
                end = min(start + batch_size, len(corpus))
                batch_indices = indices[start:end]
                items = [dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)

                # forward pass
                model_output = shared_validation_step(net, batch_indices, batch_map)

                # metrics
                metrics['loss_notreduced'] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()

                # save alignments for eval.
                for idx, ainfo in zip(batch_indices, AlignmentDecoder().batch_decode(batch_map, model_output)):
                    amr = corpus[idx]
                    node_ids = get_node_ids(amr)
                    alignments = {node_ids[node_id]: a for node_id, a in ainfo['node_alignments']}

                    predictions['amr'].append(amr)
                    predictions['alignments'].append(alignments)# write alignments

        for k, v in metrics.items():
            print('{} {:.3f}'.format(k, np.mean(v)))

        # write pred
        with open(path_pred, 'w') as f_pred:
            for amr, alignments in zip(predictions['amr'], predictions['alignments']):
                f_pred.write(amr_to_string(amr, alignments).strip() + '\n\n')

        # write gold
        if write_gold:
            with open(path_gold, 'w') as f_gold:
                for amr, alignments in zip(predictions['amr'], predictions['alignments']):
                       f_gold.write(amr_to_string(amr).strip() + '\n\n')

    if args.write_pretty:

        path = os.path.join(args.log_dir, 'alignment.trn')
        write_align_pretty(corpus, dataset, path)
        sys.exit()

    if args.write_align_dist:

        path = args.single_output
        write_align_dist(corpus, dataset, path)
        sys.exit()

    if args.write_only:

        path = os.path.join(args.log_dir, 'alignment.trn.out')
        path_gold = path + '.gold'
        path_pred = path + '.pred'
        write_align(corpus, dataset, path_gold, path_pred)
        eval_output = EvalAlignments().run(path_gold, path_pred)

        with open(path_pred + '.eval', 'w') as f:
            f.write(json.dumps(eval_output))

        sys.exit()

    if args.write_single:

        path_gold = None
        path_pred = args.single_output
        write_align(corpus, dataset, path_gold, path_pred, write_gold=False)

        sys.exit()


def break_if_oov(corpus, corpus_file, tokenizer, vocab_file, fun):
    # sanity check: tokenizer suports all tokens in
    for amr in corpus:
        for token in fun(amr):
            if token not in tokenizer.token_TO_idx:
                raise Exception(
                    f'{token} in {corpus_file} not in {vocab_file}'
                )


def main(args):
    batch_size = args.batch_size
    lr = args.lr
    max_epoch = args.max_epoch
    seed = args.seed

    model_config = JSONConfig(default_model_config()).parse(args.model_config)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.read_only:
        t = AMRTokenizer()
        for amr in read_amr2(args.trn_amr, ibm_format=False):
            t.dfs(amr)
        sys.exit()

    # TOKENIZERS
    text_tokenizer, amr_tokenizer = init_tokenizers(text_vocab_file=args.vocab_text, amr_vocab_file=args.vocab_amr)

    # DATA
    # Train
    trn_corpus = safe_read(args.trn_amr, max_length=args.max_length)
    # Validation
    val_corpus_list = [safe_read(path, max_length=args.val_max_length) for path in args.val_amr]

    # sanity check: we should allways have tokens
    assert all(bool(amr) for amr in trn_corpus), \
        "{args.trn_amr} must be tokenized"
    # sanity check: --write-only expects alignments to compare with
    if args.write_only:
        assert all(bool(amr.alignments) for amr in trn_corpus), \
            f"--write-only assumes {args.trn_amr} will be aligned"
    # sanity check: tokenizer suports all tokens in
    break_if_oov(trn_corpus, args.trn_amr, text_tokenizer, args.vocab_text, lambda a: a.tokens)
    break_if_oov(trn_corpus, args.trn_amr, amr_tokenizer, args.vocab_amr, lambda a: a.nodes.values())
    if args.write_only:
        for val_corpus, path in zip(val_corpus_list, args.val_amr):
            # sanity check: --write-only expects alignments to compare with
            assert all(bool(amr.alignments) for amr in val_corpus), \
                f"--write-only assumes {path} will be aligned"
    # sanity check: we should allways have tokens and no OOV
    for val_corpus, path in zip(val_corpus_list, args.val_amr):
        assert all(bool(amr) for amr in val_corpus), \
            "{path} must be tokenized"
        # sanity check: tokenizer suports all tokens in
        break_if_oov(val_corpus, path, text_tokenizer, args.vocab_text, lambda a: a.tokens)
        break_if_oov(val_corpus, path, amr_tokenizer, args.vocab_amr, lambda a: a.nodes.values())

    # datasets
    trn_dataset = Dataset(trn_corpus, text_tokenizer=text_tokenizer, amr_tokenizer=amr_tokenizer)
    val_dataset_list = [Dataset(x, text_tokenizer=text_tokenizer, amr_tokenizer=amr_tokenizer) for x in val_corpus_list]

    # MODEL

    net = Net.from_dataset_and_config(trn_dataset, model_config, args.cache_dir)
    if args.load is not None:
        load_checkpoint(args.load, net)

    if args.cuda:
        net.cuda()

    # ALTERNATIVE to training. Will exit if triggered.

    context = {}
    context['args'] = args
    context['net'] = net
    context['dataset'] = trn_dataset

    maybe_write(context)

    # OPTIMIZER

    opt = optim.Adam(net.parameters(), lr=lr)

    # CACHE dataset items.

    for idx in tqdm(range(len(trn_corpus)), desc='cache-trn', disable=not args.verbose):
        _ = trn_dataset[idx]

    for i_val in range(len(val_corpus_list)):
        val_corpus = val_corpus_list[i_val]
        val_dataset = val_dataset_list[i_val]
        for idx in tqdm(range(len(val_corpus)), desc='cache-val', disable=not args.verbose):
            _ = val_dataset[idx]

    # MAIN

    best_metrics = {}

    for epoch in range(max_epoch):

        epoch_metrics = {}
        epoch_metrics['epoch'] = epoch

        # TRAINING

        print('\n\n' + '@' * 40 + '\n' + 'TRAIN' + '\n')

        metrics = collections.defaultdict(list)

        def trn_step(batch_indices, batch_map, info):
            net.train()

            model_output = net(batch_map)

            # neg log likelihood
            loss = 0
            loss += torch.cat(model_output['batch_loss'], 0).mean()

            if info['should_clear']:
                opt.zero_grad()
            (loss / info['accum_steps']).backward()
            if info['should_update']:
                opt.step()

            metrics['trn_loss'] += [x.item() for x in model_output['batch_loss']]
            metrics['trn_loss_notreduced'] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()

            del loss

        # batch iterator
        if args.long_first:
            indices = np.arange(len(trn_corpus))
            lengths = [len(x.tokens) for x in trn_corpus]
            indices = [i for i, x in sorted(zip(indices, lengths), key=lambda x: -x[1])]
        elif args.short_first:
            indices = np.arange(len(trn_corpus))
            lengths = [len(x.tokens) for x in trn_corpus]
            indices = [i for i, x in sorted(zip(indices, lengths), key=lambda x: x[1])]
        else:
            indices = np.arange(len(trn_corpus))
            np.random.shuffle(indices)

        # add info to batch iterator
        def get_batch_iterator_with_info():
            def func():
                batch_list = None
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    batch_indices = indices[start:end]

                    if batch_list is None:
                        batch_list = []

                    batch_list.append(batch_indices)

                    if len(batch_list) == args.accum_steps:
                        yield batch_list
                        batch_list = None

                if batch_list is not None and len(batch_list) > 0:
                    yield batch_list

            for step, batch_list in enumerate(func()):
                for i_batch, batch_indices in enumerate(batch_list):
                    info = {}
                    info['should_clear'] = i_batch == 0
                    info['should_update'] = i_batch == (len(batch_list) - 1) # Only update gradients for last subbatch.
                    info['accum_steps'] = len(batch_list)
                    yield batch_indices, info

                if args.batches_per_epoch > 0 and (step + 1) == args.batches_per_epoch:
                    break

        # loop
        for batch_indices, info in tqdm(get_batch_iterator_with_info(), desc='trn-epoch[{}]'.format(epoch), disable=not args.verbose):
            items = [trn_dataset[idx] for idx in batch_indices]
            batch_map = batchify(items, cuda=args.cuda, train=True)
            trn_step(batch_indices, batch_map, info)

        trn_loss = np.mean(metrics['trn_loss'])
        trn_loss_notreduced = np.mean(metrics['trn_loss_notreduced'])
        trn_ppl = 2 ** (trn_loss_notreduced / np.log(2))
        trn_pr = np.mean(metrics['trn_pr'])

        epoch_metrics['trn_loss'] = trn_loss
        epoch_metrics['trn_loss_notreduced'] = trn_loss_notreduced
        epoch_metrics['trn_ppl'] = trn_ppl

        print('trn epoch = {}, loss = {:.3f}, loss-nr = {:.3f}, ppl = {:.3f}, pr = {:.3f}'.format(
            epoch, trn_loss, trn_loss_notreduced, trn_ppl, trn_pr))

        save_checkpoint(os.path.join(args.log_dir, 'model.latest.pt'), trn_dataset, net, metrics=dict(epoch=epoch, trn_loss=trn_loss, trn_loss_notreduced=trn_loss_notreduced))

        if epoch % args.save_every_epoch == 0 and epoch > 0:
            save_checkpoint(os.path.join(args.log_dir, 'model.epoch_{}.pt'.format(epoch)), trn_dataset, net, metrics=dict(epoch=epoch, trn_loss=trn_loss, trn_loss_notreduced=trn_loss_notreduced))

        # VALIDATION

        if args.skip_validation:
            continue

        print('\n\n' + '@' * 40 + '\n' + 'VALID' + '\n')

        for i_valid, val_dataset in enumerate(val_dataset_list):

            print('[{}]'.format(i_valid))
            print(args.val_amr[i_valid])

            val_corpus = val_corpus_list[i_valid]

            val_predictions = collections.defaultdict(list)

            def val_step(batch_indices, batch_map):
                with torch.no_grad():
                    net.eval()

                    model_output = net(batch_map)

                    # neg log likelihood
                    loss = torch.cat(model_output['batch_loss'], 0).mean()

                    metrics['val_{}_loss'.format(i_valid)] += [x.item() for x in model_output['batch_loss']]
                    metrics['val_{}_loss_notreduced'.format(i_valid)] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()

                    del loss

            # batch iterator
            indices = [i for i, _ in enumerate(sorted(val_corpus, key=lambda x: -len(x.tokens)))]

            for start in tqdm(range(0, len(indices), batch_size), desc='val_{}-epoch[{}]'.format(i_valid, epoch), disable=not args.verbose):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                items = [val_dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)
                val_step(batch_indices, batch_map)

            val_loss = np.mean(metrics['val_{}_loss'.format(i_valid)])
            val_loss_notreduced = np.mean(metrics['val_{}_loss_notreduced'.format(i_valid)])
            val_ppl = 2 ** (val_loss_notreduced / np.log(2))

            epoch_metrics['val_{}_loss'.format(i_valid)] = val_loss
            epoch_metrics['val_{}_loss_notreduced'.format(i_valid)] = val_loss_notreduced
            epoch_metrics['val_{}_ppl'.format(i_valid)] = val_ppl

            print('val_{} epoch = {}, loss = {:.3f}, loss-nr = {:.3f}, ppl = {:.3f}'.format(
                i_valid, epoch, val_loss, val_loss_notreduced, val_ppl))

        save_metrics(os.path.join(args.log_dir, 'model.epoch_{}.metrics'.format(epoch)), epoch_metrics)


if __name__ == '__main__':
    args = cli_args = argument_parser()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    print(json.dumps(args.__dict__))

    os.system('mkdir -p {}'.format(args.log_dir))

    with open(os.path.join(args.log_dir, 'flags.json'), 'w') as f:
        f.write(json.dumps(args.__dict__))

    main(args)
