# -*- coding: utf-8 -*-

import argparse
import collections
import json
import os
import sys

os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from fairseq.models.roberta import alignment_utils
except:
    print('Could not import alignment_utils.')

from tqdm import tqdm

from amr_utils import convert_amr_to_tree, compute_pairwise_distance, get_node_ids
from amr_utils import read_amr
from evaluation import EvalAlignments
from formatter import FormatAlignments, FormatAlignmentsPretty
from pretrained_embeddings import read_embeddings, read_amr_vocab_file, read_text_vocab_file
from tree_rnn import TreeEncoder as TreeRNNEncoder
from tree_lstm import TreeEncoder as TreeLSTMEncoder
from vocab import *


class RobertaLoader:
    def __init__(self):
        self.roberta = None

    def load(self):
        if self.roberta is None:
            self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        return self.roberta

roberta_loader = RobertaLoader()


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
        "--vocab-text",
        help="Vocab file.",
        type=str,
        default='./align_cfg/vocab.text.2021-06-30.txt'
    )
    parser.add_argument(
        "--vocab-amr",
        help="Vocab file.",
        type=str,
        default='./align_cfg/vocab.amr.2021-06-30.txt'
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
        default="/dccstor/ykt-parse/SHARED/misc/adrozdov",
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
        "--skip-align-loss",
        help="Hyperparam for loss.",
        action='store_true',
    )
    parser.add_argument(
        "--pr",
        help="Hyperparam for posterior regularization.",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--pr-after",
        help="Hyperparam for posterior regularization.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--pr-epsilon",
        help="Hyperparam for posterior regularization.",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--pr-mode",
        help="Hyperparam for posterior regularization.",
        default="prior",
        type=str,
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
        "--write-only",
        help="If true, then write alignments to file.",
        action='store_true',
    )
    parser.add_argument(
        "--use-jamr",
        help="If true, then write original alignments to file.",
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
        "--max-length",
        help="Max number of text tokens.",
        default=100,
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
        "--demo",
        help="If true, then print progress bars.",
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
        "--skip-validation",
        action='store_true'
    )
    parser.add_argument(
        "--check-for-bpe",
        action='store_true'
    )
    parser.add_argument(
        "--jbsub-eval",
        action='store_true'
    )
    parser.add_argument("--name", default=None, type=str,
                        help="Useful for book-keeping.")
    args = parser.parse_args()

    if not os.path.exists(args.home):
        args.home = os.path.expanduser('~')

    if args.val_amr is None:
        args.val_amr = [
            os.path.join(args.home, 'data/AMR2.0/aligned/cofill/train.txt.dev-unseen-v1'),
            os.path.join(args.home, 'data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1'),
        ]

    if args.tst_amr is None:
        args.tst_amr = os.path.join(args.home, 'data/AMR2.0/aligned/cofill/test.txt')

    if args.demo:
        args.trn_amr = args.val_amr[0]
        args.val_max_length = args.max_length

    if args.write_only:
        if args.trn_amr is None:
            args.trn_amr = os.path.join(args.home, 'data/AMR2.0/aligned/cofill/train.txt')

    if args.trn_amr is None:
        args.trn_amr = os.path.join(args.home, 'data/AMR2.0/aligned/cofill/train.txt.train-v1')

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

        seen = set()

        seen.add(amr.root)

        # build tree
        g = collections.defaultdict(list)
        g_labels = {}

        def sortkey(x):
            s, y, t = x
            return (s, t)

        for e in sorted(amr.edges, key=sortkey):
            s, y, t = e
            if t in seen:
                continue
            assert s <= t
            seen.add(t)

            g[s].append(t)
            g_labels[(s, t)] = y

        # render tree
        def render(s):
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


class AlignmentsWriter(object):
    def __init__(self, path, dataset, formatter, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return

        self.fout_pred = open(path + '.pred', 'w')
        self.fout_gold = open(path + '.gold', 'w')
        self.formatter = formatter
        self.dataset = dataset

    def write_batch(self, batch_indices, batch_map, model_output):
        if not self.enabled:
            return

        # write
        for i_b, (out, pred_alignments) in enumerate(self.formatter.format(batch_map, model_output, batch_indices)):
            idx = batch_indices[i_b]
            amr = self.dataset.corpus[idx]

            # write pred
            self.fout_pred.write(out.strip() + '\n\n')

            # write gold
            self.fout_gold.write(amr.toJAMRString().strip() + '\n\n')

    def close(self):
        if not self.enabled:
            return
        self.fout_pred.close()
        self.fout_gold.close()


class Dataset(object):
    def __init__(self, corpus, text_tokenizer=None, amr_tokenizer=None):
        self.corpus = corpus
        self.text_tokenizer = text_tokenizer
        self.amr_tokenizer = amr_tokenizer

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
        pairwise_dist = compute_pairwise_distance(tree)

        return g, pairwise_dist

    def __getitem__(self, idx):
        amr = self.corpus[idx]

        item = {}

        # text
        tokens = amr.tokens
        item['text_original_tokens'] = tokens
        item['text_tokens'] = self.text_tokenizer.indexify(tokens)

        # amr
        tokens, token_ids = self.amr_tokenizer.dfs(amr)
        item['amr_tokens'] = self.amr_tokenizer.indexify(tokens)
        item['amr_node_ids'] = token_ids
        item['amr_node_mask'] = [False if x < 0 else True for x in token_ids]

        g, pairwise_dist = self.get_dgl_graph(amr)
        item['g'] = g
        item['amr_pairwise_dist'] = pairwise_dist

        return item


def batchify(items, cuda=False):
    device = torch.cuda.current_device() if cuda else None

    dtypes = {
        'text_tokens': torch.long,
        'amr_tokens': torch.long,
        'amr_node_ids': torch.long,
        'amr_node_mask': torch.bool,
        'amr_pairwise_dist': list,
    }

    batch_map = {}
    for k in dtypes.keys():
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

    g = dgl.batch([x['g'] for x in items])
    g = g.to(device)

    rg = dgl.reverse(g)
    for k, v in g.edata.items():
        rg.edata[k] = v

    batch_map['g'] = g
    batch_map['rg'] = rg

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
    def from_mode(mode, size, vocab_size, vocab, project_size, dropout_p):
        if mode == 'word':
            return Embed(vocab_size, size, project_size=project_size, padding_idx=PADDING_IDX, dropout_p=dropout_p, mode='learn')
        elif mode == 'char':
            embeddings = read_embeddings(tokens=vocab)
            embeddings = torch.from_numpy(embeddings).float()
            assert embeddings.shape[0] == len(vocab)
            embed = nn.Embedding.from_pretrained(embeddings, padding_idx=PADDING_IDX, freeze=True)

            size = embed.weight.shape[1]

            return Embed(vocab_size, size, project_size=project_size, padding_idx=PADDING_IDX, dropout_p=dropout_p, mode='pretrained', embed=embed)
        elif mode == 'word+char':
            embeddings = read_embeddings(tokens=vocab)
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
                 output_mode='linear', prior='attn', align_mode='posterior', context='xy',
                 context_2=None, lambda_context=0.5, output_space='vocab'):
        super().__init__()

        self.output_size = output_size
        self.output_mode = output_mode
        self.output_space = output_space

        self.prior = prior
        self.align_mode = align_mode
        self.context = context

        self.encode_text = encode_text
        self.encode_amr = encode_amr
        self.project = nn.Linear(self.encode_text.output_size, self.encode_amr.output_size, bias=False)

        if self.output_space == 'graph':
            self.project_o = nn.Linear(self.encode_text.output_size, self.encode_amr.output_size, bias=False)

        def get_hidden_size(context):
            if context == 'xy':
                size = self.encode_text.output_size + self.encode_amr.output_size
            elif context == 'x':
                size = self.encode_text.output_size
            elif context == 'e':
                size = self.encode_text.embed.output_size
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

        # 1st context
        hidden_size = get_hidden_size(self.context)
        self.predict = build_predict(hidden_size, output_size, output_mode)

        # 2nd context
        self.context_2 = context_2
        self.lambda_context = lambda_context

        if self.context_2 is not None:
            hidden_size = get_hidden_size(self.context_2)
            self.predict_2 = build_predict(hidden_size, output_size, output_mode)

    @staticmethod
    def from_dataset_and_config(dataset, config):

        num_text_embeddings = len(dataset.text_tokenizer.token_TO_idx)
        num_amr_embeddings = len(dataset.amr_tokenizer.token_TO_idx)
        embedding_dim = config['embedding_dim']
        hidden_size = config['hidden_size']
        dropout = config['dropout']

        # TEXT

        text_embed = Embed.from_mode(mode=config['text_emb'],
                                     size=embedding_dim,
                                     project_size=config['text_project'],
                                     dropout_p=dropout,
                                     vocab_size=num_text_embeddings,
                                     vocab=dataset.text_tokenizer.vocab,
                                     )

        if config['text_enc'] == 'bilstm':
            encode_text = Encoder(text_embed, hidden_size, mode='text', bidirectional=True, dropout_p=dropout)
        elif config['text_enc'] == 'roberta':
            encode_text = RobertaEncoder(text_embed, hidden_size, dropout_p=dropout, use_lstm=False)
        elif config['text_enc'] == 'roberta+lstm':
            encode_text = RobertaEncoder(text_embed, hidden_size, dropout_p=dropout, use_lstm=True)

        # AMR

        amr_embed = Embed.from_mode(mode=config['amr_emb'],
                                    size=embedding_dim,
                                    project_size=config['amr_project'],
                                    dropout_p=dropout,
                                    vocab_size=num_amr_embeddings,
                                    vocab=dataset.amr_tokenizer.vocab,
                                    )

        if config['amr_enc'] == 'lstm':
            encode_amr = Encoder(amr_embed, hidden_size, mode='amr', bidirectional=False, dropout_p=dropout)
        elif config['amr_enc'] == 'bilstm':
            encode_amr = Encoder(amr_embed, hidden_size, mode='amr', bidirectional=True, dropout_p=dropout)
        elif config['amr_enc'] == 'tree_rnn':
            encode_amr = TreeRNNEncoder(amr_embed, hidden_size, mode='tree_rnn', dropout_p=dropout)
        elif config['amr_enc'] == 'etree_rnn':
            encode_amr = TreeRNNEncoder(amr_embed, hidden_size, mode='etree_rnn', dropout_p=dropout)
        elif config['amr_enc'] == 'tree_lstm':
            encode_amr = TreeLSTMEncoder(amr_embed, hidden_size, mode='tree_lstm', dropout_p=dropout)

        output_size = num_amr_embeddings

        # Assign kwargs automatically.
        kwargs = config.copy()
        kwargs['encode_text'] = encode_text
        kwargs['encode_amr'] = encode_amr
        kwargs['output_size'] = output_size

        del kwargs['text_emb'], kwargs['text_enc'], kwargs['text_project']
        del kwargs['amr_emb'], kwargs['amr_enc'], kwargs['amr_project']
        del kwargs['embedding_dim'], kwargs['hidden_size'], kwargs['dropout']

        net = Net(**kwargs)

        param_count = 0
        for name, p in net.named_parameters():
            should_count = True
            if 'embed.embed' in name:
                should_count = False
            elif 'inv_embed' in name:
                should_count = False
            print(name, tuple(p.shape), p.shape.numel(), should_count, p.requires_grad)
            if should_count:
                param_count += p.shape.numel()
        print('# of parameters = {}'.format(param_count))

        return net

    def forward(self, batch_map):
        x_t = batch_map['text_tokens']

        batch_size, len_t = x_t.shape
        device = x_t.device

        # TODO: Should we project embeddings?
        h_t, _, _, _ = self.encode_text(batch_map)
        z_t = self.project(h_t)
        h_a, y_a, y_a_mask, label_node_ids = self.encode_amr(batch_map)

        size_t = h_t.shape[-1]
        size_a = h_a.shape[-1]

        batch = collections.defaultdict(list)

        def align_and_predict(h_t, z_t, h_a, y_a, info):
            n_a, n_t = info['n_a'], info['n_t']

            def get_h_for_predict(context):
                if self.context == 'xy':
                    h_a_expand = h_a.expand(n_a, n_t, size_a)
                    h_t_expand = h_t.expand(n_a, n_t, size_t)
                    h = torch.cat([h_a_expand, h_t_expand], -1)
                elif self.context == 'x':
                    h = h_t.expand(n_a, n_t, size_t)
                return h

            if self.output_space == 'vocab':
                # output distribution
                h_for_predict = get_h_for_predict(self.context)
                dist = torch.softmax(self.predict(h_for_predict), 2)
                assert dist.shape == (n_a, n_t, self.output_size)

                # (optional) 2nd context
                if self.context_2 is not None:
                    dist_1 = dist

                    h_for_predict_2 = get_h_for_predict(self.context_2)
                    dist_2 = torch.softmax(self.predict(h_for_predict_2), 2)
                    assert dist_2.shape == (n_a, n_t, self.output_size)

                    new_dist = (1 - self.lambda_context) * dist_1 + self.lambda_context * dist_2

                    dist = new_dist

                # output probability
                index = y_a.expand(n_a, n_t, 1)
                p = dist.gather(index=index, dim=-1)
                assert p.shape == (n_a, n_t, 1)

            elif self.output_space == 'graph':
                s_t = self.project_o(h_t)
                logits = torch.sum(s_t.view(1, n_t, 1, size_a) * h_a.view(1, 1, n_a, size_a), -1)
                dist = torch.softmax(logits, 2).expand(n_a, n_t, n_a)
                index = torch.arange(n_a, dtype=torch.long, device=device).view(n_a, 1, 1).expand(n_a, n_t, 1)
                p = dist.gather(index=index, dim=-1)
                assert p.shape == (n_a, n_t, 1)

            # alignment distribution
            if self.prior == 'unif':
                alpha = torch.full((n_a, n_t, 1), 1/n_t, dtype=torch.float, device=device)
            elif self.prior == 'attn':
                alpha = torch.softmax(torch.sum(h_a * z_t, -1, keepdims=True), 1)
            assert alpha.shape == (n_a, n_t, 1)

            # marginal probability
            loss_notreduced = -(torch.sum(alpha * p, 1) + 1e-8).log()
            loss = loss_notreduced.sum(0)

            if self.align_mode == 'posterior':
                align = alpha * p
            elif self.align_mode == 'prior':
                align = alpha

            # posterior regularization
            amr_pairwise_dist = info['amr_pairwise_dist'].float()
            node_id = torch.arange(n_t, device=device)
            text_pairwise_dist = torch.abs(node_id.view(-1, 1) - node_id.view(1, -1)).float()

            align_ = (align + 1e-8).log().softmax(dim=1)
            align_pairwise = align_.view(n_a, 1, n_t, 1) * align_.view(1, n_a, 1, n_t)
            assert align_pairwise.shape == (n_a, n_a, n_t, n_t)

            # for i in range(n_a):
            #     for j in range(n_a):
            #         for k in range(n_t):
            #             for l in range(n_t):
            #                 check = align[i, k] * align[j, l]
            #                 actual = align_pairwise[i, j, k, l]
            #                 assert torch.isclose(check, actual).item()

            expected_text_dist = (amr_pairwise_dist.view(n_a, n_a, 1, 1) * align_pairwise).view(-1, n_t, n_t).sum(0)
            assert expected_text_dist.shape == (n_t, n_t)

            if args.pr_epsilon is not None:
                pr_penalty_tmp = (text_pairwise_dist - expected_text_dist - args.pr_epsilon).clamp(min=0)
            else:
                pr_penalty_tmp = (text_pairwise_dist - expected_text_dist)

            #assert torch.isclose(pr_penalty_tmp, pr_penalty_tmp.transpose(0, 1)).all().item()
            pr_penalty_triu = pr_penalty_tmp.triu()
            pr = pr_penalty_triu.pow(2).sum().view(1)

            return alpha, p, loss, loss_notreduced, align, pr

        def mask_and_apply(i_b, func):
            # get amr mask
            local_y_a_mask = y_a_mask[i_b].view(-1)
            n_a = local_y_a_mask.long().sum().item()

            # get text mask
            local_x_t = x_t[i_b]
            local_x_t_mask = local_x_t != PADDING_IDX
            n_t = local_x_t_mask.long().sum().item()

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
            info['amr_pairwise_dist'] = batch_map['amr_pairwise_dist'][i_b]

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


def align_features_to_words(roberta, features, alignment):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = collections.Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    check = torch.isclose(output.sum(dim=0), features.sum(dim=0), atol=1e-4).all().item()
    assert check is True
    #assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
    return output


def extract_aligned_roberta(roberta, sentence, tokens, return_all_hiddens=False):
    ''' Code inspired from:
       https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
       https://github.com/pytorch/fairseq/issues/1106#issuecomment-547238856

    Aligns roberta embeddings for an input tokenization of words for a sentence

    Inputs:
    1. roberta: roberta fairseq class
    2. sentence: sentence in string
    3. tokens: tokens of the sentence in which the alignment is to be done

    Outputs: Aligned roberta features
    '''

    # tokenize both with GPT-2 BPE and get alignment with given tokens
    bpe_toks = roberta.encode(sentence)
    alignment = alignment_utils.align_bpe_to_words(roberta, bpe_toks, tokens)

    # NOTE: This is a dirty hack to make sure that alignments are contiguous. On rare occasion
    # the space is dropped from tokenization: `Oh , who to believe ??????????????`
    # Ideally, we simply use the alignment as provided rather than prepend the space.
    for i, x in enumerate(alignment):
        if i == 0:
            continue
        if x[0] != alignment[i-1][-1]:
            alignment[i] = [x[0] - 1] + x

    # extract features and align them
    features = roberta.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    features = features.squeeze(0)   #Batch-size = 1
    #aligned_feats = alignment_utils.align_features_to_words(roberta, features, alignment)
    aligned_feats = align_features_to_words(roberta, features, alignment)

    return aligned_feats[1:-1]  #exclude <s> and </s> tokens


def batch_extract_aligned_roberta(roberta, sentence_list, tokens_list, return_all_hiddens=False):
    batch_size = len(sentence_list)

    def pad(bpe_toks_list, pad_idx=0):
        max_length = max([x.shape[-1] for x in bpe_toks_list])
        device = bpe_toks_list[0].device

        new_tensor = torch.full((batch_size, max_length), 0, dtype=torch.long, device=device)

        for i_b, x in enumerate(bpe_toks_list):
            length = x.shape[-1]
            new_tensor[i_b, :length] = x

        return new_tensor

    def fix_alignments(alignment):
        # NOTE: This is a dirty hack to make sure that alignments are contiguous. On rare occasion
        # the space is dropped from tokenization: `Oh , who to believe ??????????????`
        # Ideally, we simply use the alignment as provided rather than prepend the space.
        for i, x in enumerate(alignment):
            if i == 0:
                continue
            if x[0] != alignment[i-1][-1]:
                alignment[i] = [x[0] - 1] + x
        return alignment

    def crop(aligned_feats_list):
        offset = 1
        new_feats = []
        for i_b in range(batch_size):
            length = len(tokens_list[i_b])
            new_feats.append(aligned_feats_list[i_b][offset:offset+length])
        return new_feats


    # tokenize both with GPT-2 BPE and get alignment with given tokens
    bpe_toks_list = [roberta.encode(sentence) for sentence in sentence_list]
    bpe_toks_tensor = pad(bpe_toks_list)
    alignment_list = [fix_alignments(alignment_utils.align_bpe_to_words(roberta, bpe_toks, tokens))
                      for bpe_toks, tokens in zip(bpe_toks_list, tokens_list)]

    # extract features and align them
    features = roberta.extract_features(bpe_toks_tensor, return_all_hiddens=return_all_hiddens)
    aligned_feats_list = [align_features_to_words(roberta, features[i_b], alignment_list[i_b]) for i_b in range(batch_size)]
    cropped_feats = crop(aligned_feats_list)

    return cropped_feats


class RobertaEncoder(nn.Module):
    def __init__(self, embed, hidden_size, dropout_p=0, use_lstm=False):
        super().__init__()

        self.use_lstm = use_lstm
        self.roberta_size = 1024
        self.hidden_size = hidden_size
        self.output_size = hidden_size

        self.project = nn.Linear(self.roberta_size, self.hidden_size, bias=False)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

        if use_lstm:
            self.lstm = Encoder(embed, hidden_size, bidirectional=True, mode='text', dropout_p=dropout_p, input_size=self.output_size)
            self.output_size = self.lstm.output_size

    def forward(self, batch_map):
        #
        tokens = batch_map['text_tokens']
        tokens_mask = None
        labels = None
        labels_mask = None
        label_node_ids = None
        node_ids = None

        #
        batch_size, length = tokens.shape
        roberta_size = self.roberta_size
        hidden_size = self.hidden_size
        device = tokens.device

        #
        roberta = roberta_loader.load()
        shape = (batch_size, length, roberta_size)

        with torch.no_grad():
            output = torch.full(shape, 0, dtype=torch.float, device=device)

            tokens_list = batch_map['text_original_tokens']
            sentence_list = [' '.join(x) for x in tokens_list]
            vectors_list = batch_extract_aligned_roberta(roberta, sentence_list, tokens_list)

            offset = 1
            for i_b, vec in enumerate(vectors_list):
                local_length = len(tokens_list[i_b])
                assert vec.shape == (local_length, roberta_size)
                output[i_b, offset:offset + local_length] = vec

        output = self.project(output)
        output = self.dropout(output)

        if self.use_lstm:
            return self.lstm(batch_map, input_vector=output)

        return output, labels, labels_mask, label_node_ids


class Encoder(nn.Module):
    def __init__(self, embed, hidden_size, bidirectional=True, mode='text', dropout_p=0, input_size=None):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.mode = mode

        self.embed = embed

        if input_size is None:
            input_size = embed.output_size

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                           bidirectional=bidirectional, batch_first=True)

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
        shape = (n, batch_size, hidden_size)
        device = tokens.device

        # compute hidden states
        c0 = torch.full(shape, 0, dtype=torch.float, device=device)
        h0 = torch.full(shape, 0, dtype=torch.float, device=device)

        if input_vector is None:
            e = self.embed(tokens)
        else:
            e = input_vector
        output, _ = self.rnn(e, (h0, c0))
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


def load_checkpoint(path, net):
    toload = torch.load(path)

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
        f.write(json.dumps(metrics))


def default_model_config():
    config = {}
    config['text_emb'] = 'word'
    config['text_enc'] = 'bilstm'
    config['text_project'] = 0
    config['amr_emb'] = 'word'
    config['amr_enc'] = 'bilstm'
    config['amr_project'] = 0
    config['embedding_dim'] = 100
    config['hidden_size'] = 100
    config['dropout'] = 0
    config['output_mode'] = 'linear'
    config['output_space'] = 'vocab'
    config['prior'] = 'attn'
    config['context'] = 'xy'
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


def safe_read(path, check_for_cycles=True, max_length=0, check_for_edges=False, check_for_bpe=True):

    skipped = collections.Counter()
    corpus = read_amr(path).amrs

    if max_length > 0:
        new_corpus = []
        for amr in corpus:
            if len(amr.tokens) > max_length:
                skipped['max-length'] += 1
                continue
            new_corpus.append(amr)
        corpus = new_corpus

    if check_for_cycles:
        new_corpus = []
        t = AMRTokenizer()
        for amr in corpus:
            # UNCOMMENT if you need to support the strange example in AMR2.0 with None node.
            #if ('0.0.2.1.0', ':value', '0.0.2.1.0.0') in amr.edges and '0.0.2.1.0.0' not in amr.nodes:
            #    amr.nodes['0.0.2.1.0.0'] = 'null-02' # it should be None, but that is not in our vocab.
            try:
                t.dfs(amr)
                new_corpus.append(amr)
            except:
                skipped['malformed'] += 1
        corpus = new_corpus

    # TODO: Add support for this type of graph.
    if check_for_edges:
        new_corpus = []
        for amr in corpus:
            if len(amr.edges) == 0:
                skipped['no-edges'] += 1
                continue
            new_corpus.append(amr)
        corpus = new_corpus

    if check_for_bpe:
        roberta = roberta_loader.load()

        def clean(text):
            return text.strip()

        new_corpus = []
        for amr in corpus:
            sentence = ' '.join(amr.tokens) 
            bpe_tokens = roberta.encode(sentence)
            other_tokens = amr.tokens

            try:
                # remove whitespaces to simplify alignment
                bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
                bpe_tokens = [
                    clean(roberta.bpe.decode(x) if x not in {"<s>", ""} else x) for x in bpe_tokens
                ]
                other_tokens = [clean(str(o)) for o in other_tokens]

                # strip leading <s>
                bpe_tokens = bpe_tokens[1:]
                assert "".join(bpe_tokens) == "".join(other_tokens)
            except:
                skipped['bpe'] += 1
                continue

            new_corpus.append(amr)
        corpus = new_corpus


    print('read {}, total = {}, skipped = {}'.format(path, len(corpus), skipped))

    return corpus


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
        for amr in read_amr(args.trn_amr).amrs:
            try:
                t.dfs(amr)
            except:
                import ipdb; ipdb.set_trace()
                pass
        sys.exit()

    # tokenizers
    text_tokenizer, amr_tokenizer = init_tokenizers(text_vocab_file=args.vocab_text, amr_vocab_file=args.vocab_amr)

    # read data
    trn_corpus = safe_read(args.trn_amr, max_length=args.max_length, check_for_bpe=args.check_for_bpe)
    trn_dataset = Dataset(trn_corpus, text_tokenizer=text_tokenizer, amr_tokenizer=amr_tokenizer)

    val_corpus_list = [safe_read(path, max_length=args.val_max_length, check_for_bpe=args.check_for_bpe) for path in args.val_amr]
    val_dataset_list = [Dataset(x, text_tokenizer=text_tokenizer, amr_tokenizer=amr_tokenizer) for x in val_corpus_list]

    # net
    net = Net.from_dataset_and_config(trn_dataset, model_config)
    if args.load is not None:
        load_checkpoint(args.load, net)

    if args.cuda:
        net.cuda()

    def write_align_pretty(corpus, dataset, path, formatter):
        net.eval()

        indices = np.arange(len(corpus))

        print('writing to {}.pretty'.format(os.path.abspath(path)))
        fout = open(path + '.pretty', 'w')
        fout_gold = open(path + '.gold', 'w')

        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
                end = min(start + batch_size, len(corpus))
                batch_indices = indices[start:end]
                items = [dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)

                # forward pass
                model_output = net(batch_map)

                # write
                for i_b, (amr, out) in enumerate(formatter.format(batch_map, model_output, batch_indices)):
                    fout.write(out.strip() + '\n\n')
                    fout_gold.write(amr.toJAMRString().strip() + '\n\n')

        fout.close()
        fout_gold.close()

    def write_align(corpus, dataset, path, formatter):
        net.eval()

        indices = np.arange(len(corpus))

        writer = AlignmentsWriter(path, dataset, formatter)

        with torch.no_grad():
            for start in tqdm(range(0, len(corpus), batch_size), desc='write', disable=False):
                end = min(start + batch_size, len(corpus))
                batch_indices = indices[start:end]
                items = [dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)

                # forward pass
                model_output = net(batch_map)

                # write
                writer.write_batch(batch_indices, batch_map, model_output)

        writer.close()

    if args.write_pretty:

        path = os.path.join(args.log_dir, 'alignment.trn')
        formatter = FormatAlignmentsPretty(trn_dataset)
        write_align_pretty(trn_corpus, trn_dataset, path, formatter)
        sys.exit()

    if args.write_only:

        path = os.path.join(args.log_dir, 'alignment.trn.out')
        formatter = FormatAlignments(trn_dataset)
        write_align(trn_corpus, trn_dataset, path, formatter)
        eval_output = EvalAlignments().run(path + '.gold', path + '.pred')

        with open(path + '.pred.eval', 'w') as f:
            f.write(json.dumps(eval_output))

        sys.exit()

    # optimizer
    opt = optim.Adam(net.parameters(), lr=lr)

    best_metrics = {}

    for epoch in range(max_epoch):

        epoch_metrics = {}
        epoch_metrics['epoch'] = epoch

        # TRAIN

        print('\n\n' + '@' * 40 + '\n' + 'TRAIN' + '\n')

        metrics = collections.defaultdict(list)

        def trn_step(batch_indices, batch_map, info):
            net.train()

            model_output = net(batch_map)

            # neg log likelihood
            loss = 0
            if not args.skip_align_loss:
                loss += torch.cat(model_output['batch_loss'], 0).mean()
            if args.pr > 0 and epoch >= args.pr_after:
                loss += args.pr * torch.cat(model_output['batch_pr'], 0).mean()

            if info['should_clear']:
                opt.zero_grad()
            (loss / info['accum_steps']).backward()
            if info['should_update']:
                opt.step()

            metrics['trn_loss'] += [x.item() for x in model_output['batch_loss']]
            metrics['trn_loss_notreduced'] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()
            metrics['trn_pr'] += [x.item() for x in model_output['batch_pr']]

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
            batch_map = batchify(items, cuda=args.cuda)
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

        # VALID

        if args.skip_validation:
            continue

        print('\n\n' + '@' * 40 + '\n' + 'VALID' + '\n')

        for i_valid, val_dataset in enumerate(val_dataset_list):

            print('[{}]'.format(i_valid))
            print(args.val_amr[i_valid])

            val_corpus = val_corpus_list[i_valid]

            formatter = FormatAlignments(val_dataset)
            path = os.path.join(args.log_dir, 'alignment.epoch_{}.val_{}.out'.format(epoch, i_valid))
            writer = AlignmentsWriter(path, val_dataset, formatter, enabled=args.write_validation)

            def val_step(batch_indices, batch_map):
                with torch.no_grad():
                    net.eval()

                    model_output = net(batch_map)

                    # neg log likelihood
                    loss = torch.cat(model_output['batch_loss'], 0).mean()

                    metrics['val_{}_loss'.format(i_valid)] += [x.item() for x in model_output['batch_loss']]
                    metrics['val_{}_loss_notreduced'.format(i_valid)] += torch.cat(model_output['batch_loss_notreduced']).view(-1).tolist()

                    del loss

                    writer.write_batch(batch_indices, batch_map, model_output)

            # batch iterator
            indices = [i for i, _ in enumerate(sorted(val_corpus, key=lambda x: -len(x.tokens)))]

            for start in tqdm(range(0, len(indices), batch_size), desc='val_{}-epoch[{}]'.format(i_valid, epoch), disable=not args.verbose):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                items = [val_dataset[idx] for idx in batch_indices]
                batch_map = batchify(items, cuda=args.cuda)
                val_step(batch_indices, batch_map)

            writer.close()

            val_loss = np.mean(metrics['val_{}_loss'.format(i_valid)])
            val_loss_notreduced = np.mean(metrics['val_{}_loss_notreduced'.format(i_valid)])
            val_ppl = 2 ** (val_loss_notreduced / np.log(2))

            epoch_metrics['val_{}_loss'.format(i_valid)] = val_loss
            epoch_metrics['val_{}_loss_notreduced'.format(i_valid)] = val_loss_notreduced
            epoch_metrics['val_{}_ppl'.format(i_valid)] = val_ppl

            print('val_{} epoch = {}, loss = {:.3f}, loss-nr = {:.3f}, ppl = {:.3f}'.format(
                i_valid, epoch, val_loss, val_loss_notreduced, val_ppl))

            # eval alignments
            eval_output = EvalAlignments().run(path + '.gold', path + '.pred')

            val_token_recall = eval_output['Corpus Recall']['recall']
            epoch_metrics['val_{}_token_recall'.format(i_valid)] = val_token_recall

            val_recall = eval_output['Corpus Recall using spans for gold']['recall']
            epoch_metrics['val_{}_recall'.format(i_valid)] = val_recall

            # save checkpoint

            if check_and_update_best(best_metrics, 'val_{}_recall'.format(i_valid), val_recall, compare='gt'):
                save_checkpoint(os.path.join(args.log_dir, 'model.best.val_{}_recall.pt'.format(i_valid)), trn_dataset, net, metrics=best_metrics)

                if i_valid == 1 and args.jbsub_eval:

                    stdout_path = os.path.join(args.log_dir, 'eval.stdout.txt')
                    stderr_path = os.path.join(args.log_dir, 'eval.stderr.txt')
                    script_path = os.path.join(args.log_dir, 'eval_script.txt')
                    cmd = 'jbsub -cores 1+1 -mem 30g -q x86_6h -out {} -err {} bash {}'.format(stdout_path, stderr_path, script_path)
                    os.system(cmd)

        save_metrics(os.path.join(args.log_dir, 'model.epoch_{}.metrics'.format(epoch)), epoch_metrics)


if __name__ == '__main__':
    args = argument_parser()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    print(args.__dict__)

    os.system('mkdir -p {}'.format(args.log_dir))

    with open(os.path.join(args.log_dir, 'flags.json'), 'w') as f:
        f.write(json.dumps(args.__dict__))

    main(args)
