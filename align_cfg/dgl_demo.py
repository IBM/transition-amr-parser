import os
import sys

os.environ['DGLBACKEND'] = 'pytorch'

import dgl

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transition_amr_parser.io import read_amr


def read_vocab(path):
    mapping = {}

    with open(path) as f:
        for i, line in enumerate(f):
            mapping[line.rstrip()] = i

    return mapping


class Net(nn.Module):
    def __init__(self, amr_e, t_cell, size, output_size):
        super().__init__()
        self.cache = {}

        self.register('amr_e', amr_e)
        self.register('t_cell', t_cell)

        self.predict = nn.Linear(2 * size, output_size)

    def register(self, key, val):
        setattr(self, key, val)
        val.cache = self.cache

    def update_cache(self, key, val):
        self.cache[key] = val


class TreeCell(nn.Module):
    def __init__(self, size):
        super().__init__()

        # inside
        self.inside_node_f = nn.Linear(size, size, bias=False)
        self.inside_incoming_f = nn.Linear(size, size)

        # outside
        self.outside_node_f = nn.Linear(size, size, bias=False)
        self.outside_incoming_f = nn.Linear(size, size)

    def message_func(self, edges):
        # TODO: Incorporate edge labels.
        if edges.batch_size() == 0:
            return None

        if self.cache['mode'] == 'inside':
            msg = {'inside_individual_h': edges.src['inside_h']}

        elif self.cache['mode'] == 'outside':
            ih = edges.src['outside_h'].clone().detach().fill_(0)

            # TODO: Replace for loop.
            for i_b in range(edges.batch_size()):
                src_id = edges.src['_ID'][i_b].item()
                dst_id = edges.dst['_ID'][i_b].item()

                for i_b_2 in range(edges.batch_size()):
                    if i_b == i_b_2:
                        continue
                    src_id_2 = edges.src['_ID'][i_b_2].item()
                    if src_id != src_id_2:
                        continue
                    dst_id_2 = edges.dst['_ID'][i_b_2].item()

                    individual_ih = self.cache['batch_map']['rg'].ndata['inside_h'][dst_id_2]

                    ih[i_b] += individual_ih

            msg = {'outside_individual_h': edges.src['outside_h'], 'outside_individual_ih': ih}

        return msg

    def reduce_func(self, nodes):

        if self.cache['mode'] == 'inside':
            # Aggregate multiple incoming messages.
            h = self.inside_incoming_f(nodes.mailbox['inside_individual_h']).sum(dim=1)
            msg = {'inside_reduced_h': h}

        elif self.cache['mode'] == 'outside':
            # Combine inside and outside.
            h = self.outside_incoming_f(nodes.mailbox['outside_individual_h'].squeeze(1)) + nodes.mailbox['outside_individual_ih'].squeeze(1)
            msg = {'outside_reduced_h': h}

        return msg

    def apply_node_func(self, nodes):
        if self.cache['mode'] == 'inside':
            is_leaf = 'inside_reduced_h' not in nodes.data

            if is_leaf:
                # Computation at leaves.
                h = torch.tanh(self.inside_node_f(nodes.data['node_feat']))
            else:
                # Combine node with incoming representations.
                h = torch.tanh(self.inside_node_f(nodes.data['node_feat']) + nodes.data['inside_reduced_h'])

            # Exclude: Use this representation for node prediction.
            if is_leaf:
                # Computation at leaves.
                h_exclude = nodes.data['node_feat'].clone().detach().fill_(0)
            else:
                # Combine node with incoming representations.
                h_exclude = nodes.data['inside_reduced_h']

            msg = {'inside_h': h, 'inside_exclude_h': h_exclude}

        elif self.cache['mode'] == 'outside':
            is_root = 'outside_reduced_h' not in nodes.data

            if is_root:
                # Computation at leaves.
                h = torch.tanh(self.outside_node_f(nodes.data['node_feat']))
            else:
                # Combine node with incoming representations.
                h = torch.tanh(self.outside_node_f(nodes.data['node_feat']) + nodes.data['outside_reduced_h'])

            # Exclude: Use this representation for node prediction.
            if is_root:
                # Computation at leaves.
                h_exclude = nodes.data['node_feat'].clone().detach().fill_(0)
            else:
                # Combine node with incoming representations.
                h_exclude = nodes.data['outside_reduced_h']

            msg = {'outside_h': h, 'outside_exclude_h': h_exclude}

        return msg


def safe_read(path):
    corpus = read_amr(path).amrs

    new_corpus = []
    for amr in corpus:
        if len(amr.edges) == 0:
            continue
        new_corpus.append(amr)
    corpus = new_corpus

    return corpus


def demo():
    in_amr = os.path.expanduser('~/data/AMR2.0/aligned/cofill/dev.txt')
    amr_TO_idx = read_vocab('./align_cfg/vocab.amr.txt')
    text_TO_idx = read_vocab('./align_cfg/vocab.text.txt')
    batch_size = 4
    size = 10
    seed = 11

    np.random.seed(seed)
    torch.manual_seed(seed)

    corpus = safe_read(in_amr)
    np.random.shuffle(corpus)

    # embeddings
    amr_embeddings = nn.Embedding(len(amr_TO_idx), size, padding_idx=0)
    amr_embeddings.weight.data.normal_()

    # net
    t_cell = TreeCell(size=size)
    net = Net(amr_e=amr_embeddings, t_cell=t_cell, size=size, output_size=len(amr_TO_idx))
    opt = optim.Adam(net.parameters(), lr=2e-3)

    def batchify(items):
        batch = []

        for i_b, x in enumerate(items):
            # init graph
            g = dgl.DGLGraph()

            # add node structure
            node_ids = list(sorted(x.nodes.keys()))
            node_TO_idx = {k: i for i, k in enumerate(node_ids)}
            N = len(node_ids)
            g.add_nodes(N)

            # add node features
            node_labels_ = [x.nodes[k] for k in node_ids]
            node_labels = torch.tensor([amr_TO_idx[tok] for tok in node_labels_], dtype=torch.long)
            node_embed = net.amr_e(node_labels)
            g.nodes[:].data['node_toks'] = node_labels.view(N)
            g.nodes[:].data['node_feat'] = node_embed.view(N, -1)

            # add edge structure
            def convert_to_tree(amr):
                seen = set()

                # Note: If we skip adding the root, then cycles may form.
                seen.add(amr.root)

                def sortkey(x):
                    s, y, t = x

                    return (s, t)

                for e in sorted(amr.edges, key=sortkey):
                    s, y, t = e
                    if t in seen:
                        continue
                    seen.add(t)
                    yield e

            src, tgt, labels = [], [], []
            for s, y, t in convert_to_tree(x):
                src.append(node_TO_idx[s])
                tgt.append(node_TO_idx[t])
                labels.append(amr_TO_idx[y])
            g.add_edges(src, tgt)
            E = len(src)
            assert E > 0

            # add edge features
            edge_labels = torch.tensor(labels, dtype=torch.long)
            edge_embed = net.amr_e(edge_labels)
            g.edges[:].data['edge_toks'] = edge_labels.view(E)
            g.edges[:].data['edge_feat'] = edge_embed.view(E, -1)

            batch.append(g)

        g = dgl.batch(batch)

        batch_map = {}
        batch_map['g'] = g
        batch_map['rg'] = dgl.reverse(g)
        batch_map['items'] = items

        return batch_map


    def train_step(batch_map):

        g, rg = batch_map['g'], batch_map['rg']

        net.update_cache('batch_map', batch_map)

        def inside():
            net.update_cache('mode', 'inside')
            dgl.prop_nodes_topo(rg, net.t_cell.message_func, net.t_cell.reduce_func, apply_node_func=net.t_cell.apply_node_func)

        def outside():
            net.update_cache('mode', 'outside')
            dgl.prop_nodes_topo(g, net.t_cell.message_func, net.t_cell.reduce_func, apply_node_func=net.t_cell.apply_node_func)

        inside()
        outside()

        h = torch.cat([rg.ndata['inside_exclude_h'], g.ndata['outside_exclude_h']], -1)
        y = g.ndata['node_toks']
        logits = net.predict(h)
        loss = nn.CrossEntropyLoss()(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.item())

    for start in range(0, len(corpus), batch_size):
        end = min(start + batch_size, len(corpus))

        items = corpus[start:end]
        batch_map = batchify(items)

        train_step(batch_map)


if __name__ == '__main__':
    demo()
