import collections
import os

os.environ['DGLBACKEND'] = 'pytorch'

try:
    import dgl
except:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TreeRNN(nn.Module):
    def __init__(self, embed, size):
        super().__init__()

        self.embed = embed
        self.size = size
        self.output_size = 2 * size
        self.input_size = input_size = embed.output_size

        self.W_node = nn.Linear(input_size, size, bias=False)

        # inside
        self.W_in = nn.Linear(size, size, bias=False)

        # outside
        self.W_out = nn.Linear(size, size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def update_with_edge_label(self, z, e_label):
        return z

    def compute_node_features(self, node_tokens):
        return self.W_node(self.embed(node_tokens))

    def compute_z_in(self, z_node, z_children, e_label):
        batch_size, n, size = z_children.shape
        z_children = self.update_with_edge_label(z_children, e_label)
        z_in = torch.tanh(self.W_in(z_children).sum(1))
        return z_in

    def compute_z_in_for_outside(self, z_children, e_label):
        batch_size, n, size = z_children.shape
        z_children = self.update_with_edge_label(z_children, e_label)
        z_sum = z_children.sum(1, keepdims=True)
        z_in_tmp = (z_sum - z_children).view(batch_size * n, size)
        z_in = torch.tanh(self.W_in(z_in_tmp))
        return z_in

    def compute_z_out(self, z_parent, e_label, z_in):
        z_parent = self.update_with_edge_label(z_parent, e_label)
        z_out = torch.tanh(self.W_out(z_parent) + z_in)
        return z_out

    def compute_z_children(self, z_node, z_in, is_leaf=False):
        if is_leaf:
            return torch.tanh(z_node)
        return torch.tanh(z_node + z_in)

    def compute_z_ctxt_for_inside(self, z_node, z_in, is_leaf=False):
        if is_leaf:
            return z_node.clone().detach().fill_(0)
        return z_in

    def compute_z_parent(self, z_node, z_out, is_root=False):
        if is_root:
            return torch.tanh(z_node)
        return torch.tanh(z_node + z_out)

    def compute_z_ctxt_for_outside(self, z_node, z_out, is_root=False):
        if is_root:
            return z_node.clone().detach().fill_(0)
        return z_out


class TreeEncoder(nn.Module):
    def __init__(self, embed, size, mode='tree_rnn', dropout_p=0):
        super().__init__()

        self.embed = embed
        self.size = size
        self.output_size = 2 * size

        if mode == 'tree_rnn':
            self.enc = TreeRNN(embed, size)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    @property
    def device(self):
        return next(self.parameters()).device

    def precompute_counts_for_outside(self):
        device = self.device
        src, dist = self.g.edges()
        src = src.cpu().numpy()
        u, inv, c = np.unique(src, return_inverse=True, return_counts=True)
        self.g.edata['src_c'] = torch.tensor(c[inv].tolist(), dtype=torch.long, device=device)

    def message_func(self, edges):
        device = self.device

        if edges.batch_size() == 0:
            return None

        if self.mode == 'inside':
            z_children = edges.src['z_children']
            z_node = edges.dst['node_features']
            e_label = edges.data['edge_tokens']

            msg = {'z_children': z_children, 'z_node': z_node, 'e_label': e_label}

        elif self.mode == 'outside':
            z_parent = edges.src['z_parent']
            z_children = self.rg.ndata['z_children'][edges.dst['_ID']]
            e_label = edges.data['edge_tokens']
            z_in = z_parent.clone().detach().fill_(0)

            src_id_c = edges.data['src_c']
            unique_c = torch.unique(src_id_c)
            for c in unique_c:
                m = src_id_c == c
                local_z_children = z_children[m].view(-1, c, self.size)
                local_e_label = e_label[m].view(-1, c)
                z_in[m] = self.enc.compute_z_in_for_outside(local_z_children, local_e_label)

            msg = {'z_parent': z_parent, 'e_label': e_label, 'z_in': z_in}

        return msg

    def reduce_func(self, nodes):

        if self.mode == 'inside':
            # Aggregate multiple incoming messages.
            z_children = nodes.mailbox['z_children']
            z_node = nodes.mailbox['z_node']
            e_label = nodes.mailbox['e_label']
            z_in = self.enc.compute_z_in(z_node, z_children, e_label) # has duplicates of z_node
            msg = {'z_in': z_in}

        elif self.mode == 'outside':
            # Combine inside and outside.
            z_in = nodes.mailbox['z_in'].squeeze(1)
            z_parent = nodes.mailbox['z_parent'].squeeze(1)
            e_label = nodes.mailbox['e_label'].squeeze(1)
            z_out = self.enc.compute_z_out(z_parent, e_label, z_in)
            msg = {'z_out': z_out}

        return msg

    def apply_node_func(self, nodes):
        if self.mode == 'inside':
            is_leaf = 'z_in' not in nodes.data
            z_node = nodes.data['node_features']
            z_in = nodes.data['z_in'] if not is_leaf else None

            # Compute z_children.
            z_children = self.enc.compute_z_children(z_node, z_in, is_leaf=is_leaf) # has single instance of z_node

            # Use this representation for node prediction.
            z_ctxt = self.enc.compute_z_ctxt_for_inside(z_node, z_in, is_leaf=is_leaf)

            msg = {'z_children': z_children, 'z_ctxt_inside': z_ctxt}

        elif self.mode == 'outside':
            is_root = 'z_out' not in nodes.data
            z_node = nodes.data['node_features']
            z_out = nodes.data['z_out'] if not is_root else None

            # Compute z_parent.
            z_parent = self.enc.compute_z_parent(z_node, z_out, is_root=is_root)

            # Use this representation for node prediction.
            z_ctxt = self.enc.compute_z_ctxt_for_outside(z_node, z_out, is_root=is_root)

            msg = {'z_parent': z_parent, 'z_ctxt_outside': z_ctxt}

        return msg

    def forward(self, batch_map):
        device = batch_map['device']

        #
        g = dgl.batch([x['g'] for x in batch_map['items']])
        g = g.to(device)

        rg = dgl.reverse(g)
        for k, v in g.edata.items():
            rg.edata[k] = v

        #
        self.g = g
        self.rg = rg

        # embed
        node_features = self.enc.compute_node_features(g.ndata['node_tokens'])
        g.ndata['node_features'] = node_features
        rg.ndata['node_features'] = node_features

        # compute node features
        def inside():
            self.mode = 'inside'
            dgl.prop_nodes_topo(rg, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func)

        def outside():
            self.mode = 'outside'
            dgl.prop_nodes_topo(g, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func)

        inside()
        self.precompute_counts_for_outside()
        outside()

        # cleanup
        self.g = None
        self.rg = None

        h = torch.cat([rg.ndata['z_ctxt_inside'], g.ndata['z_ctxt_outside']], -1)

        # reshape and compute labels / masks
        device = h.device
        batch_size = g.batch_size
        size = self.output_size

        length_per_example = g.batch_num_nodes()
        length = length_per_example.max().item()

        new_h = torch.zeros(batch_size, length, size, dtype=torch.float, device=device)
        labels = torch.zeros(batch_size, length, dtype=torch.long, device=device)
        labels_mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        label_node_ids = torch.full((batch_size, length), -1, dtype=torch.long, device=device)

        offset = 0
        for i_b in range(batch_size):
            n = length_per_example[i_b].item()
            start = offset
            end = offset + n
            offset += n

            new_h[i_b, :n] = h[start:end]
            labels[i_b, :n] = g.ndata['node_tokens'][start:end]
            labels_mask[i_b, :n] = True
            label_node_ids[i_b, :n] = g.ndata['node_ids'][start:end]

        output = new_h
        output = self.dropout(output)

        return output, labels, labels_mask, label_node_ids
