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


class TreeLSTM(nn.Module):
    def __init__(self, embed, size):
        super().__init__()

        chunk_size = size // 2

        self.embed = embed
        self.chunk_size = chunk_size
        self.output_size = 2 * chunk_size
        self.input_size = input_size = embed.output_size

        self.W_node = nn.Linear(input_size, size, bias=False)

        # treelstm
        self.W_i, self.U_i = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_f, self.U_f = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_o, self.U_o = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_u, self.U_u = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def update_with_edge_label(self, z, e_label):
        return z

    def compute_node_features(self, node_tokens):
        return self.W_node(self.embed(node_tokens))

    def compute_z_in(self, z_node, z_children, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node
        z_node = z_node[:, 0]

        chunk_size = self.chunk_size

        assert z_children.shape[-1] == 2 * chunk_size, (z_children.shape, chunk_size)

        h_k, c_k = z_children.split(chunk_size, dim=2)

        h_tilde = h_k.sum(dim=1)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j = torch.sigmoid(self.W_i(z_node) + self.U_i(h_tilde))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j = torch.sigmoid(self.W_o(z_node) + self.U_o(h_tilde))
        u_j = torch.tanh(self.W_u(z_node) + self.U_u(h_tilde))

        c_j = i_j * u_j + torch.sum(f_j_k * c_k, dim=1)
        h_j = o_j * torch.tanh(c_j)

        out = torch.cat([h_j, c_j], -1)

        return out, h_tilde

    def compute_z_in_for_outside(self, z_node, z_children, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node
        z_node = z_node[:, 0]

        chunk_size = self.chunk_size

        assert z_children.shape[-1] == 2 * chunk_size, (z_children.shape, chunk_size)

        h_k, c_k = z_children.split(chunk_size, dim=2)

        h_tilde = h_k.sum(dim=1, keepdims=True)
        h_tilde_minus_1 = h_tilde - h_k
        assert h_tilde_minus_1.shape == (batch_size, n, chunk_size)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j = torch.sigmoid(self.W_i(z_node_repeat) + self.U_i(h_tilde_minus_1))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j = torch.sigmoid(self.W_o(z_node_repeat) + self.U_o(h_tilde_minus_1))
        u_j = torch.tanh(self.W_u(z_node_repeat) + self.U_u(h_tilde_minus_1))

        assert i_j.shape == (batch_size, n, chunk_size)
        assert f_j_k.shape == (batch_size, n, chunk_size)

        c_j_prev_tmp = f_j_k * c_k
        c_j_prev = torch.sum(c_j_prev_tmp, dim=1, keepdims=True)
        c_j_prev_minus_1 = c_j_prev - c_j_prev_tmp

        c_j = i_j * u_j + c_j_prev_minus_1
        assert c_j.shape == (batch_size, n, chunk_size)

        h_j = o_j * torch.tanh(c_j)
        assert h_j.shape == (batch_size, n, chunk_size)

        out = torch.cat([h_j, c_j], -1).view(batch_size * n, size)

        return out

    def compute_z_for_outside_with_parent(self, z_node, z_children, z_parent, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node
        z_node = z_node[:, 0]

        chunk_size = size // 2

        h_k, c_k = z_children.split(chunk_size, dim=2)
        h_parent, c_parent = z_parent.split(chunk_size, dim=2) # is repeated

        h_tilde = h_k.sum(dim=1, keepdims=True)
        h_tilde_minus_1 = h_tilde - h_k + h_parent
        assert h_tilde_minus_1.shape == (batch_size, n, chunk_size)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j =   torch.sigmoid(self.W_i(z_node_repeat) + self.U_i(h_tilde_minus_1))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j =   torch.sigmoid(self.W_o(z_node_repeat) + self.U_o(h_tilde_minus_1))
        u_j =      torch.tanh(self.W_u(z_node_repeat) + self.U_u(h_tilde_minus_1))

        f_parent = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_parent))

        assert i_j.shape == (batch_size, n, chunk_size)
        assert f_j_k.shape == (batch_size, n, chunk_size)

        c_j_prev_tmp = f_j_k * c_k
        c_j_prev = torch.sum(c_j_prev_tmp, dim=1, keepdims=True)
        c_j_prev_minus_1 = c_j_prev - c_j_prev_tmp + (f_parent * c_parent)

        c_j = i_j * u_j + c_j_prev_minus_1
        assert c_j.shape == (batch_size, n, chunk_size)

        h_j = o_j * torch.tanh(c_j)
        assert h_j.shape == (batch_size, n, chunk_size)

        out = torch.cat([h_j, c_j], -1).view(batch_size * n, size)

        return out

    def compute_z_out(self, z_parent, e_label, z_out_step_0):
        return z_out_step_0

    def compute_z_children(self, z_node, z_in, is_leaf=False):
        if is_leaf:
            return torch.tanh(z_node)
        return z_in

    def compute_z_ctxt_for_inside(self, z_node, z_in, h_tilde, is_leaf=False):
        if is_leaf:
            return z_node.clone().detach().fill_(0).split(self.chunk_size, dim=-1)[0]
        return torch.tanh(self.U_u(h_tilde))

    def compute_z_parent(self, z_node, z_out, is_root=False):
        if is_root:
            return z_node.clone().detach().fill_(0)
        return z_out

    def compute_z_ctxt_for_outside(self, z_node, z_out, is_root=False):
        if is_root:
            ctxt = z_node.clone().detach().fill_(0)
        else:
            ctxt = z_out
        h, c = ctxt.split(self.chunk_size, dim=-1)
        return h


class TreeLSTM_v3(nn.Module):
    def __init__(self, embed, size):
        super().__init__()

        chunk_size = size // 2

        self.embed = embed
        self.chunk_size = chunk_size
        self.output_size = 2 * chunk_size
        self.input_size = input_size = embed.output_size

        self.W_node = nn.Linear(input_size, size, bias=False)

        # treelstm
        self.W_i, self.U_i = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_f, self.U_f = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_o, self.U_o = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)
        self.W_u, self.U_u = nn.Linear(size, chunk_size, bias=False), nn.Linear(chunk_size, chunk_size, bias=True)

        self.V_i = nn.Linear(chunk_size, chunk_size, bias=False)
        self.V_f = nn.Linear(chunk_size, chunk_size, bias=False)
        self.V_o = nn.Linear(chunk_size, chunk_size, bias=False)
        self.V_u = nn.Linear(chunk_size, chunk_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def update_with_edge_label(self, z, e_label):
        return z

    def compute_node_features(self, node_tokens):
        return self.W_node(self.embed(node_tokens))

    def compute_z_in(self, z_node, z_children, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node
        z_node = z_node[:, 0]

        chunk_size = self.chunk_size

        assert z_children.shape[-1] == 2 * chunk_size, (z_children.shape, chunk_size)

        h_k, c_k = z_children.split(chunk_size, dim=2)

        h_tilde = h_k.sum(dim=1)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j = torch.sigmoid(self.W_i(z_node) + self.U_i(h_tilde))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j = torch.sigmoid(self.W_o(z_node) + self.U_o(h_tilde))
        u_j = torch.tanh(self.W_u(z_node) + self.U_u(h_tilde))

        c_j = i_j * u_j + torch.sum(f_j_k * c_k, dim=1)
        h_j = o_j * torch.tanh(c_j)

        out = torch.cat([h_j, c_j], -1)

        return out, h_tilde

    def compute_z_in_for_outside(self, z_node, z_children, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node
        z_node = z_node[:, 0]

        chunk_size = self.chunk_size

        assert z_children.shape[-1] == 2 * chunk_size, (z_children.shape, chunk_size)

        h_k, c_k = z_children.split(chunk_size, dim=2)

        h_tilde = h_k.sum(dim=1, keepdims=True)
        h_tilde_minus_1 = h_tilde - h_k
        assert h_tilde_minus_1.shape == (batch_size, n, chunk_size)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j = torch.sigmoid(self.W_i(z_node_repeat) + self.U_i(h_tilde_minus_1))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j = torch.sigmoid(self.W_o(z_node_repeat) + self.U_o(h_tilde_minus_1))
        u_j = torch.tanh(self.W_u(z_node_repeat) + self.U_u(h_tilde_minus_1))

        assert i_j.shape == (batch_size, n, chunk_size)
        assert f_j_k.shape == (batch_size, n, chunk_size)

        c_j_prev_tmp = f_j_k * c_k
        c_j_prev = torch.sum(c_j_prev_tmp, dim=1, keepdims=True)
        c_j_prev_minus_1 = c_j_prev - c_j_prev_tmp

        c_j = i_j * u_j + c_j_prev_minus_1
        assert c_j.shape == (batch_size, n, chunk_size)

        h_j = o_j * torch.tanh(c_j)
        assert h_j.shape == (batch_size, n, chunk_size)

        out = torch.cat([h_j, c_j], -1).view(batch_size * n, size)

        return out

    def compute_z_for_outside_with_parent(self, z_node, z_children, z_parent, e_label):
        batch_size, n, size = z_children.shape

        z_node_repeat = z_node

        chunk_size = size // 2

        h_k, c_k = z_children.split(chunk_size, dim=2)
        h_parent, c_parent = z_parent.split(chunk_size, dim=2) # is repeated

        h_tilde = h_k.sum(dim=1, keepdims=True)
        # h_tilde_minus_1 = h_tilde - h_k + h_parent
        h_tilde_minus_1 = h_tilde - h_k
        assert h_tilde_minus_1.shape == (batch_size, n, chunk_size)

        # TODO: Add bias.
        # TODO: Speed up.
        i_j =   torch.sigmoid(self.W_i(z_node_repeat) + self.U_i(h_tilde_minus_1) + self.V_i(h_parent))
        f_j_k = torch.sigmoid(self.W_f(z_node_repeat) + self.U_f(h_k))
        o_j =   torch.sigmoid(self.W_o(z_node_repeat) + self.U_o(h_tilde_minus_1) + self.V_o(h_parent))
        u_j =      torch.tanh(self.W_u(z_node_repeat) + self.U_u(h_tilde_minus_1) + self.V_u(h_parent))

        f_parent = torch.sigmoid(self.W_f(z_node_repeat) + self.V_f(h_parent))

        assert i_j.shape == (batch_size, n, chunk_size)
        assert f_j_k.shape == (batch_size, n, chunk_size)

        c_j_prev_tmp = f_j_k * c_k
        c_j_prev = torch.sum(c_j_prev_tmp, dim=1, keepdims=True)
        c_j_prev_minus_1 = c_j_prev - c_j_prev_tmp + (f_parent * c_parent)

        c_j = i_j * u_j + c_j_prev_minus_1
        assert c_j.shape == (batch_size, n, chunk_size)

        h_j = o_j * torch.tanh(c_j)
        assert h_j.shape == (batch_size, n, chunk_size)

        out = torch.cat([h_j, c_j], -1).view(batch_size * n, size)

        return out

    def compute_z_out(self, z_parent, e_label, z_out_step_0):
        return z_out_step_0

    def compute_z_children(self, z_node, z_in, is_leaf=False):
        if is_leaf:
            return torch.tanh(z_node)
        return z_in

    def compute_z_ctxt_for_inside(self, z_node, z_in, h_tilde, is_leaf=False):
        if is_leaf:
            return z_node.clone().detach().fill_(0).split(self.chunk_size, dim=-1)[0]
        return torch.tanh(self.U_u(h_tilde))

    def compute_z_parent(self, z_node, z_out, is_root=False):
        if is_root:
            return z_node.clone().detach().fill_(0)
        return z_out

    def compute_z_ctxt_for_outside(self, z_node, z_out, is_root=False):
        if is_root:
            ctxt = z_node.clone().detach().fill_(0)
        else:
            ctxt = z_out
        h, c = ctxt.split(self.chunk_size, dim=-1)
        return h


class TreeEncoder(nn.Module):
    def __init__(self, embed, size, mode='tree_lstm', dropout_p=0):
        super().__init__()

        if mode == 'tree_lstm':
            self.enc = TreeLSTM(embed, size)
        elif mode == 'tree_lstm_v3':
            self.enc = TreeLSTM_v3(embed, size)

        self.embed = embed
        self.size = size
        self.output_size = self.enc.output_size

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
            z_node = self.rg.ndata['node_features'][edges.src['_ID']]
            e_label = edges.data['edge_tokens']
            z_out_step_0 = z_parent.clone().detach().fill_(0)

            src_id_c = edges.data['src_c']
            unique_c = torch.unique(src_id_c)
            for c in unique_c:
                m = src_id_c == c
                local_z_parent = z_parent[m].view(-1, c, self.size)
                local_z_children = z_children[m].view(-1, c, self.size)
                local_z_node = z_node[m].view(-1, c, self.size)
                local_e_label = e_label[m].view(-1, c)
                z_out_step_0[m] = self.enc.compute_z_for_outside_with_parent(
                    local_z_node, local_z_children, local_z_parent, local_e_label)
                # z_in[m] = self.enc.compute_z_in_for_outside(local_z_node, local_z_children, local_e_label)

            msg = {'z_parent': z_parent, 'e_label': e_label, 'z_out_step_0': z_out_step_0}

        return msg

    def reduce_func(self, nodes):

        if self.mode == 'inside':
            # Aggregate multiple incoming messages.
            z_children = nodes.mailbox['z_children']
            z_node = nodes.mailbox['z_node']
            e_label = nodes.mailbox['e_label']
            z_in, h_tilde = self.enc.compute_z_in(z_node, z_children, e_label) # has duplicates of z_node
            msg = {'z_in': z_in, 'h_tilde': h_tilde}

        elif self.mode == 'outside':
            # Combine inside and outside.
            z_out_step_0 = nodes.mailbox['z_out_step_0'].squeeze(1)
            z_parent = nodes.mailbox['z_parent'].squeeze(1)
            e_label = nodes.mailbox['e_label'].squeeze(1)
            z_out = self.enc.compute_z_out(z_parent, e_label, z_out_step_0)
            msg = {'z_out': z_out}

        return msg

    def apply_node_func(self, nodes):
        if self.mode == 'inside':
            is_leaf = 'z_in' not in nodes.data
            z_node = nodes.data['node_features']
            z_in = nodes.data['z_in'] if not is_leaf else None
            h_tilde = nodes.data['h_tilde'] if not is_leaf else None

            # Compute z_children.
            z_children = self.enc.compute_z_children(z_node, z_in, is_leaf=is_leaf) # has single instance of z_node

            # Use this representation for node prediction.
            z_ctxt = self.enc.compute_z_ctxt_for_inside(z_node, z_in, h_tilde, is_leaf=is_leaf)

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

    def forward(self, batch_map, outside=True):
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
        def inside_func():
            self.mode = 'inside'
            dgl.prop_nodes_topo(rg, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func)

        def outside_func():
            self.mode = 'outside'
            dgl.prop_nodes_topo(g, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func)

        inside_func()
        if outside:
            self.precompute_counts_for_outside()
            outside_func()

        # cleanup
        self.g = None
        self.rg = None

        h = rg.ndata['z_ctxt_inside']
        if outside:
            h = torch.cat([rg.ndata['z_ctxt_inside'], g.ndata['z_ctxt_outside']], -1)

        # reshape and compute labels / masks
        device = h.device
        batch_size = g.batch_size
        size = self.output_size // 2
        if outside:
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


class TreeEncoder_v2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.enc_in = TreeEncoder(*args, **kwargs)
        self.enc_out = TreeEncoder(*args, **kwargs)

        self.output_size = self.enc_in.output_size // 2 + self.enc_out.output_size

    @property
    def embed(self):
        return self.enc_in.embed

    def forward(self, batch_map):
        output_in, labels, labels_mask, label_node_ids = self.enc_in(batch_map, outside=False)
        assert len(output_in.shape) == 3
        output_out, _, _, _ = self.enc_out(batch_map, outside=True)
        output = torch.cat([output_in, output_out], -1)
        return output, labels, labels_mask, label_node_ids

