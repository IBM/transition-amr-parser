import torch
import torch.nn as nn

try:
    from torch_geometric.data import Batch, Data
    import torch_geometric.nn as gnn
except:
    pass

from vocab_definitions import MaskInfo


class GCNEncoder(nn.Module):
    def __init__(self, embed, size, mode='gcn', dropout_p=0, num_layers=None):
        super().__init__()

        num_layers = 2 if num_layers is None else num_layers

        self.enc = GCN(embed, size, mode=mode, num_layers=num_layers)

        self.embed = embed
        self.size = size
        self.output_size = self.enc.output_size

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch_map):
        """
        Returns:

            - output: BxLxD
            - labels: BxL from AMR vocab.
            - labels_mask: True if label, else False. Useful for padding and edge labels.
            - label_node_ids: Roughly, torch.arange(len(nodes)).
        """
        device = batch_map['device']
        batch_size = len(batch_map['items'])

        data = Batch.from_data_list([x['geometric_data'].clone().to(device) for x in batch_map['items']])

        node_lengths = batch_map['amr_node_mask'].sum(-1).tolist()
        edge_lengths = [x['geometric_data'].y.shape[0] - n for x, n in zip(batch_map['items'], node_lengths)]
        max_node_length = max(node_lengths)
        size = self.enc.output_size

        gcn_output = self.enc(batch_map, data)

        shape = (sum(node_lengths) + sum(edge_lengths), size)
        assert gcn_output.shape == shape, (shape, gcn_output.shape)

        if True:
            new_h = torch.zeros(batch_size, max_node_length, size, dtype=torch.float, device=device)
            labels = torch.zeros(batch_size, max_node_length, dtype=torch.long, device=device)
            labels_mask = torch.zeros(batch_size, max_node_length, dtype=torch.bool, device=device)
            label_node_ids = torch.full((batch_size, max_node_length), -1, dtype=torch.long, device=device)

            offset = 0
            for i_b in range(batch_size):
                n = node_lengths[i_b]
                n_e = edge_lengths[i_b]
                if batch_map['add_edges'] == False:
                    assert n_e == 0

                #
                new_h[i_b, :n] = gcn_output[offset:offset + n]
                labels[i_b, :n] = data.y[offset:offset + n]
                labels_mask[i_b, :n] = True
                label_node_ids[i_b, :n] = torch.arange(n, dtype=torch.long, device=device)

                #
                offset += n + n_e

        output = new_h
        output = self.dropout(output)

        return output, labels, labels_mask, label_node_ids


class GCN(torch.nn.Module):
    def __init__(self, embed, size, mode='gcn', num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.embed = embed
        self.size = size
        self.output_size = size

        input_size = embed.output_size

        self.W_node = nn.Linear(input_size, size)

        if mode == 'gcn':
            for i in range(num_layers):
                setattr(self, 'conv{}'.format(i + 1), gnn.GCNConv(size, size))
        elif mode == 'gcn_transformer':
            for i in range(num_layers):
                setattr(self, 'conv{}'.format(i + 1), gnn.TransformerConv(size, size))
        elif mode == 'gcn_film':
            for i in range(num_layers):
                setattr(self, 'conv{}'.format(i + 1), gnn.FiLMConv(size, size))
        elif mode == 'gcn_gated':
            self.conv1 = gnn.GatedGraphConv(size, num_layers=num_layers)
        self.mode = mode

        self.mask_vec = nn.Parameter(torch.FloatTensor(input_size).normal_())

    def compute_node_features(self, node_tokens, mask=None):
        if mask is None:
            return self.W_node(self.embed(node_tokens))
        else:
            m = torch.cat(mask, 0)
            e = self.embed(node_tokens)
            e[m == MaskInfo.masked] = self.mask_vec
            return self.W_node(e)

    def forward(self, batch_map, data):
        batch_size = len(batch_map['items'])

        data.x = self.compute_node_features(data.y, mask=batch_map['mask_for_gcn'])

        # Hacky way to support any graphs with no edges.
        if any([data[i_b].edge_index.shape[0] == 0 for i_b in range(batch_size)]):
            return data.x

        x, edge_index = data.x, data.edge_index

        if self.mode == 'gcn_gated':
            x = self.conv1(x, edge_index)

        else:
            for i in range(self.num_layers):
                x = getattr(self, 'conv{}'.format(i + 1))(x, edge_index)
                if i < self.num_layers - 1:
                    x = torch.relu(x)

        return x

