import torch
import torch.nn as nn

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv
except:
    pass

from vocab_definitions import MaskInfo


class GCNEncoder(nn.Module):
    def __init__(self, embed, size, mode='gcn', dropout_p=0):
        super().__init__()

        self.enc = GCN(embed, size)

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

        lengths = [x['geometric_data'].y.shape[0] for x in batch_map['items']]
        length = max_length = max(lengths)
        size = self.enc.output_size

        gcn_output = self.enc(batch_map, data)

        shape = (sum(lengths), size)
        assert gcn_output.shape == shape, (shape, gcn_output.shape)

        new_h = torch.zeros(batch_size, length, size, dtype=torch.float, device=device)
        labels = torch.zeros(batch_size, length, dtype=torch.long, device=device)
        labels_mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        label_node_ids = torch.full((batch_size, length), -1, dtype=torch.long, device=device)

        offset = 0
        for i_b in range(batch_size):
            n = lengths[i_b]

            #
            new_h[i_b, :n] = gcn_output[offset:offset + n]
            labels[i_b, :n] = data.y[offset:offset + n]
            labels_mask[i_b, :n] = True
            label_node_ids[i_b, :n] = torch.arange(n, dtype=torch.long, device=device)

            #
            offset += n

        output = new_h

        return output, labels, labels_mask, label_node_ids


class GCN(torch.nn.Module):
    def __init__(self, embed, size):
        super().__init__()

        self.embed = embed
        self.size = size
        self.output_size = size

        input_size = embed.output_size

        self.W_node = nn.Linear(input_size, size)

        self.conv1 = GCNConv(size, size)
        self.conv2 = GCNConv(size, size)

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
        data.x = self.compute_node_features(data.y, mask=batch_map.get('mask', None))

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return x
