"""
Originally from:
https://github.com/pytorch/examples/blob/13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15/word_language_model/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src2=None, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        if src2 is None:
            src2 = self.pos_encoder(src * math.sqrt(self.ninp))
        output = self.transformer_encoder(src, self.src_mask)
        return output


class BiTransformer(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()

        self.fwd_enc = TransformerModel(ninp, nhead, nhid, nlayers, dropout)
        self.bwd_enc = TransformerModel(ninp, nhead, nhid, nlayers, dropout)

    def forward(self, src):
        assert len(src.shape) == 3

        src = self.fwd_enc.pos_encoder(src * math.sqrt(self.fwd_enc.ninp))

        # FORWARD
        fwd_out = self.fwd_enc(src, src)

        # BACKWARD
        bwd_src = torch.flip(src, [1])
        bwd_out = self.bwd_enc(bwd_src, bwd_src)
        bwd_out = torch.flip(bwd_out, [1])

        output = torch.cat([fwd_out, bwd_out], -1)

        return output


class TransformerLM(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.transformer_encoder = TransformerModel(ninp, nhead, nhid, nlayers, dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
