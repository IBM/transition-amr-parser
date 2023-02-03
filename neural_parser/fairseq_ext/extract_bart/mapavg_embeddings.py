"""From a list of symbols, get the average embeddings from some base embeddings and a base vocabulary.
"""

import torch

from .composite_embeddings import CompositeEmbedding


class MapAvgEmbeddingBART(CompositeEmbedding):
    def __init__(self, bart, bart_embeddings):
        super().__init__(bart.task.target_dictionary, bart_embeddings)

        # self.register_buffer('model', bart, persistent=False)  # persistent flag is only available for PyTorch >= 1.6
        # self.register_buffer('model', bart)    # does not work: could only be Tensor or None
        self.model = [bart]    # NOTE workaround to make it a List, so that it does not appear in state_dict
        # fairseq.models.bart.hub_interface.BARTHubInterface
        # `bart_embeddings` is typically `bart.model.decoder.embed_tokens` (torch.nn.Embedding)

        self.embedding_dim = bart_embeddings.embedding_dim
        # NOTE bart_embeddings is torch.nn.Embedding

    def map_symbol(self, sym, transform=None):
        if sym in ['<s>', '<pad>', '</s>', '<unk>', '<mask>'] or sym.startswith('madeupword'):
            # keep the special symbols
            base_indices = [self.base_dictionary.index(sym)]
        else:
            if transform is not None:
                sym = transform(sym)
            # remove BOS and EOS symbols
            base_indices = self.model[0].encode(sym)[1:-1].tolist()
        return base_indices

    def sub_tokens(self, sym, transform=None):
        if sym in ['<s>', '<pad>', '</s>', '<unk>', '<mask>'] or sym.startswith('madeupword'):
            # keep the special symbols
            splitted = [sym]
        else:
            if transform is not None:
                sym = transform(sym)
            # split the tokens based on the bart bpe
            splitted = list(map(self.model[0].bpe.decode, self.model[0].bpe.encode(sym).split()))
        # return type is List
        return splitted

    def map_avg_embeddings(self, symbols, transform=None, add_noise=False):
        map_all, extract_index, scatter_index = self.map_dictionary(symbols, transform)
        with torch.no_grad():
            embedding_weight = self.scatter_embeddings(extract_index, scatter_index)
        if add_noise:
            noise = torch.empty_like(embedding_weight)
            noise.uniform_(-0.1, +0.1)
            embedding_weight += noise
        return embedding_weight, map_all


def transform_action_symbol(sym):
    """Transform the action symbol to a form to be more easily tokenized by BART vocabulary.
    The action symbols are already in the joint bpe vocabulary with BART, with 'Ġ' special marks.

    We adopt the following rules:
    - lower-case everything, so that
        SHIFT -> shift, REDUCE -> reduce, MERGE -> merge
    - verbalize some actions, so that the meanings are better represented (for BART, LA could more related to LA city),
        COPY_LEMMA -> copy, COPY_SENSE01 -> copy sense-01, LA(:ARG0) -> left arc :arg0, LA(root) -> left arc root
    - for PRED(...), remove PRED and (), so that
        PRED(thing) -> thing, PRED(go-02) -> go-02
    - for ENTITY(...), just lowercase?
        ENTITY(person,country,name) -> entity(person,country,name) -> entity person country name

    NOTE
        - need to prepend ' ' for token boundary for the BPE vocab
        - we remove all the dashes '-'


    Args:
        sym (str): an action

    Returns:
        new_sym (str): a string of the transformed action symbol
    """
    if sym in ['<s>', '<pad>', '</s>', '<unk>', '<mask>'] or sym.startswith('madeupword'):
        # special tokens
        return sym

    INIT = 'Ġ'

    sym = sym.lstrip(INIT)

    new_sym = sym.lower()

    # new symbol could be (after lowercasing) copy, shift, root, close, >la, >ra, and other nodes
    if new_sym in ['copy', 'shift', 'root', 'close']:
        ...

    elif new_sym.startswith(('>la', '>ra')):

        if new_sym.startswith('>la'):
            marker = 'left arc '
        else:
            marker = 'right arc '

        label = new_sym.split('(')[1][:-1]
        if label.startswith(':arg'):
            if label.endswith('-of'):
                label = 'argument ' + str(int(label[4:-3])) + ' of'
            else:
                label = 'argument ' + str(int(label[4:]))
        elif label.startswith(':op'):
            if label.endswith('-of'):
                label = 'operator ' + str(int(label[3:-3])) + ' of'
            else:
                label = 'operator ' + str(int(label[3:]))
        elif label.startswith(':snt'):
            label = 'sentence ' + str(int(label[4:]))
        else:
            label = ' '.join(label.lstrip(':').split('-'))

        new_sym = marker + label

    else:
        # assume the remaining are all node names
        # do not raise error: could be some special symbols in the dictionary such as <s> <pad> etc.
        if new_sym != '-':
            new_sym = ' '.join(new_sym.split('-'))    # remove the dash "-" and add token boundaries

            # remove any white space before, for the cases such as '-03', where '-03'.split() -> [' ', '03']
            new_sym = new_sym.lstrip()

    # NOTE we need to add the space token boundary to map to the correct isolated token!
    new_sym = ' ' + new_sym

    return new_sym
