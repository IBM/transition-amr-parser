"""Given a base vocabulary and corresponding embeddings, for another vocabulary where elements are composed of
sub-elements of the base vocabulary, construct the new embeddings by pooling over the base vocabulary embeddings.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class CompositeEmbedding(nn.Module):
    def __init__(self, base_dictionary, base_embeddings):
        super().__init__()

        self.base_dictionary = base_dictionary    # fairseq.data.dictionary.Dictionary
        self.base_embeddings = base_embeddings    # torch.Tensor/torch.nn.Parameter or torch.nn.Embedding
        if isinstance(self.base_embeddings, torch.nn.Embedding):
            assert len(self.base_dictionary) == self.base_embeddings.weight.size(0)
        elif isinstance(self.base_embeddings, torch.Tensor):
            assert len(self.base_dictionary) == self.base_embeddings.size(0)
        else:
            raise TypeError

    def map_symbol(self, sym, transform=None):
        """Return the mapping in the base vocabulary of a given symbol.

        Args:
            sym (str): a given symbol, possibly composite of multiple symbols in the base vocabulary.
            transform (None or callable): a function that transforms the symbol to a different form for mapping

        Returns:
            torch.LongTensor or List: a list of indices of sub-symbols in the base vocabulary.
        """
        raise NotImplementedError

    def map_dictionary(self, symbols, transform=None):
        """Map another dictionary to the base vocabulary. Return a tensor for scatter operation.

        Args:
            symbols (List[str]): list of symbols in a dictionary. e.g. dictionary.symbols, where dictionary
                is a fairseq dictionary (fairseq.data.dictionary.Dictionary)
            transform (None or callable): a function that transforms the symbol to a different form for mapping

        Returns:
            map_all (List[List]): map indices for each symbol in dictionary.
            extract_index (torch.LongTensor): index for extract and rearrange the embedding vectors in the base vocab
            scatter_index (torch.LongTensor): index for scatter operation based on the base vocabulary embeddings
        """
        map_all = []
        extract_index = []
        scatter_index = []
        for sym_id, sym in enumerate(symbols):
            base_idx = self.map_symbol(sym, transform=transform)
            map_all.append(base_idx)
            extract_index += base_idx
            scatter_index += [sym_id] * len(base_idx)

        extract_index = torch.tensor(extract_index)
        scatter_index = torch.tensor(scatter_index)

        return map_all, extract_index, scatter_index

    def scatter_embeddings(self, extract_index, scatter_index):
        # NOTE for scatter operation, the src and index have to be of the same size
        if isinstance(self.base_embeddings, torch.nn.Embedding):
            scatter_src = torch.index_select(self.base_embeddings.weight, 0, extract_index)
        elif isinstance(self.base_embeddings, torch.Tensor):
            scatter_src = torch.index_select(self.base_embeddings, 0, extract_index)
        else:
            raise TypeError
        out = scatter_mean(scatter_src, scatter_index, dim=0)
        return out


class CompositeEmbeddingBART(CompositeEmbedding):
    def __init__(self, bart, bart_embeddings, dictionary):
        super().__init__(bart.task.target_dictionary, bart_embeddings)

        # self.register_buffer('model', bart, persistent=False)  # persistent flag is only available for PyTorch >= 1.6
        # self.register_buffer('model', bart)    # does not work: could only be Tensor or None
        self.model = [bart]    # NOTE workaround to make it a List, so that it does not appear in state_dict
        # fairseq.models.bart.hub_interface.BARTHubInterface
        # `bart_embeddings` is typically `bart.model.decoder.embed_tokens` (torch.nn.Embedding)

        self.dictionary = dictionary
        map_all, extract_index, scatter_index = self.map_dictionary(dictionary.symbols,
                                                                    transform=transform_action_symbol)
        self.map_all = map_all
        # need to register as buffer to move to be able to move to device
        self.register_buffer('extract_index', extract_index)
        self.register_buffer('scatter_index', scatter_index)

        # mask on the dictionary symbols to use as compositional embeddings
        dict_pred_mask = self.get_dictionary_mask()
        self.register_buffer('dict_pred_mask', dict_pred_mask)

        self.num_embeddings = len(self.dictionary)
        self.embedding_dim = bart_embeddings.embedding_dim
        # NOTE bart_embeddings is torch.nn.Embedding

        self.update_embeddings()    # initialize the embeddings based on base embeddings

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

    def forward(self, token_index, update=True):
        # for evaluation with no updating base embeddings, set `update=False`
        if update:
            self.update_embeddings()

        # use embedding function instead of Embedding layer to not make intermediate embeddings as model parameters
        # which would cause problem with autograd (base_embeddings -> embeddings will not be recorded as the embeddings
        # parameter will be leaf node)
        emb = nn.functional.embedding(token_index, self.embedding_weight, padding_idx=self.dictionary.pad())
        return emb

    def update_embeddings(self):
        self.embedding_weight = self.scatter_embeddings(self.extract_index, self.scatter_index)

    def get_dictionary_mask(self):
        # mask on the dictionary elements that we'd like to keep using the BART compositional embeddings
        # currently try for the PRED node actions
        dict_pred_mask = torch.zeros(len(self.dictionary), dtype=torch.bool)
        for sym_id, sym in enumerate(self.dictionary.symbols):
            if sym.startswith('PRED'):
                dict_pred_mask[sym_id] = 1
        return dict_pred_mask


def transform_action_symbol(sym):
    """Transform the action symbol to a form to be more easily tokenized by BART vocabulary.
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

    # NOTE we need to add the space token boundary to map to the correct isolated token!
    new_sym = ' ' + new_sym

    return new_sym
