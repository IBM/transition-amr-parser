"""BART GPT2 BPE encoder for tokenization with added AMR special action symbols.
"""
import json
from pathlib import Path
from typing import List, Union, Tuple

import regex as re
from fairseq import file_utils
from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.data.dictionary import Dictionary
import torch


"""
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
"""

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
DEFAULT_DICT_TXT = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt"


class AMRActionBPEEncoder(Encoder):
    """BPE tokenization and encoding to numerical values. Adding customized AMR action symbols to the tokenizer.

    Usage:
    - For normal English source sentence, use self.encode() and self.decode(), which are
        the original bpe tokenization (the added speical symbols should affect);
    - For AMR action target sequence, use self.encode_actions() and self.decode_actions(), which will
        always treats the first token with a leading white space, and will use the added special symbols.
    """

    INIT = 'Ġ'

    def __init__(self, encoder, bpe_merges, errors="replace"):
        super().__init__(encoder, bpe_merges, errors)
        self.old_enc_size = len(self.encoder)
        self.no_additions = []
        self.additions = []
        self.num_additions = 0

        # # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # self.pat = re.compile(
        #     r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # )

    @classmethod
    def build_bpe_encoder(cls, encoder_json_path=None, vocab_bpe_path=None,
                          *, node_freq_min=5, node_file_path=None, others_file_path=None):
        # use the default path if not provided
        encoder_json_path = encoder_json_path or file_utils.cached_path(DEFAULT_ENCODER_JSON)
        vocab_bpe_path = vocab_bpe_path or file_utils.cached_path(DEFAULT_VOCAB_BPE)
        # build the gpt2 bpe encoder
        with open(encoder_json_path, "r") as f:
            encoder = json.load(f)
        with open(vocab_bpe_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        bpe_encoder = cls(encoder, bpe_merges, errors="replace")
        bpe_encoder.add_amr_action_vocabulary(node_freq_min=node_freq_min,
                                              node_file_path=node_file_path,
                                              others_file_path=others_file_path)
        return bpe_encoder

    def add_amr_action_vocabulary(self, node_freq_min=5, node_file_path=None, others_file_path=None):
        tokens = []
        # node names
        if node_file_path:
            for line in Path(node_file_path).read_text().strip().splitlines():
                tok, count = line.split('\t')
                if int(count) >= node_freq_min:
                    tokens.append(self.INIT + tok)    # start of a new token

        # all other actions
        if others_file_path:
            for line in Path(others_file_path).read_text().strip().splitlines():
                tok, count = line.split('\t')
                tokens.append(self.INIT + tok)    # start of a new token

        # predicate sense postfixes: '-00', '-01', '-02', ..., '-99'
        for sn in range(100):
            tokens.append(f'-{sn:02d}')    # inside a token
            # an alternative
            # tokens.append('-' + str(sn).zfill(2))

        # filter out symbols that are already in the BPE vocabulary
        self.no_additions = [t for t in tokens if t in self.encoder]
        tokens = [t for t in tokens if t not in self.encoder]

        # add new symbols to the vocabulary
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start=old_enc_size):
            self.encoder[t] = i

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.additions = tokens
        self.num_additions = len(tokens)

    def _tok_bpe(self, token, add_space=True):
        # default
        if add_space:
            token = ' ' + token.lstrip()
        else:
            token = token.lstrip()

        tokk = []
        for tok in re.findall(self.pat, token):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            toks = self.bpe(tok).split(' ')
            tokk.extend(toks)

        return tokk

    def tokenize_act(self, act: str) -> List[str]:
        # tokenize a single action word
        assert not act.startswith(self.INIT)
        is_in_enc = self.INIT + act in self.encoder
        # NOTE must add "$" to match the end, otherwise there could be actions
        #      like 'F-47C' being matched and created unk index in the dictionary
        is_frame = re.match(r'.+-\d\d$', act) is not None

        if is_in_enc:
            bpe_toks = [self.INIT + act]
        elif is_frame:
            bpe_toks = self._tok_bpe(act[:-3], add_space=True) + [act[-3:]]
        else:
            bpe_toks = self._tok_bpe(act, add_space=True)

        return bpe_toks

    def tokenize_actions(self, actions: Union[List[str], str], word_sep=' ') \
            -> Tuple[List[List[str]], List[int], List[int]]:
        # tokenize an action sequence
        if isinstance(actions, str):
            actions = actions.strip().split(word_sep)
        elif isinstance(actions, list):
            pass
        else:
            raise TypeError

        bpe_tokens = []    # list of list
        tok_to_subtok_start = []    # map from token index to subtoken start position index
        subtok_origin_index = []    # the index of the original token for each subtoken
        num_subtok = 0    # number of subtokens in total

        for i, act in enumerate(actions):
            bpe_toks = self.tokenize_act(act)

            bpe_tokens.append(bpe_toks)    # list of list

            tok_to_subtok_start.append(num_subtok)
            subtok_origin_index += [i] * len(bpe_toks)

            num_subtok += len(bpe_toks)

        assert len(bpe_tokens) == len(actions)
        assert len(tok_to_subtok_start) == len(actions)
        assert len(subtok_origin_index) == num_subtok

        return bpe_tokens, tok_to_subtok_start, subtok_origin_index

    def encode_actions(self, actions: Union[List[str], str], word_sep=' ') \
            -> Tuple[List[int], List[str], List[int], List[int]]:
        bpe_tokens, tok_to_subtok_start, subtok_origin_index = self.tokenize_actions(actions, word_sep)
        # flatten the list
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        # encode into subtoken ids
        bpe_token_ids = [self.encoder[b] for b in bpe_tokens]
        return bpe_token_ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index

    def decode_actions(self, tokens: List[int]) -> str:
        # NOTE the decoded string will always have an white space in the front before the first token
        text = self.decode(tokens)
        return text


class AMRActionBartDictionary(Dictionary):
    """Bart dictionary customized to include new AMR action symbols.
    Tokenization is based on the GPT2 BPE encoder.
    """

    def __init__(self,
                 dict_txt_path=None,
                 node_freq_min=5,
                 node_file_path=None,
                 others_file_path=None,
                 **kwargs):
        super().__init__(**kwargs)

        # build the base dictionary on gpt2 bpe token ids (used by BART)
        dict_txt_path = dict_txt_path or file_utils.cached_path(DEFAULT_DICT_TXT)
        self.add_from_file(dict_txt_path)

        # the size of the original BART vocabulary; this is needed to truncate the pretrained BART vocabulary, as
        # it comes with '<mask>' from denoising task and
        # bart.base has larger vocabulary with more padded <madeupwordxxxx>
        self.bart_vocab_size = len(self.symbols)

        # build the extended bpe tokenizer and encoder
        self.bpe = AMRActionBPEEncoder.build_bpe_encoder(node_freq_min=node_freq_min,
                                                         node_file_path=node_file_path,
                                                         others_file_path=others_file_path)

        # add the new tokens to the vocabulary (NOTE the added symbols are the index in the bpe vocabulary)
        for tok in self.bpe.additions:
            self.add_symbol(str(self.bpe.encoder[tok]))

    def __getitem__(self, idx):
        sym = super().__getitem__(idx)
        if sym.isdigit() and int(sym) in self.bpe.decoder:
            # map back to the original symbols
            return self.bpe.decoder[int(sym)]
        else:
            # special symbols in fairseq dictionary
            return sym

    def index(self, sym, map_to_bpe_id=True):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)

        # map from surface symbol to bpe index that are the symbols of the dictionary
        if map_to_bpe_id:
            if sym in self.bpe.encoder:
                sym = str(self.bpe.encoder[sym])

        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def tokenize_act(self, act: str) -> List[str]:
        return self.bpe.tokenize_act(act)

    def tokenize_actions(self, actions: Union[List[str], str], word_sep=' ') \
            -> Tuple[List[List[str]], List[int], List[int]]:
        return self.bpe.tokenize_actions(actions, word_sep)

    def encode_actions(self, actions: Union[List[str], str], word_sep=' ') \
            -> Tuple[torch.LongTensor, List[str], List[int], List[int]]:
        # this would split any bpe tokens in the process
        bpe_token_ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index = self.bpe.encode_actions(actions, word_sep)

        nwords = len(bpe_token_ids)
        ids = torch.LongTensor(nwords)
        for i, word in enumerate(bpe_token_ids):
            ids[i] = self.index(str(word), map_to_bpe_id=False)

        return ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index

    def decode_actions(self, tensor: torch.LongTensor) -> List[str]:
        # this would join any bpe tokens already
        bpe_tensor = [int(self.symbols[i]) for i in tensor]
        # NOTE index in self.symbols[i] directly, instead of using self[i], as we have overwritten self.__getitem__ so
        #      that self.symbols[i] and self[i] are not the same anymore.
        # .split() would remove the white space at the beginning as well
        return self.bpe.decode_actions(bpe_tensor).split()
