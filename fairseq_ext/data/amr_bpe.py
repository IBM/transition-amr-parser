"""BART GPT2 BPE encoder for tokenization with added AMR special action symbols.
"""
import json
from pathlib import Path

import regex as re
from fairseq.data.encoders.gpt2_bpe_utils import Encoder


"""
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
"""


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
    def build_bpe_encoder(cls, encoder_json_path, vocab_bpe_path,
                          *, node_freq_min=5, node_file_path=None, others_file_path=None):
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
        # default to adding space before every token, regardless of the sentence beginning (NOTE)
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

    def tokenize_actions(self, actions, word_sep=' '):
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
            is_in_enc = self.INIT + act in self.encoder
            is_frame = re.match(r'.+-\d\d', act) is not None

            if is_in_enc:
                bpe_toks = [self.INIT + act]
            elif is_frame:
                bpe_toks = self._tok_bpe(act[:-3], add_space=True) + [act[-3:]]
            else:
                bpe_toks = self._tok_bpe(act, add_space=True)

            bpe_tokens.append(bpe_toks)    # list of list

            tok_to_subtok_start.append(num_subtok)
            subtok_origin_index += [i] * len(bpe_toks)

            num_subtok += len(bpe_toks)

        assert len(bpe_tokens) == len(actions)
        assert len(tok_to_subtok_start) == len(actions)
        assert len(subtok_origin_index) == num_subtok

        return bpe_tokens, tok_to_subtok_start, subtok_origin_index

    def encode_actions(self, actions, word_spe=' '):
        bpe_tokens, tok_to_subtok_start, subtok_origin_index = self.tokenize_actions(actions, word_spe)
        # flatten the list
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        # encode into subtoken ids
        bpe_token_ids = [self.encoder[b] for b in bpe_tokens]
        return bpe_token_ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index

    def decode_actions(self, tokens):
        # NOTE the decoded string will always have an white space in the front before the first token
        text = self.decode(tokens)
        return text
