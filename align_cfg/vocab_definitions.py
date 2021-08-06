class MaskInfo:
    unchanged = 0
    masked = 1
    unchanged_and_predict = 2

PADDING_IDX = 0
PADDING_TOK = '<PAD>'

BOS_IDX = 1
BOS_TOK = '<S>'

EOS_IDX = 2
EOS_TOK = '</S>'

special_tokens = [PADDING_TOK, BOS_TOK, EOS_TOK]

assert special_tokens.index(PADDING_TOK) == PADDING_IDX
assert special_tokens.index(BOS_TOK) == BOS_IDX
assert special_tokens.index(EOS_TOK) == EOS_IDX
