import time
import math

from fairseq.tokenizer import tokenize_line


def replace_unk(hypo_str, src_str, alignment, align_dict, unk, line_tokenizer=tokenize_line):
    # Tokens are strings here
    hypo_tokens = line_tokenizer(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = line_tokenizer(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict,
                            remove_bpe=None, split_token=' ', line_tokenizer=tokenize_line):
    # TODO check consistency of "split_token" and "tokenize_line"
    # TODO "line_tokenizer" not fed into "replact_unk"
    # hypo_str = tgt_dict.string(hypo_tokens, remove_bpe, split_token=split_token)
    hypo_str = split_token.join([tgt_dict[i] for i in hypo_tokens if i != tgt_dict.eos()])
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True, line_tokenizer=tokenize_line)
    return hypo_tokens, hypo_str, alignment


def join_action_pointer(action, pos):
    """Join action label and pointer value.

    Args:
        action (str): action label without pointer
        pos (int or str): pointer value

    Return:
        action_complete (str): complete action label
    """
    if action.startswith('LA') or action.startswith('RA'):
        action_parts = action.split('(')
        assert len(action_parts) == 2
        assert int(pos) >= 0
        action_complete = f'{action_parts[0]}({pos},{action_parts[1]}'
    else:
        action_complete = action
    return action_complete


def post_process_action_pointer_prediction(hypo, tgt_dict):
    """Post processing the prediction of both actions and corresponding pointer values."""
    # need to manually take care of eos, which is always included in the beam search output
    actions_nopos = [tgt_dict[i] for i in hypo['tokens'] if i != tgt_dict.eos()]    # or hypo['tokens'].tolist()
    actions_pos = hypo['pointer_tgt'].tolist()[:-1]
    # refine the pointer sequence to use -1 for all the non-arc actions
    actions_pos = [pos if act.startswith('LA') or act.startswith('RA') else -1
                   for act, pos in zip(actions_nopos, actions_pos)]
    assert len(actions_nopos) == len(actions_pos)
    actions = [join_action_pointer(act, pos) for act, pos in zip(actions_nopos, actions_pos)]
    return actions_nopos, actions_pos, actions


def time_since(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    if h == 0:
        if m == 0:
            return '%ds' % s
        else:
            return '%dm %ds' % (m, s)
    else:
        return '%dh %dm %ds' % (h, m, s)

