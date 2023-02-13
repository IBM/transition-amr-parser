from collections import Counter
import torch
from spacy.tokens import Doc
import copy
from fairseq.models.roberta.alignment_utils import spacy_nlp
from fairseq.data.data_utils import collate_tokens


def get_tokens(roberta, word):
    return roberta.task.source_dictionary.encode_line(roberta.bpe.encode(word), append_eos=False, add_if_not_exist=False)


def get_alignments_and_tokens(roberta, words):
    bpe_tokens = []
    alignment_position = 1
    alignments = []
    first_word_tokens = get_tokens(roberta, words[0])
    bpe_tokens.extend(first_word_tokens)
    alignments.append([(alignment_position + i) for i in range(0, len(first_word_tokens))])
    alignment_position = alignment_position + len(first_word_tokens)

    for word in words[1:]:
        tokens = get_tokens(roberta, " " + word)
        bpe_tokens.extend(tokens)
        alignments.append([(alignment_position + i) for i in range(0, len(tokens))])
        alignment_position = alignment_position + len(tokens)

    final_bpe_tokens = [roberta.task.source_dictionary.index('<s>')] + bpe_tokens + [roberta.task.source_dictionary.index('</s>')]
    return alignments, torch.LongTensor(final_bpe_tokens)


def align_features_to_words(roberta, features, alignment):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)
    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    return output


def extract_features_aligned_to_words_batched(model, sentences: list, use_all_layers: bool = True, return_all_hiddens: bool = False) -> torch.Tensor:
    nlp = spacy_nlp()
    bpe_toks = []
    alignments = []
    spacy_tokens = []
    for sentence in sentences:
        toks = sentence.split()
        alignment, bpe_tok = get_alignments_and_tokens(model, toks)
        bpe_toks.append(bpe_tok)
        alignments.append(alignment)
        spacy_tokens.append(toks)

    bpe_toks_collated = collate_tokens(bpe_toks, pad_idx=1)

    features = model.extract_features(bpe_toks_collated, return_all_hiddens=return_all_hiddens)
    final_features = sum(features[1:])/(len(features)-1)

    results = []
    for bpe_tok, final_feature, alignment, toks in zip(bpe_toks, final_features, alignments, spacy_tokens):
        aligned_feats = align_features_to_words(model, final_feature[0:bpe_tok.shape[0]], alignment)
        doc = Doc(
            nlp.vocab,
            words=['<s>'] + [x for x in toks] + ['</s>'],
        )
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        results.append(copy.copy(doc))

    return results


def extract_features_aligned_to_words(model, tokens: list, use_all_layers: bool = True, return_all_hiddens: bool = False) -> torch.Tensor:
    nlp = spacy_nlp()
    alignment, bpe_tok = get_alignments_and_tokens(model, tokens)
    features = model.extract_features(bpe_tok, return_all_hiddens=return_all_hiddens)
    final_features = sum(features[1:])/(len(features)-1)
    final_features = final_features.squeeze(0)
    aligned_feats = align_features_to_words(model, final_features, alignment)
    doc = Doc(
            nlp.vocab,
            words=['<s>'] + [x for x in tokens] + ['</s>']
            )
    doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
    return doc
