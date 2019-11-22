from collections import Counter
from typing import List
import warnings
import torch
import spacy
from spacy.tokens import Doc
import copy
from fairseq.models.roberta.alignment_utils import spacy_nlp
from fairseq import utils
from fairseq.data.data_utils import collate_tokens

def align_bpe_to_words(roberta, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`

    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    assert bpe_tokens.dim() == 1

    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    other_tokens = [clean(str(o)) for o in other_tokens]
    # strip leading <s>
    assert bpe_tokens[0] == '<s>'
    bpe_tokens = bpe_tokens[1:]
    
    # create alignment from every word to a list of BPE tokens
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(str(bpe_tok)):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif str(bpe_tok).startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = str(bpe_tok)[len(other_tok):]
                other_tok = ''
            else:
                #raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                warnings.warn('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                bpe_indices.append(j)
                bpe_tok = str(bpe_tok)[len(other_tok):]
                other_tok = ''
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)

    return alignment

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

def extract_features_aligned_to_words_batched(model, sentences : list, use_all_layers: bool = True , return_all_hiddens: bool = False) -> torch.Tensor:
    nlp = spacy_nlp()
    bpe_toks = []
    alignments = []
    spacy_tokens = []
    for sentence in sentences:
        bpe_tok = model.encode(sentence)
        spacy_toks_ws = sentence.split()
        alignment = align_bpe_to_words(model, bpe_tok, spacy_toks_ws)
        bpe_toks.append(bpe_tok)
        alignments.append(alignment)
        spacy_tokens.append(spacy_toks_ws)
    
    # for i,tok in enumerate(bpe_toks):
    #     print(f"tokens {i}:", tok.shape)
    bpe_toks_collated = collate_tokens(bpe_toks, pad_idx=1)

    features = model.extract_features(bpe_toks_collated, return_all_hiddens=return_all_hiddens)
    final_features = sum(features[1:])/(len(features)-1)

    results = []
    for bpe_tok, final_feature, alignment, spacy_toks_ws in zip(bpe_toks, final_features, alignments, spacy_tokens):
        # print("Num tokens: ", bpe_tok.shape[0])
        # print("Num words: ", len(spacy_toks_ws))
        # print("Num features: ", final_feature[0:bpe_tok.shape[0]].shape)
        aligned_feats = align_features_to_words(model, final_feature[0:bpe_tok.shape[0]], alignment)
        # print("Aligned shape: ", aligned_feats.shape)
        doc = Doc(
            nlp.vocab,
            #words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'],
            words=['<s>'] + [x for x in spacy_toks_ws] + ['</s>'],
            #spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False],
        )
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        results.append(copy.copy(doc))

    return results


      