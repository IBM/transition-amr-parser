import os

import torch
from fairseq.data import Dictionary
from fairseq_ext.extract_bart.composite_embeddings import CompositeEmbeddingBART, transform_action_symbol


if __name__ == '__main__':
    vocab_path = '/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart/EXP/data/graphmp-swaparc-ptrlast_o8.3_act-states/processed/dict.actions_nopos.txt'
    vocab = Dictionary.load(vocab_path)

    bart = torch.hub.load('pytorch/fairseq', 'bart.base')

    cemb = CompositeEmbeddingBART(bart, bart.model.decoder.embed_tokens, vocab)

    trans_actions = []
    for sym in vocab.symbols:
        new_sym = transform_action_symbol(sym)    # str
        splitted = cemb.sub_tokens(new_sym)    # list
        # trans_actions.append((new_sym, splitted))
        trans_actions.append(new_sym + '  -->  ' + '|'.join(splitted) + '\n')

    tmp_dir = 'fairseq_ext/tests_data'
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, 'dict.actions_nopos.bartmap.txt'), 'w') as f:
        f.writelines(trans_actions)

    breakpoint()
