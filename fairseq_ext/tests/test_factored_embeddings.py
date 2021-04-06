import torch
from fairseq.data import Dictionary
from fairseq_ext.modules.factored_embeddings import FactoredEmbeddings


if __name__ == '__main__':
    vocab_path = '/n/tata_ddos_ceph/jzhou/transition-amr-parser-o8/EXP/data/graphmp-swaparc-ptrlast_o8.3_act-states/processed/dict.actions_nopos.txt'
    embed_dim = 256
    vocab = Dictionary.load(vocab_path)
    femb = FactoredEmbeddings(vocab, embed_dim)

    indices = torch.tensor([[1, 3, 8], [10, 5000, 666]]).cuda()
    femb.to('cuda')
    
    embeddings = femb(indices)

    breakpoint()
