import torch
from fairseq.data import Dictionary
from neural_parser.fairseq_ext.extract_bart.composite_embeddings import CompositeEmbeddingBART


if __name__ == '__main__':
    vocab_path = '/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart/EXP/data/graphmp-swaparc-ptrlast_o8.3_act-states/processed/dict.actions_nopos.txt'
    vocab = Dictionary.load(vocab_path)

    bart = torch.hub.load('pytorch/fairseq', 'bart.base')

    cemb = CompositeEmbeddingBART(bart, bart.model.decoder.embed_tokens, vocab)

    indices = torch.tensor([[1, 3, 8], [10, 5000, 666]])

    indices = indices.cuda()
    cemb.to('cuda')

    embeddings = cemb(indices, update=True)

    breakpoint()

    # test backprop
    optimizer = torch.optim.SGD(cemb.parameters(), lr=1)
    for i in range(2):
        print()
        optimizer.zero_grad()
        print(cemb.base_embeddings.weight[:1].sum())
        print(cemb.base_embeddings.weight[:2])
        ll = cemb(torch.tensor([0, 1, 2]).cuda(), update=True).sum() * 10
        ll.backward()
        print(cemb.base_embeddings.weight.grad)
        optimizer.step()
        print(cemb.base_embeddings.weight[:1].sum())
        print(cemb.base_embeddings.weight[:2])
        print()

    breakpoint()
