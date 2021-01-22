from fairseq_ext.roberta.pretrained_embeddings import PretrainedEmbeddings as PERoberta
from fairseq_ext.roberta.pretrained_embeddings_bert import PretrainedEmbeddings


infile = 'EXP/data/en_embeddings/roberta_large_top24/train.en-actions.en'
def tokenize(line): return line.strip().split('\t')


with open(infile, 'r') as f:
    lines = [line for line in f]
sentences = [" ".join(tokenize(str(sentence).rstrip())) for sentence in lines]

sentence = sentences[3]
print(sentence)

pe = PretrainedEmbeddings('bert-large-cased', list(range(1, 25)))
word_features, worpieces_bert, word2piece = pe.extract(sentence)

breakpoint()

pe_roberta = PERoberta('roberta.large', list(range(1, 25)))
word_features_r, wordpieces_r, word2piece_r = pe_roberta.extract(sentence)

breakpoint()
