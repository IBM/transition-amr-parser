from amr import JAMR_CorpusReader

amr_file = '../data/train.txt'
new_amr_file = '../data/train.no_wiki.txt'


cr = JAMR_CorpusReader()
cr.load_amrs(amr_file, verbose=False)
amrs = cr.amrs
sent_idx=0
for amr in amrs:
    wiki_edges = []
    wiki_nodes = []
    for s,r,t in amr.edges:
        if r==':wiki':
            wiki_edges.append((s,r,t))
            wiki_nodes.append(t)
    for e in wiki_edges:
        amr.edges.remove(e)
    for n in wiki_nodes:
        del amr.nodes[n]
        if n in amr.alignments:
            del amr.alignments[n]
        print('deleting wiki:',sent_idx)
    sent_idx+=1

with open(new_amr_file, 'w+',encoding='utf8') as f:
    for amr in amrs:
        f.write(amr.toJAMRString())




