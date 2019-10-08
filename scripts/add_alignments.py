from amr import JAMR_CorpusReader

amr_file = '../data/dev.jamr.no_wiki.txt'
alignments_file = '../data/dev.alignments.txt'

align_per_sent = []
with open(alignments_file, 'r') as f:
    align = dict()
    for line in f:
        if not line.strip():
            align_per_sent.append(align)
            align = dict()
            continue
        if line.startswith('[amr]'):
            continue

        sent_idx, node, tokens = (t for t in line.split())
        tokens = [int(t) for t in tokens.split(',')]
        sent_idx = int(sent_idx)
        assert(sent_idx==len(align_per_sent))

        align[node] = tokens

cr = JAMR_CorpusReader()
cr.load_amrs(amr_file)
amrs = cr.amrs
sent_idx=0
for amr, align in zip(amrs,align_per_sent):
    print(' '.join(amr.tokens))
    for n in amr.nodes:
        jamr_align = amr.alignments[n] if n in amr.alignments else []
        all_align = align[n] if n in align else []
        if not set(all_align).issuperset(jamr_align):
            print('JAMR:',sent_idx,n,amr.nodes[n],jamr_align)
            print('All:',sent_idx,n,amr.nodes[n],all_align)
    sent_idx+=1



