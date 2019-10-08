import sys
from collections import Counter
from amr import JAMR_CorpusReader


def get_token(gold_amr, t):
    if 0 <= t-1 < len(gold_amr.tokens):
        return gold_amr.tokens[t - 1]
    else:
        return 'NA'


if __name__ == '__main__':

    file = sys.argv[1]

    cr = JAMR_CorpusReader()
    cr.load_amrs(file)
    gold_amrs = cr.amrs

    count = 0
    sentences = set()
    rels = Counter()
    for sent_idx, gold_amr in enumerate(gold_amrs):
        for i, tok in enumerate(gold_amr.tokens):
            align = gold_amr.alignmentsToken2Node(i + 1)
            # merge alignments
            root = gold_amr.findSubGraph(align).root
            for n in gold_amr.nodes:
                if n in align:
                    continue
                edges = [(s, r, t) for s, r, t in gold_amr.edges if s in align and t in align]
                if not edges:
                    continue
                if n in gold_amr.alignments and gold_amr.alignments[n]:
                    nodes = gold_amr.alignmentsToken2Node(gold_amr.alignments[n][0])
                    align2 = gold_amr.alignments[n]
                else:
                    nodes = [n]
                    align2 = []
                for s, r, t in gold_amr.edges:
                    if n == s and t in align and t != root:
                        print()
                        print([gold_amr.nodes[m] for m in nodes], [gold_amr.nodes[m] for m in align])
                        print([get_token(gold_amr, t) for t in align2], [get_token(gold_amr, t) for t in gold_amr.alignments[root]])
                        print(gold_amr.nodes[s], r, gold_amr.nodes[t])
                        count += 1
                        sentences.add(sent_idx)
                        rels[' '.join([gold_amr.nodes[s], r, gold_amr.nodes[t]])] += 1
                    if n == t and s in align and s != root:
                        print()
                        print([gold_amr.nodes[m] for m in nodes], [gold_amr.nodes[m] for m in align])
                        print([get_token(gold_amr, t) for t in align2], [get_token(gold_amr, t) for t in gold_amr.alignments[root]])
                        print(gold_amr.nodes[s], r, gold_amr.nodes[t])
                        count += 1
                        sentences.add(sent_idx)
                        rels[' '.join([gold_amr.nodes[s], r, gold_amr.nodes[t]])] += 1
    print('occurences:', count)
    print('sentences:', len(sentences))
    print('most common edges:', rels)
