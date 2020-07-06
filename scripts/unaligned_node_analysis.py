import sys
from collections import Counter
from amr import JAMR_CorpusReader


def main():
    cr = JAMR_CorpusReader()
    cr.load_amrs(sys.argv[1], verbose=False)

    special_alignments = Counter()

    for amr in cr.amrs:
        for node_id in amr.nodes:
            if node_id not in amr.alignments or not amr.alignments[node_id]:
                special_alignments[amr.nodes[node_id]] += 1

    for special in sorted(special_alignments, reverse=True, key=lambda x: special_alignments[x]):
        print(special.strip(), special_alignments[special])


if __name__ == '__main__':
    main()
