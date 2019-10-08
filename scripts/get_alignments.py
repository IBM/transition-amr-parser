import sys
from amr import JAMR_CorpusReader

if __name__ == '__main__':

    args = sys.argv
    infile = args[1]

    cr = JAMR_CorpusReader()
    cr.load_amrs(infile)
    gold_amrs = cr.amrs

    for sentidx, amr in enumerate(gold_amrs):
        for n in amr.alignments:
            print(str(sentidx)+'\t'+n+'\t'+','.join(str(s) for s in amr.alignments[n]))
        print()
