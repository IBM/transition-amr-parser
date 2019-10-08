#!/usr/bin/python
import operator
import sys
import re
import string
import os

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])

id = ""
node_lns1 = []
node_lns2 = []
aln_ht = {}
mrgok = True

for line1 in f1:
    line1 = line1.strip("\n\r")
    line2 = f2.readline().strip("\n\r")

    line = line1

    if "::tok" in line1:
        toks1 = line1.split()
        toks2 = line2.split()
        if line1 != line2 :
            mrgok = False
            print line2
            continue
        else:
            mrgok = True
        
    #continue

    if "::id" in line1:
        id = line.split()[2]

    if "::node" in line1:
        node_lns1.append(line1)
        node_lns2.append(line2)
        toks = line1.split("\t")
        #print len(toks)
        if len(toks) == 4 and "-" in toks[-1]:
            aln = toks[-1]
            if aln not in aln_ht:
                aln_ht[aln] = []
            aln_ht[aln].append(toks[2])
        continue

    if "::root" in line1:
        for i in range(0,len(node_lns1)):
            if not mrgok:
                print node_lns2[i]
                continue
            toks= node_lns1[i].split("\t")
            if not (len(toks) == 4 and "-" in toks[-1]):
                if len(toks) == 3:
                    toks.append("")
                toks2 = node_lns2[i].split("\t")
                if len(toks2) == 4 and "-" in toks2[-1]:
                    aln = toks2[-1]
                    if aln not in aln_ht or toks2[2] not in aln_ht[aln]:
                        toks[-1] = aln
                        node_lns1[i] = "\t".join(toks) #+ "\t#merged_from_jamr"
            print node_lns1[i]

        node_lns1 = []
        node_lns2 = []
        aln_ht = {}

    print line
