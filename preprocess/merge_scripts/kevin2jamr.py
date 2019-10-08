#!/usr/bin/python
import operator
import sys
import re
import string
import os


def kevin2jamr ( aln ) :
    (tok_idx,node) = aln.split("-")
    tidx = int(tok_idx)
    tokrange = str(tidx)+"-"+str(tidx+1)
    
    idxs = node.split(".")
    new_idxs = []
    for idx in idxs:
        new_idxs.append(str(int(idx)-1))
    node = ".".join(new_idxs)
    return (tokrange,node)

#print kevin2jamr(sys.argv[1])

fin_kevin = open(sys.argv[1])
k_alns = []
k_node_hts = []
k_span_hts = []
for line in fin_kevin:
    alns = line.strip().split()
    kaln = "# ::alignments"
    node_ht = {}
    span_ht = {}
    for aln in alns:
        if 'r' not in aln:
            (rng,nod) = kevin2jamr(aln)
            kaln = kaln + " " + rng + "|" + nod
            node_ht[nod] = rng
            span_ht[rng] = 1
    k_alns.append(kaln)
    k_node_hts.append(node_ht)
    k_span_hts.append(span_ht)

fin_devaln = open(sys.argv[2])

aln_ht = {}
nodes = []
i = 0
for oline in fin_devaln:
    line = oline.strip()
    if "::alignments" in line:
        line = k_alns[i]
        i = i + 1
        aln_ht = {}

    if "::node" in line or "::root" in line:
        #print line
        toks = line.split()
        last = toks[-1]
        digits = last.split("-")
        if len(toks) == 4 or len(digits) != 2 or not digits[0].isdigit() or not digits[1].isdigit():
            toks.append("")

        toks[-1] = ""

        if "multi-sentence" not in line:
            if toks[2] in k_node_hts[i-1]:
                toks[-1] = k_node_hts[i-1][toks[2]]
                key = toks[3]+" +++ "+toks[-1]
                aln_ht[key] = 1

        nodes.append(toks)
        

        if "::root" in line:
            for toks in nodes:
                if toks[2] not in k_node_hts[i-1]:
                    key = toks[3]+" +++ "+toks[-1]
                    if key in aln_ht:
                        toks[-1] = ""
                print "# " + "\t".join(toks[1:3]) + "\t" + " ".join(toks[3:-1]) + "\t" + toks[-1]
            nodes = []
            aln_ht = {}

        continue
    
    print oline.rstrip()
    
