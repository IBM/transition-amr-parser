#!/usr/bin/python
import operator
import sys
import re
import string
import os

#for these nodes, convert word alignments to span alignments for multiword nodes
key_words = ["name","score-entity","date-entity"]

fin = open(sys.argv[1])

node_lns = []
aln_line = ""
tok_line = ""
for oline in fin:
    line = oline.strip()
    if "::node" in line and "multi-sentence" not in line:
        node_ln = line.split()
        if len(node_ln) > 5 or (len(node_ln) == 5 and '-' not in node_ln[-1]):
            tmp = node_ln
            node_ln = []
            node_ln.append(tmp[0])
            node_ln.append(tmp[1])
            node_ln.append(tmp[2])
            node_ln.append(" ".join(tmp[3:3+len(tmp)-5+1]))
            if '-' in tmp[-1]:
                node_ln.append(tmp[-1])
            else:
                node_ln[-1] = node_ln[-1] + ' ' + tmp[-1]
        node_lns.append(node_ln)
        continue

    if "::alignments" in line:
        aln_line = line
        continue

    if "::tok" in line:
        tok_line = line

    if "::root" in line:
        if True:
            for n in range(0,len(node_lns)):
                if len(node_lns[n]) == 4 :
                    node_name = node_lns[n][-1]
                    if node_name not in key_words:
                        continue
                    emp_node = node_lns[n][2]
                    str_i = 10000
                    end_i = -1
                    node_set = []
                    for node_ln in node_lns:
                        node_num_pre = ".".join(node_ln[2].split(".")[0:-1])
                        if node_num_pre == emp_node and len(node_ln) != 4 :
                            node_set.append(node_ln[3])
                            rang = node_ln[-1]
                            (i,j) = rang.split('-')
                            if int(i) < str_i:
                                str_i = int(i)
                            if int(j) > end_i:
                                end_i = int(j)
                    if end_i >= 0 and str_i < 10000:
                        if end_i - str_i <= float(len(node_set))*2:
                            node_lns[n].append(str(str_i)+'-'+str(end_i))
                            aln_line = aln_line + " " + str(str_i) + '-' + str(end_i) + "|" + emp_node
                        else:
                            node_lns[n].append(str(str_i)+'-'+str(str_i+1))
                            #print tok_line
                            #print node_name + "\t" + str(end_i-str_i) + "\t" + " ".join(node_set)

            print aln_line
            for n in range(0,len(node_lns)):
                print node_lns[n][0] + " " + "\t".join(node_lns[n][1:])
            
        node_lns = []
        aln_line = ""

    print oline.rstrip()
