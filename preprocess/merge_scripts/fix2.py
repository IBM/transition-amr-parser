#!/usr/bin/python
import operator
import sys
import re
import string
import os

#tries to remove repeated alignments

fin = open(sys.argv[1])

key_words = []#"have-org-role-91","monetary-quantity","temporal-quantity","thing","rate-entity-91","organization"]

node_ht = {}
word_ht = {}
node_lns = []
aln_line = ""
tok_line = ""
for oline in fin:
    line = oline.strip()
    if "::tok" in line:
        tok_line = line

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

        if len(node_ln) == 5:
            rkey = node_ln[-1]
            if rkey not in node_ht:
                node_ht[rkey] = []
            node_ht[rkey].append(node_ln[2])
        word_ht[node_ln[2]] = node_ln[3]
        node_lns.append(node_ln)
        continue

    if "::alignments" in line:
        aln_line = line
        continue

    if "::root" in line:

        if False:
            for n in range(0,len(node_lns)):
                if len(node_lns[n]) == 4 and node_lns[n][-1] != 'and':
                    
                    emp_node = node_lns[n][2]
                    str_i = 10000
                    end_i = -1
                    for node_ln in node_lns:
                        if node_ln[2].startswith(emp_node) and len(node_ln) != 4 :
                            rang = node_ln[-1]
                            (i,j) = rang.split('-')
                            if int(i) < str_i:
                                str_i = int(i)
                            if int(j) > end_i:
                                end_i = int(j)
                    if end_i >= 0 and str_i < 10000:
                        rkey = str(str_i)+'-'+str(end_i)
                        node_lns[n].append(rkey)
                        if rkey not in node_ht:
                            node_ht[rkey] = []
                        node_ht[rkey].append(node_lns[n][2])
                        aln_line = aln_line + " " + str(str_i) + '-' + str(end_i) + "|" + emp_node


        for key in node_ht:
            if len(node_ht[key]) > 1:
                nodes = sorted(node_ht[key])
                bad = False
                good_len = len(nodes)
                for i in range(1,len(nodes)):
                    pre_this = ".".join(nodes[i].split(".")[0:-1]) 
                    pre_last = ".".join(nodes[i-1].split(".")[0:-1])

                    #if not (  ( nodes[i-1] == pre_this or pre_last == pre_this ) and
                    #          (word_ht[nodes[i]] != word_ht[nodes[i-1]] or 'date-entity' in word_ht[nodes[0]] ) and
                    #          ( "-0" not in word_ht[nodes[i-1]] or word_ht[nodes[i]] == "amr-unknown" ) ):

                    if not (  ( nodes[i-1] == pre_this or pre_last == pre_this ) and word_ht[nodes[i]] != word_ht[nodes[i-1]] and ( "-0" not in word_ht[nodes[i-1]] or word_ht[nodes[i]] == "amr-unknown" ) ):
                    #if not (  ( nodes[i].startswith(nodes[i-1]) ) or (nodes[i-1].split(".")[0:-1] == nodes[i].split(".")[0:-1] and word_ht[nodes[i]] != word_ht[nodes[i-1]] ) or (word_ht[nodes[i]] in key_words) or (word_ht[nodes[i-1]] in key_words) ):
                        if word_ht[nodes[i-1]] == '-+':
                            nodes[i-1] = nodes[i]
                        bad = True
                        good_len = i
                        break
                if bad:
                    node_ht[key] = nodes[0:good_len]
                if False:
                    print tok_line                                         
                    words = []
                    for node in node_ht[key]:
                        words.append(word_ht[node])
                    print key + "\t" + "|".join(node_ht[key]) + "\t" + " ".join(words) + "\t---\t" + tok_line
                    #print key + "\t" + "|".join(node_ht[key]) + "\t" + " ".join(words)

        print aln_line
        for n in range(0,len(node_lns)):
            if len(node_lns[n]) == 5 and node_lns[n][2] not in node_ht[node_lns[n][-1]]:
                node_lns[n][-1] = ""
            print node_lns[n][0] + " " + "\t".join(node_lns[n][1:])
     
        node_ht = {}
        word_ht = {}
        node_lns = []
        aln_line = ""

    print oline.rstrip()
