#!/usr/bin/python
import operator
import sys
import re
import string
import os

key_words = ["location","country","city","continent","person","name","government-organization","have-org-role-91","monetary-quantity","temporal-quantity","rate-entity-91","url-entity","date-entity","score-entity","ordinal-entity","percentage-entity","phone-number-entity","date-interval","criminal-organization","thing","publication","multiple"]#,"person","have-rel-role-91"]

fin = open(sys.argv[1])

node_lns = []
edge_lns = []

node_kids = {}
node_name = {}
edge_lbls = {}
node_align = {}

aln_line = ""
root_line = ""
for oline in fin:
    line = oline.strip()
    if "::node" in line:
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
        node_name[node_ln[2]] = node_ln[3]
        if len(node_ln) > 4:
            node_align[node_ln[2]] = node_ln[-1]
        continue

    if "::alignments" in line:
        aln_line = line
        continue

    if "::root" in line:
        root_line = line
        continue

    if "::edge" in line:
        edge_ln = line.split()
        if edge_ln[-2] not in node_kids:
            node_kids[edge_ln[-2]] = []
        node_kids[edge_ln[-2]].append(edge_ln[-1])
        edge_lbls[edge_ln[-2]+"<=>"+edge_ln[-1]] = edge_ln[3]
        edge_lns.append(line)
        continue
    
    if not line.startswith("#") and len(node_lns)>0:

        #now do something here!!!
        for n in range(len(node_lns)-1,-1,-1):
            if len(node_lns[n]) == 4 and "multi-sentence" not in node_lns[n]:
                emp_node_id = node_lns[n][2]
                emp_node_nm = node_name[emp_node_id]
                best_kid_id = ""

                if emp_node_id not in node_kids or "-0" in emp_node_nm or emp_node_nm == "and": 
                    continue

                #if len(node_kids[emp_node_id]) == 1:
                #    kid_id = node_kids[emp_node_id][0]
                #    if kid_id in node_align:
                #        best_kid_id = kid_id
                
                for i in range(0,len(node_kids[emp_node_id])):
                    kid_id = node_kids[emp_node_id][i]
                    kid_lbl = edge_lbls[emp_node_id+"<=>"+kid_id]
                    if kid_id not in node_align:
                        continue
                    if kid_lbl == "name":
                        best_kid_id = kid_id
                        break
                    if "-quantity" in emp_node_nm and kid_lbl == "unit":
                        best_kid_id = kid_id
                        break
                    if "have-org-role-91" == emp_node_nm and kid_lbl == "ARG2":
                        best_kid_id = kid_id
                        break
                    if "rate-entity" == emp_node_nm and kid_lbl == "ARG2":
                        best_kid_id = kid_id
                        break

                if best_kid_id == "" and emp_node_nm not in key_words:
                    continue

                if best_kid_id == "":
                    for i in range(0,len(node_kids[emp_node_id])):
                        kid_id = node_kids[emp_node_id][i]
                        kid_lbl = edge_lbls[emp_node_id+"<=>"+kid_id]
                        kid_nm = node_name[kid_id]
                        if kid_id in node_align and kid_lbl != "mod" and kid_nm in key_words :
                            best_kid_id = kid_id
                            break

                if best_kid_id != "":
                        align = node_align[best_kid_id]
                        node_lns[n].append(align)
                        node_align[emp_node_id] = align
                        all_lbls = []
                        for key in edge_lbls:
                            if key.startswith(emp_node_id+"<=>"):
                                all_lbls.append(edge_lbls[key])
                        #print node_name[emp_node_id] + "\t:" + edge_lbls[emp_node_id+"<=>"+best_kid_id] + "\t" + node_name[best_kid_id] + "\t|\t" + " ".join(all_lbls)
                

        print aln_line
        for n in range(0,len(node_lns)):
            print node_lns[n][0] + " " + "\t".join(node_lns[n][1:])
        print root_line
        for n in range(0,len(edge_lns)):
            print edge_lns[n] #[0] + " " + "\t".join(edge_lns[n][1:])            

        node_lns = []
        edge_lns = []

        node_name = {}
        node_kids = {}
        node_align = {}
        edge_lbls = {}

        aln_line = ""

    print oline.rstrip()
