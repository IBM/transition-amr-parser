#!/usr/bin/python
import operator
import sys
import re
import string
import os

famr = open(sys.argv[1])
fwiki = open(sys.argv[2])
# FIXME: External dependency
if os.path.exists("/dccstor/multi-parse/amr/trn.wikis"):
    ftrn = open("/dccstor/multi-parse/amr/trn.wikis")
else:
    ftrn = open("amr_corpus/amr2.0/wiki/trn.wikis")

wiki_ht = {}
for line in ftrn:
    if len(line.strip().split('\t')) != 2:
        continue
    (n,w) = line.strip().split('\t')
    wiki_ht[n] = w

all_wikis = []
wikis = []
for line in fwiki:
    if line.strip() == "":
        all_wikis.append(wikis)
        wikis = []
    else:
        wikis.append(line.strip().split())

lc = 0
while True:
    line = famr.readline()
    if not line:
        break
    line = line.rstrip()
    if line.strip()=="" :
        lc += 1
    if ":name" in line:

        #get name
        namelines = []
        nextline = famr.readline()
        namelines.append(nextline.rstrip())
        tok = ""
        while "op" in nextline:
            tok += nextline[nextline.find("\"")+1:nextline.rfind("\"")] + " "
            if ")" in nextline:
                break
            nextline = famr.readline()
            namelines.append(nextline.rstrip())
        tok = tok.strip()

        #get wiki of the name
        #print tok
        if tok in wiki_ht:
            wiki = wiki_ht[tok]
            line = line.replace(":name",":wiki " + wiki + "\t:name")
        else:
            if tok != "":
                for i in range(len(all_wikis[lc])):
                    if tok.split()[0] in all_wikis[lc][i][0] :# or (all_wikis[lc][i][1] != '-' and all_wikis[lc][i][1] == tok) or tok in all_wikis[lc][i][0] :
                        wiki = all_wikis[lc][i][1]
                        if wiki != '-':
                            wiki = "\""+wiki+"\""
                        line = line.replace(":name",":wiki " + wiki + "\t:name")
                        break

        print(line)
        print("\n".join(namelines))
    else:
        print(line)
