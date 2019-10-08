#!/usr/bin/python
import operator
import sys
import re
import string
import os

fin = open(sys.argv[1])

key_words = ["name","date-entity","score-entity"]

i = -1
j = -1

for oline in fin:
    line = oline.strip()
    if "::root" in line:
        i = -1
        j = -1
    if "::node" in line:
        if line.split()[3] in key_words:
            length = len(line.split("\t"))
            rng_str = line.split()[-1]
            if length == 4:
                (l,r)  = rng_str.split("-")
                i = int(l)
                j = int(r)
                if j-i == 1:
                    i = -1
                    j = -1
        else:
            if i >= 0:
                length = len(line.split("\t"))
                rng_str = line.split()[-1]
                if length == 4:
                    (l,r)  = rng_str.split("-")
                    #print line
                    if i <= int(l) and j >= int(r):
                        rng_str = str(i)+'-'+str(j)
                        #print line
                    toks = line.split("\t")
                    toks[-1] = rng_str
                    print "\t".join(toks)

                    if j < int(r):
                        i = -1
                        j = -1

                    continue
                else:
                    i = -1
                    j = -1

    

    print oline.rstrip()
        
