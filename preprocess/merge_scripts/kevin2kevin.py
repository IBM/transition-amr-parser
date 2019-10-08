#!/usr/bin/python
import operator
import sys
import re
import string
import os


def kevin2kevin ( aline ) :
    alns = []
    lvls = []
    lvls.append("1")
    aline = aline.strip()
    
    for (i,c) in enumerate(aline):
        if c == '(':
            #lvls[-1] = str(int(lvls[-1]) + 1)
            lvls.append(".")
            lvls.append("0")
        if c == ':':
            nextcolonidx = aline.find(":",i+1)
            substr = aline[i+1:nextcolonidx]
            toks = substr.split()
            if toks[0] == 'wiki':
                continue
            if len(toks)==2 and len(toks[1]) <=2 and toks[1] != '-' and toks[1][0].isalpha():
                continue
            lvls[-1] = str(int(lvls[-1]) + 1)
        if c == ')':
            lvls.pop()
            lvls.pop()
        if c == '~' and aline[i+1] == 'e':
            j = i+3
            idx = ""
            while aline[j].isdigit():
                idx = idx + aline[j]
                j = j + 1
            node = "".join(lvls)
            if lvls[-1] == "0":
                node = "".join(lvls[0:-2])

            is_label = False
            k = i-1 
            while k >= 0 and aline[k] != " " :
                k = k - 1
            if aline[k+1] == ":" :
                is_label = True
            if not is_label:
                alns.append( idx + "-" + node)
    return alns

#print "\n\n"
#print kevin2kevin(sys.argv[1])
#exit()

fin = open(sys.argv[1])
for line in fin:
    line = line.strip()
    alns = kevin2kevin(line)
    print " ".join(set(alns))
