import os
import sys


if __name__ == '__main__':

    famr, fwiki, wiki_folder = sys.argv[1:]

    famr = open(famr)
    fwiki = open(fwiki)
    ftrn = open(f"{wiki_folder}/trn.wikis")
    wiki_ht = {}
    for line in ftrn:
        if len(line.strip().split('\t')) != 2:
            continue
        (n,w) = line.strip().split('\t')
        wiki_ht[n] = w

    # FIXME: Temporary fix for AMR3.0
    amr3_file = f"{wiki_folder}/amr3trn.wikis"
    if os.path.isfile(amr3_file):
        ftrn3 = open(amr3_file)
        for line in ftrn3:
            if len(line.strip().split('\t')) != 2:
                continue
            (n,w) = line.strip().split('\t')
            if n[-1:]==' ':
                n = n[:-1]
            if n not in wiki_ht:
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
