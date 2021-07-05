import sys

def align_the_unaligned(unalnd_nodes,words,is_alnd_words):
    ret_alns = {}
    for (node, prev) in unalnd_nodes:
        aln = -1
        bestscr = 10000
        for i in range(len(words)):
            if not is_alnd_words[i]:
                scr = abs(prev-i)
                if scr < bestscr:
                    bestscr = scr
                    aln = i
        if aln != -1:
            ret_alns[node] = str(aln)+"-"+str(aln+1)
            is_alnd_words[aln] = True
            #print(node+"\t"+words[aln])

    return ret_alns

fg = open(sys.argv[1])
unalnd_nodes = []
isalnd_words = []
words = []
graph_lines = []
prev = 0
for line in fg:
    line = line.rstrip()
    
    if line == "":
        alns = align_the_unaligned(unalnd_nodes,words,isalnd_words)
        for (i,line) in enumerate(graph_lines):
            if line in alns:
                graph_lines[i] = line+"\t"+alns[line]
        print("\n".join(graph_lines)+"\n")
        unalnd_nodes = []
        isalnd_words = []
        words = []
        graph_lines = []
        prev = 0
        continue

    if "::tok" in line:
        words = line.split()[2:]
        isalnd_words = [False]*len(words)

    if "::node" in line:
        if len(line.split("\t")) == 3:
            unalnd_nodes.append((line,prev))
        else:
            (a,b) = line.split("\t")[-1].split('-')
            if not a.isdigit() or not b.isdigit():
                nline = "\t".join(line.split("\t")[:-1])+" "+line.split("\t")[-1]
                line = nline
                continue
            for i in range(int(a),int(b)):
                if i < len(isalnd_words):
                    isalnd_words[i] = True
            prev = int(b)-1
    graph_lines.append(line)
