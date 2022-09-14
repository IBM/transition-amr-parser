import sys
from colorama import Fore, Back, Style
from transition_amr_parser.docamr_io import (
    AMR,
    read_amr
)

amrs = read_amr(sys.argv[1]).values()

for amr in amrs:
    tokens = amr.tokens
    coref_nodes = []
    for node in amr.nodes:
        if amr.nodes[node] == 'coref-entity':
            coref_nodes.append(node)

    coref_chains = {}
    for cnode in coref_nodes:
        coref_chains[cnode] = []
        chain_nodes = []
        for e in amr.edges:
            if e[0] not in amr.nodes or e[2] not in amr.nodes:
                continue
            if e[1] == ':coref' and e[0] == cnode:
                coref_chains[cnode].extend(amr.alignments[e[2]])
                chain_nodes.append(amr.nodes[e[2]])
            if e[1] == ':coref-of' and e[2] == cnode:
                coref_chains[cnode].extend(amr.alignments[e[0]])
                chain_nodes.append(amr.nodes[e[0]])
                
        outstr=""
        for (i,tok) in enumerate(tokens):
            if i in coref_chains[cnode]:
                print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
            else:
                print(tok, end=" ")
        print("\n===============================")
        input("Press Enter to continue...")
