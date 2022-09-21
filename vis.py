from ctypes import alignment
from macpath import join
import sys
from colorama import Fore, Back, Style
from transition_amr_parser.docamr_io import (
    AMR,
    read_amr
)
from docSmatch import smatch
from side_by_side import print_side_by_side
import pickle
from tqdm import tqdm
import os



    
def main(args): 
    # gold = read_amr(sys.argv[1]).values()
    gold = read_amr(args.f[1]).values()
    dev = read_amr(args.f[0]).values()
    f1 = open(args.f[0],'r')
    f2 = open(args.f[1],'r')
    # out_file = open('vis.out','w')
    if os.path.isfile('best_gold_to_pred_alignments.pt'):
        pickle_file = open('best_gold_to_pred_alignments.pt','rb')
        gold_to_pred_alignments = pickle.load(pickle_file)
    else:
        gold_to_pred_alignments = []
        for sent_num, (cur_amr1, cur_amr2) in tqdm(enumerate(smatch.generate_amr_lines(f1, f2), start=1), desc='Retrieving Smatch alignments'):
            __, __,*a = smatch.get_amr_match(cur_amr1, cur_amr2,
                                            sent_num=sent_num,  # sentence number
                                            coref=True,get_alignment=True)
            if len(a)>0:
                gold_to_pred_alignments.append(a[0])
        with open('best_gold_to_pred_alignmens.pt','wb') as pickle_file:
            pickle.dump(gold_to_pred_alignments,pickle_file)                                                       
    
    for num,(amr,amr2) in enumerate(zip(gold,dev)):
        print("DOC ",num)
        tokens = amr.tokens
        tokens2 = amr2.tokens
        coref_nodes = []
        coref_nodes2 = []
        for node in amr.nodes:
            if amr.nodes[node] == 'coref-entity':
                coref_nodes.append(node)
        
        for node in amr2.nodes:
            if amr2.nodes[node] == 'coref-entity':
                coref_nodes2.append(node)

        coref_chains = {}
        coref_chains2 = {}
        chain_nodes2 = {}
        chain_nodes = {}

        for cnode2 in coref_nodes2:
            coref_chains2[cnode2] = []
            chain_nodes2[cnode2] = []
            
            for e in amr2.edges:
                if e[0] not in amr2.nodes or e[2] not in amr2.nodes:
                    continue
                if e[1] == ':coref' and e[0] == cnode2:
                    if e[2] in amr2.alignments:
                        coref_chains2[cnode2].extend(amr2.alignments[e[2]])
                    chain_nodes2[cnode2].append(amr2.nodes[e[2]])
                    
                    

                if e[1] == ':coref-of' and e[2] == cnode2:
                    if e[0] in amr2.alignments:
                        coref_chains2[cnode2].extend(amr2.alignments[e[0]])
                    chain_nodes2[cnode2].append(amr2.nodes[e[0]])
                    
                    
        
        for cnode in coref_nodes:
            coref_chains[cnode] = []
            chain_nodes[cnode] = []
            
            

            for e in amr.edges:
                if e[0] not in amr.nodes or e[2] not in amr.nodes:
                    continue
                if e[1] == ':coref' and e[0] == cnode:
                    if e[2] in amr.alignments:
                        coref_chains[cnode].extend(amr.alignments[e[2]])
                    chain_nodes[cnode].append(amr.nodes[e[2]])
                    
                if e[1] == ':coref-of' and e[2] == cnode:
                    if e[0] in amr.alignments:
                        coref_chains[cnode].extend(amr.alignments[e[0]])
                    chain_nodes[cnode].append(amr.nodes[e[0]])
                    
            
            
                    
            outstr=""
            to_print = []
            to_print2  = []
            pred_coref_node = gold_to_pred_alignments[num][cnode]

            if args.different_doc:
                for i,(tok,tok2) in enumerate(zip(tokens,tokens2)):
                    is_chain = False
                    if i in coref_chains[cnode]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        to_print.append(f"{Fore.BLUE+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                        is_chain = True
                        # to_print.append(" ")
                    else:
                        #print(tok, end=" ")
                        to_print.append(tok)
                        # to_print.append(" ")
                    if i in coref_chains2[pred_coref_node]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        # CORRECT mattch with gold
                        if is_chain:
                            to_print2.append(f"{Fore.GREEN+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                        else:
                            to_print2.append(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                        # to_print2.append(" ")
                    else:
                        #print(tok, end=" ")
                        # MISSED COREF NODE
                        # if is_chain:
                        #     to_print2.append(f"{Fore.RED+Style.BRIGHT}"+tok2+f"{Style.RESET_ALL}")
                        # else:
                            to_print2.append(tok2)
                        # to_print2.append(" ")
                
                print_side_by_side(" ".join(to_print)," ".join(to_print2),col_padding=24, delimiter='|++++|')
            else:
                for i,tok in enumerate(tokens):
                    is_chain = False
                    color = None
                    if i in coref_chains[cnode]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        is_chain = True
                        # to_print.append(" ")
                    
                    if pred_coref_node is not None and i in coref_chains2[pred_coref_node]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        # CORRECT mattch with gold
                        if is_chain:
                            color = f"{Fore.GREEN+Style.BRIGHT}"
                        else:
                            color = f"{Fore.RED+Style.BRIGHT}"
                        # to_print2.append(" ")
                    if color is None:
                        #print(tok, end=" ")
                        # MISSED COREF NODE
                        if is_chain:
                            color = f"{Fore.BLUE+Style.BRIGHT}"
                        else:
                            color = ''
                        # to_print2.append(" ")
                    to_print.append(color+tok+f"{Style.RESET_ALL}")
            
            original_stdout = sys.stdout
            print(" ".join(to_print))
            
            
            print('\nNodes corefed in gold',f"{Fore.BLUE+Style.BRIGHT}"+' '.join(chain_nodes[cnode])+f"{Style.RESET_ALL}")
            print('Nodes corefed in pred',f"{Fore.MAGENTA+Style.BRIGHT}"+' '.join(chain_nodes2[pred_coref_node] if pred_coref_node in chain_nodes2 else ' No nodes' )+f"{Style.RESET_ALL}")
            

            print("\n===============================")
            input("Press Enter to continue... \n")
    # out_file.close()
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DocAMR visualization")
    parser.add_argument(
        '-f',
        nargs=2,
        type=str,
        help=('Two files containing AMR pairs. '
              'AMRs in each file are separated by a single blank line'))
    
    parser.add_argument(
        '--different-doc',
        action='store_true',
        default=False,
        help='if two different docamrs (two different tokenized text) needs to be compared')

    args = parser.parse_args()
    main(args)