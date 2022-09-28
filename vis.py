from ctypes import alignment
from macpath import join
import sys
from colorama import Fore, Back, Style
from transition_amr_parser.docamr_io import (
    AMR,
    read_amr
)
from docSmatch import smatch
# from side_by_side import print_side_by_side
import pickle
from tqdm import tqdm
import os
colors = {}
colors['terminal'] = {'red':f"{Fore.RED+Style.BRIGHT}",'blue':f"{Fore.BLUE+Style.BRIGHT}",'green':f"{Fore.GREEN+Style.BRIGHT}",'magenta':f"{Fore.MAGENTA+Style.BRIGHT}",'reset':f"{Style.RESET_ALL}"}
colors['html'] = {'red':'<span style="background-color:pink;font-weight:bold;">','blue':'<span style="background-color:cyan;font-weight:bold;">','green':'<span style="background-color:lime;font-weight:bold;">','magenta':'<span style="color:darkmagenta;font-weight:bold;">','reset':'</span>'}
htmlclose='</body></html>'



    
def main(args): 
    # gold = read_amr(sys.argv[1]).values()
    gold = read_amr(args.f[0]).values()
    dev = read_amr(args.f[1]).values()
    f1 = open(args.f[1],'r')
    f2 = open(args.f[0],'r')

    scores = []
    # out_file = open('vis.out','w')
    if os.path.isfile('DATA/best_gold_to_pred_alignments.pt'):
        pickle_file = open('DATA/best_gold_to_pred_alignments.pt','rb')
        gold_to_pred_alignments = pickle.load(pickle_file)
        pickle_file = open('DATA/scores.pt','rb')
        scores = pickle.load(pickle_file)
    else:
        gold_to_pred_alignments = []
        for sent_num, (cur_amr1, cur_amr2) in tqdm(enumerate(smatch.generate_amr_lines(f1, f2), start=1), desc='Getting Smatch alignments'):
            xx, ss,*a = smatch.get_amr_match(cur_amr1, cur_amr2,
                                            sent_num=sent_num,  # sentence number
                                            coref=True,get_alignment=True)
            p = float(xx[0])/xx[1]
            r = float(xx[0])/xx[2]
            scores.append((2*p*r)/(p+r))
            if len(a)>0:
                gold_to_pred_alignments.append(a[0])
        with open('DATA/best_gold_to_pred_alignments.pt','wb') as pickle_file:
            pickle.dump(gold_to_pred_alignments,pickle_file)
        with open('DATA/scores.pt','wb') as pickle_file:
            pickle.dump(scores,pickle_file)
    
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
        chain_nodes_ids2 = {}
        chain_nodes_aligned_nodes = {}
        
        if args.output_html:
            htmlstr = '''
                    <html>
                        <body>
                            <h1>Coref chains in gold amr vs pred amr</h1>
                            <h3> <span style="background-color:lime;font-weight:bold;">Green</span> : coref match (true positive) <br>
                                <span style="background-color:cyan;font-weight:bold;">Blue</span> : gold coref but not in pred (false negative) <br>
                                <span style="background-color:pink;font-weight:bold;">Red</span> : pred coref but not in gold (false positive) <br>
                            </h3>
                    '''
            htmlstr += "<h3><br>Score: " + str(round(scores[num],2)) + "   <br> </h3><hr>"
            
            if not os.path.isdir('DATA/vis_output_html'):
                os.mkdir('DATA/vis_output_html')
            htmlfile = open("DATA/vis_output_html/corefchains_doc"+str(num)+".html","w")
            htmlfile.write(htmlstr)



        for cnode2 in coref_nodes2:
            coref_chains2[cnode2] = []
            chain_nodes2[cnode2] = []
            chain_nodes_ids2[cnode2] = []
            
            for e in amr2.edges:
                if e[0] not in amr2.nodes or e[2] not in amr2.nodes:
                    continue
                if e[1] == ':coref' and e[0] == cnode2:
                    if e[2] in amr2.alignments:
                        coref_chains2[cnode2].extend(amr2.alignments[e[2]])
                    chain_nodes2[cnode2].append(amr2.nodes[e[2]])                    
                    chain_nodes_ids2[cnode2].append(e[2])

                if e[1] == ':coref-of' and e[2] == cnode2:
                    if e[0] in amr2.alignments:
                        coref_chains2[cnode2].extend(amr2.alignments[e[0]])
                    chain_nodes2[cnode2].append(amr2.nodes[e[0]])
                    chain_nodes_ids2[cnode2].append(e[0])
                    
        
        for cnode in coref_nodes:
            coref_chains[cnode] = []
            chain_nodes[cnode] = []
            chain_nodes_aligned_nodes[cnode] = []
            

            for e in amr.edges:
                if e[0] not in amr.nodes or e[2] not in amr.nodes:
                    continue
                aligned_node = None
                if e[1] == ':coref' and e[0] == cnode:
                    if e[2] in amr.alignments:
                        coref_chains[cnode].extend(amr.alignments[e[2]])
                    chain_nodes[cnode].append(amr.nodes[e[2]])
                    if e[2] in gold_to_pred_alignments[num]:
                        aligned_node = gold_to_pred_alignments[num][e[2]]
                    chain_nodes_aligned_nodes[cnode].append(aligned_node)
                                                                                   
                if e[1] == ':coref-of' and e[2] == cnode:
                    if e[0] in amr.alignments:
                        coref_chains[cnode].extend(amr.alignments[e[0]])
                    chain_nodes[cnode].append(amr.nodes[e[0]])
                    if e[0] in gold_to_pred_alignments[num]:
                        aligned_node = gold_to_pred_alignments[num][e[0]]
                    chain_nodes_aligned_nodes[cnode].append(aligned_node)
                    
                    
            outstr=""
            to_print = []
            to_print2  = []
            pred_coref_node = None
            if cnode in gold_to_pred_alignments[num]:
                pred_coref_node = gold_to_pred_alignments[num][cnode]
            
               
            if args.different_doc:
                assert args.output_html is not True,'Not Implemented'
                raise NotImplementedError
                # for i,(tok,tok2) in enumerate(zip(tokens,tokens2)):
                #     is_chain = False
                #     if i in coref_chains[cnode]:
                #         # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                #         to_print.append(f"{Fore.BLUE+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                #         is_chain = True
                #         # to_print.append(" ")
                #     else:
                #         #print(tok, end=" ")
                #         to_print.append(tok)
                #         # to_print.append(" ")
                #     if i in coref_chains2[pred_coref_node]:
                #         # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                #         # CORRECT mattch with gold
                #         if is_chain:
                #             to_print2.append(f"{Fore.GREEN+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                #         else:
                #             to_print2.append(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}")
                #         # to_print2.append(" ")
                #     else:
                #         #print(tok, end=" ")
                #         # MISSED COREF NODE
                #         # if is_chain:
                #         #     to_print2.append(f"{Fore.RED+Style.BRIGHT}"+tok2+f"{Style.RESET_ALL}")
                #         # else:
                #             to_print2.append(tok2)
                #         # to_print2.append(" ")
                
                # print_side_by_side(" ".join(to_print)," ".join(to_print2),col_padding=24, delimiter='|++++|')
            else:
                if args.output_html:
                    color_picker = colors['html']
                else:
                    color_picker = colors['terminal']
                for i,tok in enumerate(tokens):
                    is_chain = False
                    color = None
                    if i in coref_chains[cnode]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        is_chain = True
                        # to_print.append(" ")
                    
                    if pred_coref_node is not None and pred_coref_node in coref_chains2 and i in coref_chains2[pred_coref_node]:
                        # print(f"{Fore.RED+Style.BRIGHT}"+tok+f"{Style.RESET_ALL}", end=" ")
                        # CORRECT mattch with gold
                        if is_chain:
                            color = color_picker['green']
                        else:
                            color = color_picker['red']
                        # to_print2.append(" ")
                    if color is None:
                        #print(tok, end=" ")
                        # MISSED COREF NODE
                        if is_chain:
                            color = color_picker['blue']
                        else:
                            color = ''
                        # to_print2.append(" ")
                    to_print.append(color+tok+color_picker['reset'])
            if args.output_html:
                    htmlout = '<p>'+' '.join(to_print)+'</p>'
                    to_print_node = []
                    for i in range(len(chain_nodes[cnode])):
                        color = ''
                        if pred_coref_node and chain_nodes_aligned_nodes[cnode][i] in chain_nodes_ids2[pred_coref_node]:
                            color = color_picker['green']
                        else:
                            color = color_picker['blue']
                        to_print_node.append(color+chain_nodes[cnode][i]+color_picker['reset'])
                    htmlout+='<p>Nodes corefed in gold ' + ' '.join(to_print_node)+'</p>'
                    #htmlout+='<p>Nodes corefed in pred ' + ' '.join(chain_nodes2[pred_coref_node] if pred_coref_node in chain_nodes2 else ['No','gold','coref','matching', 'pred', 'coref',str(pred_coref_node)])+'</p><br>'
                    htmlout+='<hr>'
                    htmlfile.write(htmlout)
                    
                    
                    
            else:
                original_stdout = sys.stdout
                print(" ".join(to_print))
                
                
                print('\nNodes corefed in gold',f"{Fore.BLUE+Style.BRIGHT}"+' '.join(chain_nodes[cnode])+f"{Style.RESET_ALL}")
                print('Nodes corefed in pred',+' '.join(chain_nodes2[pred_coref_node] if pred_coref_node in chain_nodes2 else ' No nodes' )+f"{Style.RESET_ALL}")
                

                print("\n===============================")
                input("Press Enter to continue... \n")
        
        htmlfile.write(htmlclose)
        htmlfile.close()
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

    parser.add_argument(
        '--output-html',
        action='store_true',
        default=True,
        help='if the output needs to be in html format'
    )

    args = parser.parse_args()
    main(args)
