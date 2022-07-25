from argparse import ArgumentParser
from copy import deepcopy
from transition_amr_parser.io import read_blocks
from transition_amr_parser.docamr_io import read_amr_penman, process_corefs, make_pairwise_edges
from transition_amr_parser.doc_amr import connect_sen_amrs,make_doc_amrs
import penman

def write_doc_amr_from_sen(in_amr,coref_fof,fof_path,out_amr,norm='no_merge'):

    coref_files = [fof_path+line.strip() for line in open(coref_fof)]
    corefs = process_corefs(coref_files)
        
    # Read AMR as a generator with tqdm progress bar
    # tqdm_amrs = read_blocks(amr_file)
    # tqdm_amrs.set_description(f'Computing oracle') 
    tqdm_amrs_str = read_blocks(in_amr)
    # amrs = read_amr_penman(tqdm_amrs_str)
    amrs = read_amr_penman(tqdm_amrs_str)
    damrs = []
    
    # plain_doc_amrs = make_doc_amrs(corefs,amrs,coref=False).values()
    
    # if out_amr is None:
    #     args.out_amr = args.out_actions.rstrip('.actions')+'_'+args.norm+'.docamr'
   
    # use corefs to merge sentence level AMRs into Documentr level AMRs
    amrs = make_doc_amrs(corefs,amrs).values()
    
    with open(out_amr, 'w') as fid:
        for amr in amrs:
            damr = deepcopy(amr)
            connect_sen_amrs(damr)
            damr.normalize(norm)
            
            damr = make_pairwise_edges(damr)
            #get sentence ends indices
            damr.get_sen_ends()
            #FIXME remove unicode in every token of sentence
            damr.remove_unicode()
            #manually aligning document top to last token
            damr.penman = penman.decode(damr.to_penman())
            document_top,top_rel,top_name = damr.penman.triples[0]
            assert top_name=='document'
            damr.alignments[document_top] = [len(damr.tokens)-1]
            # damr.alignments[document_top] = [0]
            damr.check_connectivity()
            fid.write(damr.__str__())

def main(args):
    write_doc_amr_from_sen(args.in_amr,args.coref_fof,args.fof_path,args.out_amr,args.norm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--in-amr",
        help="In file containing AMR in penman format",
        type=str
    )
    parser.add_argument(
        "--coref-fof",
        help="xml files containing AMR coreference information ",
        type=str
    )
    parser.add_argument(
        "--fof-path",
        help="path to coref fof ",
        type=str
    )
    parser.add_argument(
        "--norm",
        help="norm of DocAMR ",
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="path to save docamr",
        type=str
    )




    args = parser.parse_args()
    main(args)