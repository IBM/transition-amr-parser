from argparse import ArgumentParser
from copy import deepcopy
from transition_amr_parser.io import read_blocks
from transition_amr_parser.docamr_io import read_amr_penman, process_corefs, make_pairwise_edges,read_amr_metadata
from transition_amr_parser.doc_amr import connect_sen_amrs,make_doc_amrs
from transition_amr_parser.doc_amr_baseline.corefconversion.conll_transform import read_file as conll_read_file
from transition_amr_parser.doc_amr_baseline.corefconversion.conll_transform import compute_chains as conll_compute_chains
from transition_amr_parser.doc_amr_baseline.make_doc_amr import process_coref_conll
from transition_amr_parser.doc_amr_baseline.baseline_io import read_amr_add_sen_id
import penman
import pickle
import os
import glob
import copy

def write_doc_amr_from_sen(in_amr,coref_fof,fof_path,coref_type,out_amr,norm='no_merge',dont_make_pairwise_edges=False):
    
        
    # Read AMR as a generator with tqdm progress bar
    # tqdm_amrs = read_blocks(amr_file)
    # tqdm_amrs.set_description(f'Computing oracle') 
    if os.path.isdir(in_amr):
        in_amr+='/'
        pat='*'
        amrs = {}
        
        if not glob.glob(in_amr+pat+'.amr'):
            if not glob.glob(in_amr+pat+'.parse'):
                raise Exception("--path_to_amr folder does not contain .amr files or .parse files ")
            else:
                filepaths = glob.iglob(in_amr+pat+'.parse')
        else:
            filepaths = glob.iglob(in_amr+pat+'.amr')
        
        filepaths = list(filepaths)
        amrs_dict = {}
        for filepath in filepaths:
            doc_id = filepath.split('/')[-1].split('.')[0]
            #FIXME hardcoding jamr alignment ibm_format=True
            tqdm_amrs_str = read_blocks(filepath)
            # amrs[doc_id] = read_amr_add_sen_id(filepath,doc_id,remove_id=True,tokenize=False,ibm_format=True)
            amrs[doc_id] = read_amr_metadata(tqdm_amrs_str,doc_id=doc_id,add_id=True)
            amrs_dict.update(amrs[doc_id])
    else:
        tqdm_amrs_str = read_blocks(in_amr)
        amrs = read_amr_penman(tqdm_amrs_str)
    
    chains = None
    if coref_type=='gold':
        assert fof_path is not None,'fof path  not given'
        coref_files = [fof_path+line.strip() for line in open(coref_fof)]
        corefs = process_corefs(coref_files)
        chains = True
        
    elif coref_type=='conll':
        coref_chains = {}
        out = conll_read_file(coref_fof)
        for n,(i,val) in enumerate(out.items()):
            
            docid_spl = i.split('); part ')
            doc_id = docid_spl[0].split('/')[-1]+'_'+str(int(docid_spl[1]))
            coref_chains[doc_id] = conll_compute_chains(val)
            assert len(coref_chains)>0,"Coref file is empty"
        corefs = process_coref_conll(amrs,coref_chains,add_coref=True,save_triples=False,out=None)
        amrs = copy.deepcopy(amrs_dict)
        chains = False
        
    elif coref_type=='allennlp':
        coref_chains = {}
        out = pickle.load(open(coref_fof,'rb'))
        for i,(doc_id,val) in enumerate(out.items()):

            coref_chains[doc_id] = val
        assert len(coref_chains)>0,"Coref file is empty"
        corefs = process_coref_conll(amrs,coref_chains,add_coref=True,verbose=False,save_triples=False)
        amrs = copy.deepcopy(amrs_dict)
        chains = False
        
    else:
        raise Exception('Invalid coref_type')
    # plain_doc_amrs = make_doc_amrs(corefs,amrs,coref=False).values()
    
    # if out_amr is None:
    #     args.out_amr = args.out_actions.rstrip('.actions')+'_'+args.norm+'.docamr'
   
    # use corefs to merge sentence level AMRs into Documentr level AMRs
    damrs = make_doc_amrs(corefs,amrs,chains=chains).values()
    
    with open(out_amr, 'w') as fid:
        for amr in damrs:
            damr = deepcopy(amr)
            connect_sen_amrs(damr)
            damr.normalize(norm)
            if not dont_make_pairwise_edges:
                damr = make_pairwise_edges(damr)
            #get sentence ends indices
            damr.get_sen_ends()
            #FIXME remove unicode in every token of sentence
            damr.remove_unicode()
            #manually aligning document top to last token
            damr.penman = penman.decode(damr.to_penman(jamr=args.jamr))
            document_top,top_rel,top_name = damr.penman.triples[0]
            if top_name=='document':
            # for node_id,node_name in damr.nodes.items():
            #     if node_name =='document':
            #         document_top = node_id
            #         break

                damr.alignments[document_top] = [len(damr.tokens)-1]
            # damr.alignments[document_top] = [0]
            damr.check_connectivity()
            fid.write(damr.__str__(jamr=args.jamr))

def main(args):
    write_doc_amr_from_sen(args.in_amr,args.coref_fof,args.fof_path,args.coref_type,args.out_amr,args.norm,args.dont_make_pairwise_edges)

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
        type=str,
        default=None
    )
    parser.add_argument(
        "--norm",
        help="norm of DocAMR ",
        type=str,
        default='no-merge'
    )
    parser.add_argument(
        "--out-amr",
        help="path to save docamr",
        type=str
    )
    parser.add_argument(
        "--coref-type",
        help="type of coref (gold,allennlp,conll)",
        type=str,
        default='gold'
    )
    parser.add_argument(
        "--jamr",
        help="if the alignments are in jamr notation",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--dont-make-pairwise-edges",
        help="make pairwise edges out of coref edges",
        action='store_true'
    )



    args = parser.parse_args()
    main(args)