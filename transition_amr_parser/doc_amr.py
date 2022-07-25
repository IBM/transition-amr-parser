import argparse
import os
import re
import copy
from tqdm import tqdm

from amr_io import (
    AMR,
    read_amr,
    process_corefs
)
from ipdb import set_trace

def make_doc_amrs(corefs, amrs, coref=True,chains=True):
    doc_amrs = {}

    desc = "making doc-level AMRs"
    if not coref:
        desc += " (without corefs)"
    for doc_id in tqdm(corefs, desc=desc):
        (doc_corefs,doc_sids,fname) = corefs[doc_id]
        if doc_sids[0] not in amrs:
            import ipdb; ipdb.set_trace()
        doc_amr = copy.deepcopy(amrs[doc_sids[0]])
        for sid in doc_sids[1:]:
            if sid not in amrs:
                import ipdb; ipdb.set_trace()
            if amrs[sid].root is None:
                continue
            doc_amr = doc_amr + amrs[sid]
        doc_amr.amr_id = doc_id
        doc_amr.doc_file = fname
        if coref:
            if chains:
                doc_amr.add_corefs(doc_corefs)
            else:
                doc_amr.add_edges(doc_corefs)
        
        doc_amrs[doc_id] = doc_amr

    return doc_amrs

def connect_sen_amrs(amr):

    if len(amr.roots) <= 1:
        return

    node_id = amr.add_node("document")
    amr.root = str(node_id)
    for (i,root) in enumerate(amr.roots):
        amr.edges.append((amr.root, ":snt"+str(i+1), root))


def main(args):

    assert args.out_amr        

    if args.amr3_path and args.coref_fof:
        
        # read cross sentenctial corefs from document AMR
        coref_files = [args.amr3_path+"/"+line.strip() for line in open(args.coref_fof)]
        corefs = process_corefs(coref_files)

        # Read AMR
        directory = args.amr3_path + r'/data/amrs/unsplit/'
        amrs = {}
        for filename in tqdm(os.listdir(directory), desc="Reading sentence-level AMRs"):
            amrs.update(read_amr(directory+filename))

        # write documents without corefs
        plain_doc_amrs = make_doc_amrs(corefs,amrs,coref=False).values()
        with open(args.out_amr+".nocoref", 'w') as fid:
            for amr in plain_doc_amrs:
                damr = copy.deepcopy(amr)
                connect_sen_amrs(damr)
                fid.write(damr.__str__())        
        # add corefs into Documentr level AMRs
        amrs = make_doc_amrs(corefs,amrs).values()
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                damr = copy.deepcopy(amr)
                connect_sen_amrs(damr)
                print("\nnormalizing "+damr.doc_file.split("/")[-1])
                print("normalizing "+damr.amr_id)
                damr.normalize(rep=args.rep, flip=args.flipped)
                fid.write(damr.__str__())

    if args.in_doc_amr_unmerged :
        amrs = read_amr(args.in_doc_amr_unmerged).values()
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                damr = copy.deepcopy(amr)
                print("\nnormalizing "+damr.amr_id)
                damr.normalize(rep=args.rep, flip=args.flipped)
                fid.write(damr.__str__())

    if args.in_doc_amr_pairwise :
        amrs = read_amr(args.in_doc_amr_pairwise).values()
        with open(args.out_amr, 'w') as fid:
            for amr in amrs:
                damr = copy.deepcopy(amr)
                print("\nnormalizing "+damr.amr_id)
                damr.make_chains_from_pairs(args.pairwise_coref_rel)
                damr.normalize(rep=args.rep, flip=args.flipped)
                fid.write(damr.__str__())
                
def argument_parser():

    parser = argparse.ArgumentParser(description='Read AMRs and Corefs and put them together', \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--amr3-path",
        help="path to AMR3 annoratations",
        type=str
    )
    parser.add_argument(
        "--coref-fof",
        help="File containing list of xml files with coreference information ",
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="Output file containing AMR in penman format",
        type=str,
    )
    parser.add_argument(
        "--in-doc-amr-unmerged",
        help="path to a doc AMR file in 'no-merge' format",
        type=str
    )    
    parser.add_argument(
        "--in-doc-amr-pairwise",
        help="path to a doc AMR file with coref chains as pairwise edges",
        type=str
    )    
    parser.add_argument(
        "--pairwise-coref-rel",
        default='same-as',
        help="edge label representing pairwise coref edges",
        type=str
    )    
    parser.add_argument(
        '--rep',
        default='docAMR',
        help='''Which representation to use, options: 
        "no-merge" -- No node merging, only chain-nodes
        "merge-names" -- Merge only names
        "docAMR" -- Merge names and drop pronouns
        "merge-all" -- Merge all nodes''',
        type=str
    )
    parser.add_argument(
        '--flipped',
        help='whether or not to use the flipped representation i.e. parent->coref-entity->child',
        action='store_true'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
