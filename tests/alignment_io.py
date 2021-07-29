from transition_amr_parser.io import read_amr2
from collections import defaultdict


def get_node_hash(amr, nid):
    parents = [f'{amr.nodes[id]} {edge}' for id, edge in amr.parents(nid)]
    children = [f'{edge} {amr.nodes[id]}' for id, edge in amr.children(nid)]
    return f'{amr.nodes[nid]} ' \
        + ' '.join(sorted(parents)) + ' | '\
        + ' '.join(sorted(children))


def map_node_ids(amr, ref_amr):

    # id ref AMR by name+children+parent hash
    ref_node_by_key = defaultdict(list)
    for nid in ref_amr.nodes.keys():
        ref_node_by_key[get_node_hash(ref_amr, nid)].append(nid)

    # assign nodes based on hash
    nid_map = {}
    num_amb = 0
    missing = []
    ref_found = []
    for nid, _ in amr.nodes.items():
        key = get_node_hash(amr, nid)
        if key not in ref_node_by_key:
            missing.append(nid)
            continue
        ref_ids = ref_node_by_key[key]
        if len(ref_ids) > 1:
            num_amb += 1
        nid_map[nid] = ref_ids[0]
        ref_found.append(ref_ids[0])

    # assing missing nodes based on name. This assumes missing names are due to
    # reasons other than ambigous names
    if missing:
        ref_missing = list(set(ref_amr.nodes.keys()) - set(ref_found))
        ref_miss_by_name = {ref_amr.nodes[nid]: nid for nid in ref_missing}
        for nid in missing:
            key = amr.nodes[nid]
            if key not in ref_miss_by_name:
                raise Exception('AMR alignment failed')
            nid_map[nid] = ref_miss_by_name[key]

    return nid_map


# got data from DATA.AMR1-3.cofill_alignments.zip
unaligned_amr_file = 'DATA/AMR2.0/corpora/train.txt.no_wiki'
reference_aligned_amr_file = 'DATA/AMR2.0/aligned/cofill/train.txt'

# this is the normal imput, no JAMR, no alignments
amrs = read_amr2(unaligned_amr_file, ibm_format=False, tokenize=True)

# we want to end up printing something like this
ref_amrs = read_amr2(reference_aligned_amr_file, ibm_format=True)


# simulating that I add alignments
assert len(amrs) == len(ref_amrs)
aligned_amrs = []
for amr, ref_amr in zip(amrs, ref_amrs):

    # simulated alignment, here we select at random
    # NOTE: We need to give tokens consisten with alignment!
    amr.tokens = ref_amr.tokens

    # map node ids in, otherwise identical, AMR graphs
    nid_map = map_node_ids(amr, ref_amr)

    # use map to assign alignments
    amr.alignments = {}
    for nid in amr.nodes.keys():
        amr.alignments[nid] = ref_amr.alignments[nid_map[nid]]

    aligned_amrs.append(amr)

# writing them to file
with open('tmp.amr', 'w') as fid:
    for amr in aligned_amrs:
        fid.write(f'{amr.__str__(jamr=True)}\n')
