from transition_amr_parser.io import read_amr2
from random import choice

# got data from DATA.AMR1-3.cofill_alignments.zip
unaligned_amr_file = 'DATA/AMR2.0/corpora/dev.txt'
reference_aligned_amr_file = 'DATA/AMR2.0/aligned/cofill/dev.txt'

# this is the normal imput, no JAMR, no alignments
amrs = read_amr2(unaligned_amr_file, ibm_format=False, tokenize=True)

# we want to end up printing something like this
ref_amrs = read_amr2(reference_aligned_amr_file, ibm_format=True)

# simulating that I add alignments
assert len(amrs) == len(ref_amrs)
aligned_amrs = []
for amr, ref_amr in zip(amrs, ref_amrs):

    # simulated alignment, here we select at random
    # NOTE: We need to gove tokens consisten with alignment!
    amr.tokens = ref_amr.tokens
    amr.alignments = {}
    for node_id, _ in amr.nodes.items():
        amr.alignments[node_id] = [choice(list(range(len(amr.tokens))))]
    aligned_amrs.append(amr)

# writing them to file
with open('tmp.amr', 'w') as fid:
    for amr in aligned_amrs:
        fid.write(f'{amr.__str__(jamr=True)}\n')
