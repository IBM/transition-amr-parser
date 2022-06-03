from transition_amr_parser.io import read_blocks
from transition_amr_parser.amr import AMR, protected_tokenizer
import re
import penman
import json
import os
from ipdb import set_trace


def get_token_positions(tokens, sentence):
    # get the char offset for each token in the JAMR annotated AMR
    positions = []
    i = 0
    for j, token in enumerate(tokens):
        start = i + sentence[i:].find(token)
        dend = start + len(token)
        positions.append((start, dend))
        i = dend

    return positions


def get_sentence_alignments(aid, amr, positions, map):

    sentence_alignments = dict(id=aid, node_offsets={})
    for nid, nname in amr.nodes.items():
        nid2 = map[nid]
        jamr_alignments = amr.alignments.get(nid, None)
        if jamr_alignments is None:
            # single token matching
            sentence_alignments['node_offsets'][nid2] = None
        else:
            # mutiple tokens matching
            sentence_alignments['node_offsets'][nid2] = \
                [positions[pp] for pp in jamr_alignments]

    return sentence_alignments


def add_alignments(amrp, sentence_alignments):

    # store alignments in ISI format match jamr tokenization with
    # target tokenization below
    tokensp, pos = protected_tokenizer(amrp.metadata['snt'])
    amrp.metadata['tok'] = ' '.join(tokensp)
    for instance in amrp.instances():

        nid2 = instance.source
        node_offsets = sentence_alignments['node_offsets'][nid2]

        if node_offsets is None:
            continue

        key = (instance.source, ':instance', instance.target)

        # if the offsets match the target tokenization below, just use it,
        # if not greedely select highest overlap segment
        def overlap(seg_a, seg_b):
            return max(
                0,
                min(seg_a[-1], seg_b[-1]) - max(seg_a[0], seg_b[0])
            )
        eal = []
        for o in node_offsets:
            if o in pos:
                # exact match
                eal.append(pos.index(o))
            else:
                # get highest matching segment
                cands = [(overlap(o, x), j) for j, x in enumerate(pos)]
                eal.append(sorted(cands, key=lambda x: x[0])[-1][1])

        # print(penman.encode(amrp))
        align = penman.surface.Alignment(tuple(eal), prefix='')
        if key in amrp.epidata:
            amrp.epidata[key].append(align)
        else:
            amrp.epidata[key] = [align]


def realign_ner(amr_tmp):

    # fix named entities alignments
    for nid, nname in amr_tmp.nodes.items():
        ner_tag = [n for n, edge in amr_tmp.parents(nid) if ':name' == edge]
        if nname == 'name' and ner_tag:
            leaves = [
                n
                for n, e in sorted(
                    amr_tmp.children(nid),
                    key=lambda x: x[1]
                ) if re.match(':op[0-9]+', e)
            ]

            # get positions from leaves
            leaves_positions = []
            for n in leaves:
                if n not in amr_tmp.alignments or amr_tmp.alignments[n] is None:
                    continue
                for p in amr_tmp.alignments[n]:
                    leaves_positions.append(p)

            leaves_positions = sorted(set(leaves_positions))

            if len(leaves) > 1:
                if len(leaves) == len(leaves_positions):
                    # align leaves by order
                    for i, leaf in enumerate(leaves):
                        amr_tmp.alignments[leaf] = [leaves_positions[i]]
                    # ensure subgraph aligned to first leaf
                    amr_tmp.alignments[nid] = leaves_positions[0:1]
                    amr_tmp.alignments[ner_tag[0]] = leaves_positions[0:1]
                elif len(leaves_positions) == 1:
                    # ensure subgraph aligned to leaf
                    amr_tmp.alignments[nid] = leaves_positions
                    amr_tmp.alignments[ner_tag[0]] = leaves_positions
                else:
                    pass

            elif len(leaves_positions):
                # ensure subgraph aligned to leaf
                amr_tmp.alignments[nid] = leaves_positions
                amr_tmp.alignments[ner_tag[0]]

    return amr_tmp


def main():

    offsets = []
    # for tag in ['dev']:
    for tag in ['train', 'dev', 'test']:

        # read map
        maps = []
        with open(f'/dccstor/multi-parse/seq2seq/amr2.0.{tag}.nvars') as fid:
            for line in fid.readlines():
                if not line.startswith('Read'):
                    maps.append(exec(f'maps.append({line.rstrip()})'))
        maps = list(filter(None, maps))

        # read AMRs
        amrps = []
        aafile = f'DATA/AMR2.0/corpora/{tag}.txt'
        original_blocks = read_blocks(aafile)

        afile = f'DATA/AMR2.0/aligned/cofill/{tag}.txt'
        print(f'Read {afile}')
        for index, penman_txt in enumerate(read_blocks(afile)):

            amr = AMR.from_metadata(penman_txt)
            # for meta data
            amrp = penman.decode(original_blocks[index])

            # when possible align NER leaves to each token and parwents to
            # first leaf
            amr = realign_ner(amr)

            # determine positions of tokens for the JAMR tokens
            positions = get_token_positions(amr.tokens, amr.sentence)

            # get JAMR alignments to untokenized sentence (chat offsets)
            sentence_alignments = get_sentence_alignments(
                amrp.metadata['id'], amr, positions, maps[index])
            # store later
            offsets.append(sentence_alignments)

            # add alignments to PENMAN amr notation with its on tokenization
            add_alignments(amrp, sentence_alignments)
            #
            amrps.append(amrp)

        # write offsets
        new_folder = 'DATA/AMR2.0/aligned/cofill_penman/'
        os.makedirs(new_folder, exist_ok=True)
        new_file = f'{new_folder}/{tag}.jsonl'
        with open(new_file, 'w') as fid:
            for sal in offsets:
                fid.write(f'{json.dumps(sal)}\n')
        print(new_file)

        # write alignments
        new_file = f'{new_folder}/{tag}.txt'
        with open(new_file, 'w') as fid:
            for amrp in amrps:
                fid.write(f'{penman.encode(amrp)}\n\n')
        print(new_file)


if __name__ == '__main__':
    main()
