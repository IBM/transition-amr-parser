import re
from tqdm import tqdm
import sys
from collections import defaultdict, Counter
from transition_amr_parser.io import writer


def read_bio(file_path):
    with open(file_path) as fid:
        raw_bios = []
        raw_bio = []
        for line in fid.readlines():
            if line.strip():
                raw_bio.append(line.strip()) 
            else:
                raw_bios.append(raw_bio)
                raw_bio = []
    return raw_bios


if __name__ == '__main__':

    in_tags, out_basename = sys.argv[1:]

    sentences = read_bio(in_tags)

    sense_regex = re.compile('(.*) [BI]-PRED\((.*-[0-9]+)\)')
    pred_regex = re.compile('(.*) [BI]-PRED\((.*)\)')
    addnode_regex = re.compile('(.*) [BI]-ADDNODE\((.*)\)')
    blank_regex = re.compile('(.*) O')

    # store counts
    token_by_sense = defaultdict(lambda: Counter())
    token_by_lemma = defaultdict(lambda: Counter())
    token_by_addnode = defaultdict(lambda: Counter())

    # Word senses PRED(*-<number>)
    wsd_write = writer(f'{out_basename}.wsd', add_return=True)
    # All ADDNODE actions (subgraph)
    # TODO: Find a proper filtering of reification and other macros
    macros_write = writer(f'{out_basename}.mcr', add_return=True)

    for sent in tqdm(sentences):
        for labeled_token in sent:
            if sense_regex.match(labeled_token):
                # node with sense
                # extract label and token
                token, sense = sense_regex.match(labeled_token).groups()
                token_by_sense[sense].update([token])
                # update writers
                wsd_write(labeled_token)
                macros_write(f'{token} O')

            elif pred_regex.match(labeled_token):                
                # node with lemma (ignored)
                token, lemma = pred_regex.match(labeled_token).groups()
                token_by_lemma[lemma].update([token])
                # update writers
                wsd_write(f'{token} O')
                macros_write(f'{token} O')

            elif addnode_regex.match(labeled_token):
                # subgraph from addnode
                token, addnode = addnode_regex.match(labeled_token).groups()
                token_by_addnode[addnode].update([token]) 
                # update writers
                wsd_write(f'{token} O')
                macros_write(labeled_token)
 
            elif blank_regex.match(labeled_token):    
                # update writers
                wsd_write(labeled_token)
                macros_write(labeled_token)
            else:
                import ipdb; ipdb.set_trace(context=30)
                print()

        # Add blank
        wsd_write('')
        macros_write('')

    # 'volume-quantity'
    # 'value-interval'
    # 'thing,rate-entity-91,temporal-quantity'

    # Close writers
    wsd_write()
    macros_write()
