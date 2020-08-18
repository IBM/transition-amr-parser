import re
from tqdm import tqdm
import sys
from collections import defaultdict, Counter
from transition_amr_parser.io import writer


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


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


def get_filtered_labeled_token(labeled_token):
    '''
    Given <token label> , filter out labels not satisfying regex
    '''

    if sense_regex.match(labeled_token):
        # node with sense
        # extract label and token
        token, sense = sense_regex.match(labeled_token).groups()
        token_by_sense[sense].update([token])
        # labeled tokens for each task
        wsd_labeled_token = labeled_token
        mcr_labeled_token = f'{token} O'
        ner_labeled_token = f'{token} O'
        
    elif pred_regex.match(labeled_token):                
        # node with lemma (ignored)
        token, lemma = pred_regex.match(labeled_token).groups()
        token_by_lemma[lemma].update([token])
        # labeled tokens for each task
        wsd_labeled_token = f'{token} O'
        mcr_labeled_token = f'{token} O'
        ner_labeled_token = f'{token} O'

    elif addnode_regex.match(labeled_token):
        # subgraph from addnode
        token, addnode = addnode_regex.match(labeled_token).groups()
        token_by_addnode[addnode].update([token]) 
        # labeled tokens for each task
        wsd_labeled_token = f'{token} O'
        mcr_labeled_token = labeled_token
        if ',name' in addnode:
            ner_labeled_token = labeled_token
        else:    
            ner_labeled_token = f'{token} O'
 
    elif blank_regex.match(labeled_token):    
        # labeled tokens for each task
        wsd_labeled_token = labeled_token
        mcr_labeled_token = labeled_token
        ner_labeled_token = labeled_token

    else:
        import ipdb; ipdb.set_trace(context=30)
        print()

    return wsd_labeled_token, mcr_labeled_token, ner_labeled_token


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
    macros_write = writer(f'{out_basename}.mcr', add_return=True)
    ner_write = writer(f'{out_basename}.ner', add_return=True)

    # Filtered sentences
    filtered_wsd = []
    filtered_mcr = []
    filtered_ner = []

    sentence_index = 0
    for sent in tqdm(sentences):

        # labeled tokens for the subtasks
        wsd_sentence = []
        mcr_sentence = []
        ner_sentence = []
        for labeled_token in sent:

            # Given <token label> , filter out labels not satisfying regex
            wsd_labeled_token, mcr_labeled_token, ner_labeled_token = \
                get_filtered_labeled_token(labeled_token)

            # Append labeled token
            wsd_sentence.append(wsd_labeled_token)
            mcr_sentence.append(mcr_labeled_token)
            ner_sentence.append(ner_labeled_token)

        # write to disk, but filter out sentences that have no annotations 
        # (due to the filtering above)
        if any([w.split()[-1] != 'O' for w in wsd_sentence]):
            wsd_write('\n'.join(wsd_sentence) + '\n')
        else:
            filtered_wsd.append(sentence_index)
        if any([w.split()[-1] != 'O' for w in mcr_sentence]):
            macros_write('\n'.join(mcr_sentence) + '\n')
        else:
            filtered_mcr.append(sentence_index)
        if any([w.split()[-1] != 'O' for w in ner_sentence]):
            ner_write('\n'.join(ner_sentence) + '\n')
        else:
            filtered_ner.append(sentence_index)
        sentence_index += 1

    # Close writers
    wsd_write()
    macros_write()
    ner_write()

    # warn of filtered sentences
    if filtered_wsd:
        num_sents = len(sentences)
        num_labeled = num_sents - len(filtered_wsd)
        message = f'{num_labeled}/{num_sents} WSD sentences with labels'
        print(yellow_font(message))
    if filtered_mcr:
        num_sents = len(sentences)
        num_labeled = num_sents - len(filtered_mcr)
        message = f'{num_labeled}/{num_sents} MCR sentences with labels'
        print(yellow_font(message))
    if filtered_ner:
        num_sents = len(sentences)
        num_labeled = num_sents - len(filtered_ner)
        message = f'{num_labeled}/{num_sents} NER sentences with labels'
        print(yellow_font(message))
