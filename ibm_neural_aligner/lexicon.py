import collections
import json
import os

from amr_utils import safe_read as safe_read_
from tqdm import tqdm

def safe_read(path, **kwargs):
    kwargs['ibm_format'], kwargs['tokenize'] = True, False
    return safe_read_(path, **kwargs)

#path_old = './DATA/AMR2.0/aligned/cofill/dev.txt'
#path_neu = './tmp_out/dev.aligned.txt'
path_old = './DATA/AMR2.0/aligned/cofill/train.txt'
path_neu = './tmp_out/train.aligned.txt'

def build_lexicon(path):
    amrs = safe_read(path)
    lexicon = collections.defaultdict(collections.Counter)

    for amr in tqdm(amrs):
        for node_id, text_id_list in amr.alignments.items():
            if len(text_id_list) > 1:
                continue

            for text_id in text_id_list:
                text = amr.tokens[text_id]
                node = amr.nodes[node_id]

            lexicon[node][text] += 1

    return lexicon

lex_old = build_lexicon(path_old)
lex_neu = build_lexicon(path_neu)

def compare_lexicon(lex_old, lex_neu):
    for node, lex in sorted(lex_old.items(), key=lambda x: len(x[1])):

        row_old = []
        for text in sorted(lex_old[node].keys()):
            row_old.append(text)

        row_neu = []
        for text in sorted(lex_neu[node].keys()):
            row_neu.append(text)

        print(node)
        print('old', row_old)
        print('neu', row_neu)
        print('')

def compare_lexicon_stats(lex_old, lex_neu):
    threshold = 10
    stats = collections.Counter()
    for node, lex in sorted(lex_old.items(), key=lambda x: len(x[1])):

        row_old = []
        for text in sorted(lex_old[node].keys()):
            row_old.append(text)

        row_neu = []
        for text in sorted(lex_neu[node].keys()):
            row_neu.append(text)

        stats['total'] += 1

        if len(row_old) >= len(row_neu):
            stats['old >= neu'] += 1
        else:
            print(f'old < neu, {len(row_old)} - {len(row_neu)} = {len(row_old) - len(row_neu)}')
            print(node)
            print('old', row_old)
            print('neu', row_neu)
            print('')

        if len(row_old) == len(row_neu):
            stats['old == neu'] += 1

        if len(row_old) <= threshold:
            stats['old <= t'] += 1

        if len(row_neu) <= threshold:
            stats['neu <= t'] += 1
        else:
            print(f'neu > t, {len(row_old)} - {len(row_neu)} = {len(row_old) - len(row_neu)}')
            print(node)
            print('old', row_old)
            print('neu', row_neu)
            print('')

    for k, v in stats.items():
        if k == 'total':
            continue

        n = stats['total']
        print(f'{k} : {v} / {n} ({v/n:.3f})')

def view_lexicon(lexicon):
    for node, lex in sorted(lexicon.items(), key=lambda x: len(x[1])):

        row = []
        for text in sorted(lexicon[node].keys()):
            row.append(text)

        print('{} {}'.format(node, row))

compare_lexicon(lex_old, lex_neu)
compare_lexicon_stats(lex_old, lex_neu)

