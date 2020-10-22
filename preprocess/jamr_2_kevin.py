import os
import sys
import re


def merge_dir_kevin(file):
    amrs = []
    toks = []
    with open(file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i in [0, 1]:
                continue
            if line.startswith('# ::tok'):
                toks.append(line[len('# ::tok '):])
            if line.startswith('#'):
                continue
            amrs.append(line)
        amrs.append('\n')
    amrs = ''.join(amrs)
    amrs = amrs.replace('\r', '')
    amrs = amrs.replace('\n\n\n', '\n\n')
    amrs = amrs.replace('\n\n', '\n')
    amrs = re.sub('\n[ ]+', ' ', amrs)
    amrs = amrs.replace('  ', ' ')
    amrs = [amr+'\n' for amr in amrs.split('\n') if amr.strip()]

    # filter out links
    sent_idx = 0
    bad_indices = []
    for amr, tok in zip(amrs,toks):
        if tok.startswith('< a href =') or tok.strip()=='.':
            bad_indices.append(sent_idx)
        sent_idx += 1
    with open(file.replace('.jamr', '.bad_amrs'), 'w+', encoding='utf8') as f:
        for sent_idx in bad_indices:
            f.write(str(sent_idx) + '\t'+ toks[sent_idx].strip() + '\t'+ amrs[sent_idx])
    amrs = ''.join(amr for i,amr in enumerate(amrs) if i not in bad_indices)
    toks = ''.join(tok for i, tok in enumerate(toks) if i not in bad_indices)
    with open(file.replace('.jamr','.amrs'), 'w+', encoding='utf8') as f:
        f.write(amrs)
    with open(file.replace('.jamr','.sents'), 'w+', encoding='utf8') as f:
        f.write(toks)
    print(file.replace('.jamr', '.amrs'))
    print(file.replace('.jamr', '.sents'))
    print(file.replace('.jamr', '.bad_amrs'))
    print(amrs.count('\n'),'+',len(bad_indices))


if __name__ == '__main__':
    for output in sys.argv[1:]:
        merge_dir_kevin(output)
