import re


type = ['train','dev','test']

for t in type:
    amr_file = t +'.kevin.txt'
    new_amr_file = t +'.kevin.txt'
    links_file = t+'.bad_amrs.txt'
    with open(links_file, 'r', encoding='utf8') as f:
        links = [line.split('\t') for line in f.readlines() if line.strip()]

    amrs = open(amr_file, 'r', encoding='utf8').readlines()

    amrs = [amr for amr in amrs if amr.strip() and not amr.startswith('#')]
    for sent_idx, toks, amr in links:
        amrs.insert(int(sent_idx),amr)
    l = len(amrs)
    amrs = ''.join(amrs)
    with open(new_amr_file, 'w+', encoding='utf8') as f:
        f.write(amrs)
    print(new_amr_file)
    print(l)

