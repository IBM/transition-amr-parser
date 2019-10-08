type = ['train','dev','test']

for t in type:
    amr_file = t +'.merged.txt'
    new_amr_file = t +'.merged.txt'


    amrs = open(amr_file, 'r', encoding='utf8').read()

    amrs = amrs.replace('Qaid\tba-mushaqqat','Qaid ba-mushaqqat')
    l = amrs.count('# ::snt')
    with open(new_amr_file, 'w+', encoding='utf8') as f:
        f.write(amrs)
    print(new_amr_file)
    print(l)

