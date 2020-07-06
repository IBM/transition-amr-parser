import re


type = ['train','dev','test']

for t in type:
    amr_file = t +'.txt'
    new_amr_file = t +'.no_wiki.txt'


    amrs = open(amr_file, 'r', encoding='utf8').read()

    amrs = re.sub(':wiki ".+?"( )?','', amrs)
    amrs = re.sub(':wiki -( )?','', amrs)
    l = amrs.count('# ::snt')
    with open(new_amr_file, 'w+', encoding='utf8') as f:
        f.write(amrs)
    print(new_amr_file)
    print(l)

