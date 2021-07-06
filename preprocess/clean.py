import sys

if __name__ == '__main__':
    amr_file = sys.argv[1] 
    new_amr_file = amr_file
    amrs = open(amr_file, 'r', encoding='utf8').read()
    amrs = amrs.replace('Qaid\tba-mushaqqat','Qaid ba-mushaqqat')
    l = amrs.count('# ::snt')
    with open(new_amr_file, 'w+', encoding='utf8') as f:
        f.write(amrs)
    print(new_amr_file)
    print(l)
