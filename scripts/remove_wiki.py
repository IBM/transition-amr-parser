import re
import sys


if __name__ == '__main__':

    # argument handling
    amr_file, new_amr_file = sys.argv[1:]

    with open(amr_file, encoding='utf-8') as fid:
        amrs = fid.read()
    amrs = re.sub(':wiki ".+?"( )?','', amrs)
    amrs = re.sub(':wiki -( )?','', amrs)
    l = amrs.count('# ::snt')
    with open(new_amr_file, 'w+', encoding='utf-8') as f:
        f.write(amrs)
    print(new_amr_file)
    print(l)
