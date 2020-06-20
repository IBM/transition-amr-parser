import os
import sys

LDC_dir = sys.argv[1]

dirs_and_outputs = [('training','train.txt'),
                    ('dev','dev.txt'),
                    ('test','test.txt')]


def merge_dir(dir, outfile):
    amrs = []
    for filename in sorted(os.listdir(dir)):
        if not filename.startswith("amr"):
            continue
        with open(os.path.join(dir,filename), 'r', encoding='utf8') as f:
            print(filename)
            for i,line in enumerate(f):
                if i in [0,1]:
                    continue
                if line.startswith('# ::align'):
                    continue
                amrs.append(line)
            amrs.append('\n')
    amrs = ''.join(amrs)
    amrs = amrs.replace('\r','')
    amrs = amrs.replace('\n\n\n','\n\n')
    amrs = amrs.replace('\u0092',"'")
    amrs = amrs.replace('\u0085'," ")

    with open(outfile,'w+', encoding='utf8') as f:
        f.write(amrs)
        print(amrs.count('# ::snt'))


for dir, output_file in dirs_and_outputs:
    dir1 = os.path.join(LDC_dir,'data','amrs','split',dir)
    if "2014" in LDC_dir:
        dir1 = os.path.join(LDC_dir,'data','split',dir)
    merge_dir(dir1,output_file)
