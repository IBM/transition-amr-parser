import os
import sys

def merge_dir(dir, outfile):

    # collect amrs
    amrs = []
    for filename in sorted(os.listdir(dir)):
        if not filename.startswith("amr"):
            continue
        with open(os.path.join(dir, filename), encoding='utf-8') as f:
            print(filename)
            for i,line in enumerate(f):
                if i in [0, 1]:
                    continue
                if line.startswith('# ::align'):
                    continue
                amrs.append(line)
            amrs.append('\n')

    # normalization
    amrs = ''.join(amrs)
    amrs = amrs.replace('\r','')
    amrs = amrs.replace('\n\n\n','\n\n')
    amrs = amrs.replace('\u0092',"'")
    amrs = amrs.replace('\u0085'," ")

    # write data
    with open(outfile,'w+', encoding='utf-8') as f:
        f.write(amrs)
        print(amrs.count('# ::snt'))

if __name__ == '__main__':
    input_dir, output_dir = sys.argv[1:]
    os.makedirs(output_dir, exist_ok=True)
    merge_dir(f'{input_dir}/training/', f'{output_dir}/train.txt')
    merge_dir(f'{input_dir}/dev/', f'{output_dir}/dev.txt')
    merge_dir(f'{input_dir}/test/', f'{output_dir}/test.txt')
