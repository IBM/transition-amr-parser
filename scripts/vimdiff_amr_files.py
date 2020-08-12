import sys
import subprocess


def get_one_amr(fid):
    amr = []
    line = fid.readline()
    while line.strip():
        amr.append(line)
        line = fid.readline()
    return amr


def write(file_name, content):
    with open(file_name, 'w') as fid:
        fid.write(content)
        

if __name__ == '__main__':
    
    amr1_file, amr2_file = sys.argv[1:]

    different_amrs = []
    num_amrs = 0
    with open(amr1_file) as fid1, open(amr2_file) as fid2:
        while True:
            amr1 = get_one_amr(fid1)
            amr2 = get_one_amr(fid2)
            penman1 = ''.join([x for x in amr1 if x[0] != '#'])
            penman2 = ''.join([x for x in amr2 if x[0] != '#'])
            if penman1 != penman2:
                different_amrs.append((num_amrs, penman1, penman2))
            num_amrs += 1
            print(f'\r{num_amrs}', end='')
            if amr1 == [] and amr2 == []:
                break

    print(f'\n{len(different_amrs)}/{num_amrs} different AMRs')

    for n, p1, p2 in different_amrs:
        input(f'\nPress any key to compare sentence {n}')
        write('tmp1', p1)
        write('tmp2', p2)
        subprocess.call(['vimdiff', 'tmp1', 'tmp2'])        
