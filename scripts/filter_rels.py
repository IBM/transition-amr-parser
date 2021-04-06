from tqdm import tqdm
import sys


def count_amrs(in_amr_file):
    num_amrs = 0
    with open(in_amr_file) as fid:
        for line in fid:
            if line.strip() == '':
                num_amrs += 1
    return num_amrs


def penman_generator(in_amr_file, parse=False):
    with open(in_amr_file) as fid:
        amr_annotation = []
        for line in fid:
            if line.strip() == '':
                yield amr_annotation
                amr_annotation = []
            else:
                amr_annotation.append(line)


if __name__ == '__main__':

    # argument handling
    in_amr_file, out_amr_file = sys.argv[1:]

    # count number of AMRs
    print('Counting AMRs')
    num_amrs = count_amrs(in_amr_file)

    num_amrs_out = 0
    with open(out_amr_file, 'w') as fid_out:
        for penman in tqdm(penman_generator(in_amr_file), total=num_amrs):
            skip = False
            for line in penman:
                if line.startswith('# ::edge') and '\trel\t' in line:
                    skip = True
                    break
            if not skip:
                num_amrs_out += 1
                fid_out.write(''.join(penman) + '\n')

    print(f'{num_amrs - num_amrs_out}/{num_amrs} filtered')
