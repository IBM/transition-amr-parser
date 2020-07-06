import sys


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        fname = 'oracles/o7+Word100/dev.actions'
    else:
        fname = sys.argv[1]
    f = open(fname, 'r')
    lines = [line.strip().split() for line in f if line.strip()]
    lens = [len(line) for line in lines]
    len_avg = sum(lens) / len(lens)
    print(f'Average length of sequence: {len_avg}')
    f.close()
