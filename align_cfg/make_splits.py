import argparse
import json
import os
import collections
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', default=os.path.expanduser('~/data/AMR2.0/aligned/cofill/train.txt'), type=str)
args = parser.parse_args()

def readfile(path):
    data = []
    b = None
    with open(path) as f:
        for line in f:
            if line.strip():
                if b is None:
                    b = ''
                b += line
            else:
                if b is not None:
                    data.append(b)
                b = None
        if b is not None:
            data.append(b)
    return data

def writefile(data, path):
    print('writing', path)
    with open(path, 'w') as f:
        for b in data:
            f.write(b)
            f.write('\n')


# read
data = readfile(args.input)
print(len(data))

# shuffle
np.random.seed(113)
np.random.shuffle(data)

# split
n = 1000

# train
train = data[n:]

# unseen dev
unseen = data[:n]

# seen dev
seen = train[:n]

# write
writefile(train, args.input + '.train-v1')
writefile(unseen, args.input + '.dev-unseen-v1')
writefile(seen, args.input + '.dev-seen-v1')
