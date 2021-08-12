import sys
import re
from transition_amr_parser.io import protected_tokenizer
from random import shuffle


def read_raw_amr(amr_file):

    # AMR file with ::snt and ::tok fields (JAMR)
    tokens = []
    sents = []
    with open(amr_file) as fid:
        for line in fid:
            if line.strip().startswith('# ::snt'):
                sents.append(line.split('# ::snt')[-1].strip())
            elif line.strip().startswith('# ::tok'):
                tokens.append(line.split('# ::tok')[-1].strip())
    assert len(tokens) == len(sents)
    return sents, tokens


def main(amr_file, do_break=False):

#     # indices to ignore
#     ignore_indices =[
#         384, 385, 973, 1541,
#         669,                  # 'a
#         865,                  # 120.
#         1335,                 # gov't
#         1411,                 # !!!)
#         1520,                 # PA.
#     ]
    ignore_indices = []

    # read data
    sents, tokens = read_raw_amr(amr_file)

    # random order
    indices = list(range(len(tokens)))
    shuffle(indices)

    # simple tokenizer
    count = 0
    for index in indices:
        new_tokens = ' '.join(protected_tokenizer(sents[index], simple=True)[0])
        if tokens[index] == new_tokens:
            count += 1
        elif do_break and index not in ignore_indices:
            print(index)
            print(sents[index])
            print(tokens[index])
            print(new_tokens)
            import ipdb; ipdb.set_trace(context=30)
            protected_tokenizer(sents[index])

    perc = count * 100. / len(tokens)
    print(f'simple match {count}/{len(tokens)} {perc:.2f} %')

    # JAMR like tokenizer
    count = 0
    for index in indices:
        new_tokens = ' '.join(protected_tokenizer(sents[index])[0])

        if tokens[index] == new_tokens:
            count += 1
        elif do_break and index not in ignore_indices:
            print(index)
            print(sents[index])
            print(tokens[index])
            print(new_tokens)
            import ipdb; ipdb.set_trace(context=30)
            protected_tokenizer(sents[index])

    perc = count * 100. / len(tokens)
    print(f'JAMR-like match {count}/{len(tokens)} {perc:.2f} %')


if __name__ == '__main__':
    main(sys.argv[1], do_break=False)
