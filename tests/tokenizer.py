import sys
import re
from transition_amr_parser.io import protected_tokenizer
from random import shuffle


if __name__ == '__main__':

    #do_break = True
    do_break = False

    # AMR file with ::snt and ::tok fields (JAMR)
    amr_file = sys.argv[1]

    tokens = []
    sents = []
    with open(amr_file) as fid:
        for line in fid:
            if line.strip().startswith('# ::snt'):
                sents.append(line.split('# ::snt')[-1].strip())
            elif line.strip().startswith('# ::tok'):
                tokens.append(line.split('# ::tok')[-1].strip())

    assert len(tokens) == len(sents)

    indices = list(range(len(tokens)))
    shuffle(indices)

    count = 0
    for index in indices:

        #if index != 4836:
        #    continue

        new_tokens = ' '.join(protected_tokenizer(sents[index])[0])

        if tokens[index] == new_tokens:
            count += 1
        elif do_break: # and index not in [
#             384, 385, 973, 1541,
#             669,                 # 'a
#             865,                 # 120.
#             1335,                # gov't
#             1411,                # !!!)
#             1520,                # PA.
#         ]:
            print(index)
            print(sents[index])
            print(tokens[index])
            print(new_tokens)
            import ipdb; ipdb.set_trace(context=30)
            protected_tokenizer(sents[index])

    perc = count * 100. / len(tokens)
    print(f'match {count}/{len(tokens)} {perc:.2f} %')
