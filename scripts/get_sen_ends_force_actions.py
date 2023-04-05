import glob
import argparse
from transition_amr_parser.amr_machine import make_eos_force_actions


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='given a file containing sentences of a doc, this function gives the \
          force actions and the unicode removed version of the file to be used by amr-parse'
    )
    parser.add_argument(
        '--in-file',
        type=str,
        help='Input doc text file containing sentence per line and new line at the end of every doc'
    )
    args = parser.parse_args()

    return args


def get_force_actions(in_file, sen_ends=None):

    sents = []
    docs = []
    for line in open(in_file, 'r').readlines():
        if line == '\n':
            docs.append(sents)
            sents = []
        else:
            sents.append(line.strip('\n'))

    dataset_utf = open(in_file+'.refined', 'w')
    force_action_file = open(in_file+'.force_actions', 'w')
    if sen_ends is None:
        sen_ends = []
    tokens = []

    for docsen in docs:

        offset = 0
        newsens = []
        for sen in docsen:
            enc = sen.rstrip().encode("ascii", "ignore")

            newsen = enc.decode()

            newsens.append(newsen)

            tok = newsen.split()
            tokens.extend(tok)
            sen_ends.append(offset+len(tok)-1)
            offset += len(tok)

        force_actions = make_eos_force_actions(tokens, sen_ends)
        force_action_file.write(str(force_actions)+'\n')
        dataset_utf.write(' '.join(newsens)+'\n')
        


if __name__ == '__main__':
    args = argument_parsing()
    get_force_actions(args.in_file)
